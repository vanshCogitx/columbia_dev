"""Evals system, Layer 2 — LLM-as-judge. Runs as a background task the moment
a response_feedback row is inserted (see routers/chat.py's post_feedback).

Stage B: draft a quality score for the (query, response) pair, then have the
judge critique and revise its own draft (self-refine) — both scores are
stored so the correction step's value is visible over time.

Stage C: only for bad verdicts (thumbs-down feedback, or a low corrected
score) — identifies which agent's prompt most plausibly caused the failure.
Cartesian's raw response (saved per-message in message_traces, see
cartesian.py's _save_message_trace) includes executionPath/agentRawResponses:
the exact list of agent nodes that ran for this specific response, and what
each one actually output. Stage C uses that to pull only those agents'
prompts from the active ingested workflow (see tools/ingest_workflow.py) —
typically ~8 of the ~12 total — plus each agent's real output alongside its
prompt, instead of guessing via the full library or a similarity search.
Falls back to the full library only if no trace was saved for this message
(e.g. it predates this feature).

The core scoring logic (_score_and_attribute) is source-agnostic: it takes a
query/response/rating/reason/raw_output directly rather than fetching by
feedback_id, so the same Stage B/C machinery also backs the "Testing" tab's
simulation-turn feedback (see run_simulation_judge_pipeline, and
simulation.py for the scenario runner itself).
"""
import os
import re
import json
import random
import asyncio
import logging
from typing import Optional

import httpx

import db

logger = logging.getLogger("columbia_backend.judge")

# Retry policy for transient failures (429/5xx/timeout/mid-stream error) —
# honor Retry-After when present, otherwise exponential backoff with jitter.
RETRY_MAX_ATTEMPTS = 3
RETRY_INITIAL_BACKOFF = 1.0
RETRY_MAX_BACKOFF = 60.0

# NVIDIA's own direct inference API (build.nvidia.com), not routed through
# NVIDIA's own direct inference API (build.nvidia.com), not routed through
# OpenRouter. Originally pointed at Nemotron 3 Ultra (550B) for its 1M-token
# context, but a direct curl test showed Ultra's free-tier capacity is
# severely overloaded right now — zero response even after 90s. Nemotron 3
# Super (120B, same reasoning family — still streams to 'reasoning_content'
# the same way, so the dashboard's live thinking view keeps working)
# responded instantly in the same test. Switched to Super for that reason;
# re-verify it has enough context headroom for Stage C's ~25k-token prompt
# library specifically (should be fine — 25k tokens is modest by modern
# standards — but hasn't been tested at that size yet).
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"

BAD_SCORE_THRESHOLD = 0.5

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

# Shared grounding rule, appended to every judge prompt. Without this, the
# model will sometimes state an inference (e.g. guessing a price's currency
# from a URL's country domain) as if it were a verified fact, when nothing
# in the actual data confirms it — a real hallucination that can flip a
# verdict for the wrong reason. Caught via manual review of a real judgment.
GROUNDING_RULE = """
GROUNDING RULE (critical): Only state something as fact if it is directly present in the query, response, or prompts you were given. If you want to raise a possibility that isn't explicitly stated in the data (e.g. guessing a currency from a domain name, assuming a size chart, inferring intent not written anywhere) — you may mention it, but you MUST explicitly flag it as an unverified guess (e.g. "possibly...", "this cannot be confirmed from the given data, but..."), never state it as established fact, and never let an unverified guess be the primary reason for a score or verdict. If your reasoning leans on such a guess, say so explicitly."""

# Without this, the model's internal reasoning phase runs long enough to
# regularly blow past the wall-clock timeout before ever producing an
# answer — confirmed directly: draft/critique calls without this instruction
# took 80-90s each, and Stage C's large-context call never completed at all
# across 3 full retry attempts. A short, blunt instruction to stop
# overthinking is the cheapest lever available before resorting to smaller
# reasoning-token budgets alone.
BREVITY_RULE = """
SPEED RULE (critical): Think briefly. A short paragraph of internal reasoning is enough — never a step-by-step trace through every detail, and never re-read or re-summarize the full input back to yourself before answering. Reach a conclusion quickly. A fast, confident judgment beats a slow, exhaustive one — long reasoning risks never finishing at all."""

DRAFT_SYSTEM_PROMPT = """You are a strict quality judge for an AI shopping assistant's responses. Given a user's query and the assistant's response, score how well the response satisfies the query.

Score from 0.0 (completely fails to help) to 1.0 (excellent, fully satisfies the query).

Consider: relevance of any products shown, whether the response actually addresses what was asked, correctness, and whether the tone/content is appropriate. If the user provided their own feedback reason, weigh it heavily but verify it against the actual response yourself rather than blindly trusting it — the user's own stated reason can also be imprecise or overstated, so check it against the real response text.
""" + GROUNDING_RULE + BREVITY_RULE + """

Output ONLY raw JSON, no markdown fences, no commentary:
{"score": <float 0-1>, "reasoning": "<why this score, specifically what's good or bad>"}"""

CRITIQUE_SYSTEM_PROMPT = """You are reviewing your own prior quality judgment of an AI shopping assistant's response, to catch mistakes or blind spots before finalizing it.

You will be shown the original query, response, and your own draft score and reasoning. Critically re-examine your draft: did you miss anything, misjudge relevance, or reason too shallowly? Also specifically check whether your draft reasoning stated any unverified guess as fact — if so, correct it. Then give a corrected (possibly identical) score and reasoning.
""" + GROUNDING_RULE + BREVITY_RULE + """

Output ONLY raw JSON, no markdown fences, no commentary:
{"corrected_score": <float 0-1>, "corrected_reasoning": "<final reasoning, incorporating any correction>"}"""

ROOT_CAUSE_SYSTEM_PROMPT = """You are debugging a multi-agent AI pipeline that produced a bad response. You will be given the user's query, the bad response, an analysis of what's wrong with it, and — for each agent that actually ran in this specific pipeline execution — that agent's system prompt/instructions AND its actual raw output for this run.

Your job: identify which ONE agent's prompt is most likely responsible for this specific failure, quote the exact sentence (or short passage) from that agent's prompt you believe caused it, and explain the causal chain connecting that instruction to the observed failure. Use each agent's actual output as evidence — prefer pointing to an agent whose real output visibly went wrong (e.g. empty results, wrong field, missed constraint) over one that merely could have contributed in theory.

Only point to genuine, specific wording issues in a prompt — don't invent a cause if the prompts look reasonable; in that case say so honestly and note the failure may stem from the underlying model's execution rather than the prompt wording itself.

Be decisive and concise: skim the prompts for the one or two most plausible culprits based on the failure symptom, form a judgment, and answer immediately. Do NOT exhaustively trace step-by-step through every agent in the pipeline one by one, and do NOT quote or restate large chunks of each prompt back to yourself while thinking — skim for keywords relevant to the failure, jump straight to the likely culprit. This is a large input; exhaustive reasoning over it will not finish in time.
""" + GROUNDING_RULE + BREVITY_RULE + """

Output ONLY raw JSON, no markdown fences, no commentary:
{"root_cause_node": "<agent name, or null if inconclusive>", "root_cause_snippet": "<quoted sentence, or null>", "root_cause_explanation": "<causal reasoning, or explanation of why inconclusive>"}"""


def parse_json_response(text: str) -> dict:
    candidate = text.strip()
    fence_match = _CODE_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    # Nemotron is a reasoning model — its chain-of-thought sometimes lands in
    # 'content' itself instead of the dedicated 'reasoning' field, ahead of
    # the actual JSON answer. Fall back to the outermost {...} block in the
    # text rather than giving up immediately.
    first_brace = candidate.find("{")
    last_brace = candidate.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            obj = json.loads(candidate[first_brace:last_brace + 1])
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass
    logger.error("judge: failed to parse JSON from judge response: %r", text)
    return {}


async def _build_stage_c_library_from_raw_output(raw_output: Optional[dict], trace_label: str, project_id: int) -> str:
    """Builds Stage C's agent-prompt input, narrowed to only the agents that
    actually ran for this specific response — determined from Cartesian's own
    execution trace (output.agentRawResponses), not a guess. Each included
    agent gets both its prompt AND its real output for this run, so the judge
    has actual evidence instead of just prompt text. Falls back to every
    agent's prompt if raw_output is missing or has no usable agent list.
    Source-agnostic: raw_output may come from message_traces (real feedback)
    or a simulation_turns row (test feedback) — see callers below.
    trace_label is only for logging (e.g. "message_id=123" or "turn_id=456").
    project_id scopes the prompt library to the workflow that actually
    produced this response — see projects.py's module docstring."""
    node_outputs: dict[str, str] = {}
    if raw_output:
        for entry in raw_output.get("agentRawResponses") or []:
            node_id = entry.get("nodeId")
            text = entry.get("text")
            if node_id and isinstance(text, str):
                node_outputs[node_id] = text

    if node_outputs:
        prompt_rows = await db.db_pool.fetch(
            """
            SELECT wn.node_id, wn.alias, wn.agent_instructions FROM workflow_nodes wn
            JOIN workflow_versions wv ON wv.id = wn.workflow_version_id
            WHERE wv.is_active = true AND wv.project_id = $1
              AND wn.node_type = 'agent' AND wn.agent_instructions IS NOT NULL
              AND wn.node_id = ANY($2::text[])
            """,
            project_id, list(node_outputs.keys()),
        )
        if prompt_rows:
            logger.info(
                "judge: Stage C narrowed to %d/%d agents that actually ran (%s): %s",
                len(prompt_rows), len(node_outputs), trace_label, [r["alias"] for r in prompt_rows],
            )
            return "\n\n".join(
                f"--- {r['alias']} ---\nPROMPT:\n{r['agent_instructions']}\n\n"
                f"ACTUAL OUTPUT FOR THIS RUN:\n{node_outputs.get(r['node_id'], '(no output captured)')}"
                for r in prompt_rows
            )
        logger.warning(
            "judge: Stage C trace found agents %s (%s) but none matched workflow_nodes — falling back to full library",
            list(node_outputs.keys()), trace_label,
        )

    logger.warning(
        "judge: Stage C found no usable execution trace (%s) — falling back to full prompt library",
        trace_label,
    )
    prompt_rows = await db.db_pool.fetch(
        """
        SELECT wn.alias, wn.agent_instructions FROM workflow_nodes wn
        JOIN workflow_versions wv ON wv.id = wn.workflow_version_id
        WHERE wv.is_active = true AND wv.project_id = $1
          AND wn.node_type = 'agent' AND wn.agent_instructions IS NOT NULL
        """,
        project_id,
    )
    return "\n\n".join(f"--- {r['alias']} ---\n{r['agent_instructions']}" for r in prompt_rows)


async def _build_stage_c_library(message_id: int, project_id: int) -> str:
    """Real-feedback wrapper: fetches raw_output from message_traces, then
    delegates to the source-agnostic core above."""
    trace = await db.db_pool.fetchrow(
        "SELECT raw_output FROM message_traces WHERE message_id = $1", message_id
    )
    raw_output = None
    if trace and trace["raw_output"]:
        raw_output = json.loads(trace["raw_output"]) if isinstance(trace["raw_output"], str) else trace["raw_output"]
    return await _build_stage_c_library_from_raw_output(raw_output, f"message_id={message_id}", project_id)


STREAM_TTL_SECONDS = 3600  # cache hygiene only — response_judgments/simulation_turn_judgments (Postgres) are the real source of truth

# Redis Stream key prefixes — one per kind of live-progress consumer on the
# evals dashboard. Real feedback keeps the original "judge_stream" prefix
# unchanged so routers/evals.py's existing SSE endpoint needs no changes.
STREAM_PREFIX_FEEDBACK = "judge_stream"
STREAM_PREFIX_SIMULATION_JUDGE = "sim_judge_stream"
STREAM_PREFIX_SIM_RUN = "sim_run_stream"


def _stream_key(prefix: str, entity_id: int) -> str:
    return f"{prefix}:{entity_id}"


async def _publish_stream_event(stream_key: str, event: dict) -> None:
    """Appends one event to this Redis Stream, for the evals dashboard's
    live SSE views (see routers/evals.py, routers/testing.py). Uses a
    Stream, not Pub/Sub, specifically so a client that connects a moment
    after the pipeline started doesn't miss the opening events — Streams
    retain history for late subscribers, Pub/Sub does not."""
    await db.redis_client.xadd(stream_key, {"data": json.dumps(event)})
    await db.redis_client.expire(stream_key, STREAM_TTL_SECONDS)


# Public aliases for cross-module use (simulation.py). Internal call sites in
# this module keep the underscore-prefixed names, since "stream_key" is also
# used extensively here as a local variable holding the computed string,
# which would collide with a public function of the same name.
make_stream_key = _stream_key
publish_stream_event = _publish_stream_event


async def _call_judge(
    system_prompt: str, user_prompt: str, stream_key: str, stage: str,
    max_tokens: int = 1200, reasoning_max_tokens: int = 800, wall_clock_timeout: float = 300,
) -> tuple[dict, str]:
    # Nemotron is a reasoning model — NVIDIA's own API enables its
    # chain-of-thought via 'chat_template_kwargs.enable_thinking' + a
    # 'reasoning_budget' (NOT OpenRouter's 'reasoning: {enabled, max_tokens}'
    # wrapper — different provider, different param shape), and streams it
    # back on 'delta.reasoning_content' (not 'delta.reasoning'). Streams
    # token-by-token (stream: true) so the evals dashboard can show the
    # model's thinking live, publishing each delta to this feedback event's
    # Redis Stream as it arrives. Returns (parsed_json, thinking_text).
    #
    # httpx's own `timeout=` only bounds each individual read — a stream that
    # keeps trickling a chunk every few seconds forever never trips it, so a
    # stalled/slow-drip response can hang indefinitely. asyncio.wait_for wraps
    # the whole call in a real wall-clock cap so that failure mode becomes a
    # normal TimeoutError instead of a task that just never returns (this is
    # exactly what happened to two real feedback events that got permanently
    # stuck in production before this fix).
    async def _do_stream() -> tuple[list[str], list[str]]:
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                NVIDIA_URL,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": NVIDIA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": reasoning_max_tokens,
                    "stream": True,
                },
                timeout=wall_clock_timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    # Some streaming chunks (e.g. metadata-only/final usage chunks)
                    # carry "choices": [] rather than omitting the key entirely —
                    # a plain .get("choices", [{}])[0] still IndexErrors on those
                    # since the default only applies when the key is missing.
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    # A rate-limit/error that hits *after* streaming has
                    # already started a 200 response never becomes an HTTP
                    # error — it can arrive as a normal-looking chunk with
                    # finish_reason: "error" instead (confirmed behavior on
                    # OpenRouter; kept here defensively since it's a cheap
                    # check either way). Without checking this explicitly,
                    # the loop just keeps waiting for more 'data:' lines that
                    # are never coming — a second, distinct way for a call to
                    # hang beyond the wall-clock-timeout case above.
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason == "error":
                        raise RuntimeError(f"NVIDIA API reported a mid-stream error (finish_reason=error) for stage '{stage}'")
                    delta = choices[0].get("delta", {})
                    reasoning_delta = delta.get("reasoning_content")
                    content_delta = delta.get("content")
                    if reasoning_delta:
                        thinking_parts.append(reasoning_delta)
                        await _publish_stream_event(stream_key, {"stage": stage, "type": "thinking", "delta": reasoning_delta})
                    if content_delta:
                        content_parts.append(content_delta)
                        await _publish_stream_event(stream_key, {"stage": stage, "type": "content", "delta": content_delta})
        return thinking_parts, content_parts

    # Retry transient failures with backoff, per OpenRouter's own guidance:
    # honor Retry-After when present (typical on 429s), otherwise exponential
    # backoff with jitter. Free-tier models are the least reliable tier by
    # design (heaviest throttling, underlying-provider congestion that
    # OpenRouter itself doesn't see) — most stalls/errors we've hit in
    # practice are transient, so a couple of retries clears the large
    # majority of them silently instead of surfacing as a failure.
    thinking_parts = content_parts = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            thinking_parts, content_parts = await asyncio.wait_for(_do_stream(), timeout=wall_clock_timeout)
            break
        except asyncio.TimeoutError as e:
            last_exc, retry_after = RuntimeError(f"judge stage '{stage}' exceeded its {wall_clock_timeout:.0f}s wall-clock limit"), None
        except httpx.HTTPStatusError as e:
            retryable = e.response.status_code in (429, 500, 502, 503, 504)
            if not retryable:
                raise
            last_exc = e
            retry_after = e.response.headers.get("Retry-After")
        except (httpx.TransportError, RuntimeError) as e:
            last_exc, retry_after = e, None

        if attempt == RETRY_MAX_ATTEMPTS - 1:
            raise last_exc
        delay = float(retry_after) if retry_after else min(RETRY_MAX_BACKOFF, RETRY_INITIAL_BACKOFF * (2 ** attempt))
        delay *= 1 + random.uniform(-0.2, 0.2)
        logger.warning(
            "judge: stage '%s' attempt %d/%d failed (%s), retrying in %.1fs",
            stage, attempt + 1, RETRY_MAX_ATTEMPTS, last_exc, delay,
        )
        await asyncio.sleep(delay)

    full_thinking = "".join(thinking_parts)
    parsed = parse_json_response("".join(content_parts))
    await _publish_stream_event(stream_key, {"stage": stage, "type": "stage_done"})
    return parsed, full_thinking


async def _score_and_attribute(
    query_text: str, response_text: str, rating: Optional[str], reason: Optional[str],
    raw_output: Optional[dict], stream_key: str, trace_label: str, project_id: int,
) -> dict:
    """The shared Stage B/C core, source-agnostic: caller supplies the
    query/response/rating/reason/raw_output directly rather than this
    function fetching them itself, so it backs BOTH run_judge_pipeline (real
    feedback, sourced from response_feedback/messages/message_traces) and
    run_simulation_judge_pipeline (test feedback, sourced from
    simulation_turn_feedback/simulation_turns) with zero duplicated judging
    logic. Returns a dict keyed exactly like response_judgments'/
    simulation_turn_judgments' columns, so callers can INSERT it as-is."""
    feedback_block = f"User feedback: thumbs {rating}" if rating else "User feedback: none (automated evaluation)"
    if reason:
        feedback_block += f", reason: {reason}"

    # Stage B — draft, then self-critique/revise
    draft, draft_thinking = await _call_judge(
        DRAFT_SYSTEM_PROMPT,
        f"User query: {query_text}\n\nAssistant response: {response_text}\n\n{feedback_block}",
        stream_key, "draft",
    )
    initial_score = draft.get("score")
    initial_reasoning = draft.get("reasoning", "")

    critique, critique_thinking = await _call_judge(
        CRITIQUE_SYSTEM_PROMPT,
        f"User query: {query_text}\n\nAssistant response: {response_text}\n\n{feedback_block}\n\n"
        f"Your draft score: {initial_score}\nYour draft reasoning: {initial_reasoning}\n\n"
        "Critique and correct your draft judgment.",
        stream_key, "critique",
    )
    corrected_score = critique.get("corrected_score", initial_score)
    corrected_reasoning = critique.get("corrected_reasoning", initial_reasoning)

    root_cause_node = root_cause_snippet = root_cause_explanation = attribution_thinking = None
    is_bad = rating == "down" or (
        isinstance(corrected_score, (int, float)) and corrected_score < BAD_SCORE_THRESHOLD
    )

    if is_bad:
        library = await _build_stage_c_library_from_raw_output(raw_output, trace_label, project_id)
        if library:
            attribution, attribution_thinking = await _call_judge(
                ROOT_CAUSE_SYSTEM_PROMPT,
                f"User query: {query_text}\n\nBad response: {response_text}\n\n{feedback_block}\n\n"
                f"Failure analysis: {corrected_reasoning}\n\nAGENT PROMPTS AND OUTPUTS:\n{library}",
                stream_key, "attribution",
                max_tokens=2500,
                # Narrowed to only the agents that actually ran (typically
                # ~8 of ~12, see _build_stage_c_library_from_raw_output)
                # instead of every agent's prompt every time — the prior
                # full-library input (~25k tokens) never finished across 3
                # full retry attempts regardless of model or reasoning
                # budget, because the bottleneck was input/prefill time, not
                # reasoning length.
                reasoning_max_tokens=1200,
                wall_clock_timeout=300,
            )
            root_cause_node = attribution.get("root_cause_node")
            root_cause_snippet = attribution.get("root_cause_snippet")
            root_cause_explanation = attribution.get("root_cause_explanation")
        else:
            logger.warning("judge: no active workflow prompt library found, skipping Stage C for %s", trace_label)

    return {
        "initial_score": initial_score,
        "corrected_score": corrected_score,
        "judge_reasoning": corrected_reasoning,
        "root_cause_node": root_cause_node,
        "root_cause_snippet": root_cause_snippet,
        "root_cause_explanation": root_cause_explanation,
        "draft_thinking": draft_thinking,
        "critique_thinking": critique_thinking,
        "attribution_thinking": attribution_thinking,
    }


def _pipeline_done_event(result: dict) -> dict:
    return {
        "type": "pipeline_done",
        "initial_score": result["initial_score"],
        "corrected_score": result["corrected_score"],
        "judge_reasoning": result["judge_reasoning"],
        "root_cause_node": result["root_cause_node"],
        "root_cause_snippet": result["root_cause_snippet"],
        "root_cause_explanation": result["root_cause_explanation"],
    }


async def run_judge_pipeline(feedback_id: int) -> None:
    if not NVIDIA_API_KEY:
        logger.warning("judge: NVIDIA_API_KEY not configured, skipping feedback_id=%s", feedback_id)
        return
    stream_key = _stream_key(STREAM_PREFIX_FEEDBACK, feedback_id)
    try:
        feedback = await db.db_pool.fetchrow(
            """
            SELECT rf.id, rf.chat_id, rf.message_id, rf.rating, rf.reason, rf.project_id
            FROM response_feedback rf
            WHERE rf.id = $1
            """,
            feedback_id,
        )
        if not feedback:
            logger.error("judge: feedback_id=%s not found", feedback_id)
            return

        ai_msg = await db.db_pool.fetchrow("SELECT text FROM messages WHERE id = $1", feedback["message_id"])
        if not ai_msg:
            logger.error("judge: message_id=%s not found for feedback_id=%s", feedback["message_id"], feedback_id)
            return
        response_text = ai_msg["text"]

        user_msg = await db.db_pool.fetchrow(
            "SELECT text FROM messages WHERE chat_id = $1 AND id < $2 AND sender = 'user' ORDER BY id DESC LIMIT 1",
            feedback["chat_id"], feedback["message_id"],
        )
        query_text = user_msg["text"] if user_msg else "(no preceding user query found)"

        trace = await db.db_pool.fetchrow(
            "SELECT raw_output FROM message_traces WHERE message_id = $1", feedback["message_id"]
        )
        raw_output = None
        if trace and trace["raw_output"]:
            raw_output = json.loads(trace["raw_output"]) if isinstance(trace["raw_output"], str) else trace["raw_output"]

        result = await _score_and_attribute(
            query_text, response_text, feedback["rating"], feedback["reason"],
            raw_output, stream_key, f"message_id={feedback['message_id']}", feedback["project_id"],
        )

        await db.db_pool.execute(
            """
            INSERT INTO response_judgments
                (feedback_id, initial_score, corrected_score, judge_reasoning,
                 root_cause_node, root_cause_snippet, root_cause_explanation,
                 draft_thinking, critique_thinking, attribution_thinking)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            feedback_id, result["initial_score"], result["corrected_score"], result["judge_reasoning"],
            result["root_cause_node"], result["root_cause_snippet"], result["root_cause_explanation"],
            result["draft_thinking"], result["critique_thinking"], result["attribution_thinking"],
        )
        logger.info(
            "judge: feedback_id=%s initial_score=%s corrected_score=%s root_cause_node=%s",
            feedback_id, result["initial_score"], result["corrected_score"], result["root_cause_node"],
        )
        await _publish_stream_event(stream_key, _pipeline_done_event(result))
    except Exception as e:
        # Detached background task — must not let an exception vanish silently
        # into an "exception was never retrieved" asyncio warning instead of
        # being visible anywhere. Also record it as a real row (not just a log
        # line) so the dashboard can tell "failed" apart from "still running"
        # instead of spinning forever — a row existing is what "judged" means
        # to the feed API.
        logger.error("judge: run_judge_pipeline failed for feedback_id=%s: %s", feedback_id, e)
        try:
            await db.db_pool.execute(
                "INSERT INTO response_judgments (feedback_id, error) VALUES ($1, $2)",
                feedback_id, str(e),
            )
        except Exception as insert_err:
            logger.error("judge: failed to record error row for feedback_id=%s: %s", feedback_id, insert_err)
        await _publish_stream_event(stream_key, {"type": "pipeline_error", "detail": str(e)})


async def run_simulation_judge_pipeline(turn_feedback_id: int) -> None:
    """Testing-tab counterpart to run_judge_pipeline: same shape, but scores
    a simulated test turn (simulation_turn_feedback/simulation_turns)
    instead of real feedback, writing to simulation_turn_judgments instead
    of response_judgments. Fully reuses _score_and_attribute — no duplicated
    judging logic. Triggered from routers/testing.py's turn-feedback
    endpoint the same way run_judge_pipeline is triggered from
    routers/chat.py's post_feedback (bare asyncio.create_task, fire-and-forget)."""
    if not NVIDIA_API_KEY:
        logger.warning("judge: NVIDIA_API_KEY not configured, skipping turn_feedback_id=%s", turn_feedback_id)
        return
    stream_key = _stream_key(STREAM_PREFIX_SIMULATION_JUDGE, turn_feedback_id)
    try:
        feedback = await db.db_pool.fetchrow(
            "SELECT id, turn_id, rating, reason FROM simulation_turn_feedback WHERE id = $1",
            turn_feedback_id,
        )
        if not feedback:
            logger.error("judge: turn_feedback_id=%s not found", turn_feedback_id)
            return

        turn = await db.db_pool.fetchrow(
            """
            SELECT t.simulated_user_text, t.ai_response_text, t.raw_output, s.project_id
            FROM simulation_turns t
            JOIN simulation_runs r ON r.id = t.run_id
            JOIN simulation_scenarios s ON s.id = r.scenario_id
            WHERE t.id = $1
            """,
            feedback["turn_id"],
        )
        if not turn:
            logger.error("judge: turn_id=%s not found for turn_feedback_id=%s", feedback["turn_id"], turn_feedback_id)
            return

        raw_output = None
        if turn["raw_output"]:
            raw_output = json.loads(turn["raw_output"]) if isinstance(turn["raw_output"], str) else turn["raw_output"]

        result = await _score_and_attribute(
            turn["simulated_user_text"], turn["ai_response_text"] or "",
            feedback["rating"], feedback["reason"], raw_output, stream_key,
            f"turn_id={feedback['turn_id']}", turn["project_id"],
        )

        await db.db_pool.execute(
            """
            INSERT INTO simulation_turn_judgments
                (turn_feedback_id, initial_score, corrected_score, judge_reasoning,
                 root_cause_node, root_cause_snippet, root_cause_explanation,
                 draft_thinking, critique_thinking, attribution_thinking)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            turn_feedback_id, result["initial_score"], result["corrected_score"], result["judge_reasoning"],
            result["root_cause_node"], result["root_cause_snippet"], result["root_cause_explanation"],
            result["draft_thinking"], result["critique_thinking"], result["attribution_thinking"],
        )
        logger.info(
            "judge: turn_feedback_id=%s initial_score=%s corrected_score=%s root_cause_node=%s",
            turn_feedback_id, result["initial_score"], result["corrected_score"], result["root_cause_node"],
        )
        await _publish_stream_event(stream_key, _pipeline_done_event(result))
    except Exception as e:
        logger.error("judge: run_simulation_judge_pipeline failed for turn_feedback_id=%s: %s", turn_feedback_id, e)
        try:
            await db.db_pool.execute(
                "INSERT INTO simulation_turn_judgments (turn_feedback_id, error) VALUES ($1, $2)",
                turn_feedback_id, str(e),
            )
        except Exception as insert_err:
            logger.error("judge: failed to record error row for turn_feedback_id=%s: %s", turn_feedback_id, insert_err)
        await _publish_stream_event(stream_key, {"type": "pipeline_error", "detail": str(e)})


SIMULATION_VERDICT_SYSTEM_PROMPT = """You are evaluating whether a simulated test conversation between a shopper and an AI shopping assistant satisfied a test scenario's success criteria.

You will be given the scenario description, a list of success criteria, and the full conversation transcript (each simulated shopper message paired with the assistant's response). Judge each criterion against what actually happened in the transcript, then decide whether the conversation as a whole passes.
""" + GROUNDING_RULE + BREVITY_RULE + """

Output ONLY raw JSON, no markdown fences, no commentary:
{"verdict": "pass" or "fail", "reasoning": "<brief per-criterion breakdown>"}"""


async def run_simulation_verdict(scenario_description: str, success_criteria: list[str], transcript: str, stream_key: str) -> dict:
    """Called once by simulation.py at the end of a run's turn loop, over the
    full transcript — distinct from run_simulation_judge_pipeline, which
    scores one turn at a time only when a human clicks thumbs up/down on it.
    Reuses _call_judge exactly like Stage B/C (same retry/backoff/streaming/
    parsing), publishing to the same stream_key the run's turn-lifecycle
    events already used (stage="verdict") so the frontend's one SSE
    connection shows both in one place."""
    criteria_block = "\n".join(f"- {c}" for c in success_criteria)
    verdict, verdict_thinking = await _call_judge(
        SIMULATION_VERDICT_SYSTEM_PROMPT,
        f"SCENARIO: {scenario_description}\n\nSUCCESS CRITERIA:\n{criteria_block}\n\nTRANSCRIPT:\n{transcript}",
        stream_key, "verdict",
        max_tokens=1500, reasoning_max_tokens=800, wall_clock_timeout=300,
    )
    return {
        "verdict": verdict.get("verdict"),
        "verdict_reasoning": verdict.get("reasoning"),
        "verdict_thinking": verdict_thinking,
    }
