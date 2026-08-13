"""Evals system, Layer 2 — LLM-as-judge. Runs as a background task the moment
a response_feedback row is inserted (see routers/chat.py's post_feedback).

Stage B: draft a quality score for the (query, response) pair, then have the
judge critique and revise its own draft (self-refine) — both scores are
stored so the correction step's value is visible over time.

Stage C: only for bad verdicts (thumbs-down feedback, or a low corrected
score) — hands the judge every agent's actual prompt from the active
ingested workflow (see tools/ingest_workflow.py) and asks it to identify
which one's wording most plausibly caused the failure. We can't isolate
which single agent actually ran for a given response (Cartesian doesn't
expose a per-node execution trace), so this reasons over the full prompt
library rather than a narrowed-down subset.
"""
import os
import re
import json
import random
import asyncio
import logging

import httpx

import db

logger = logging.getLogger("columbia_backend.judge")

# Retry policy for transient OpenRouter failures (429/5xx/timeout/mid-stream
# error), per OpenRouter's own documented guidance: honor Retry-After when
# present, otherwise exponential backoff with jitter.
RETRY_MAX_ATTEMPTS = 3
RETRY_INITIAL_BACKOFF = 1.0
RETRY_MAX_BACKOFF = 60.0

# OpenRouter's free NVIDIA Nemotron 3 Ultra endpoint — chosen over calling
# Groq directly because its 1M-token context window and request-count-based
# (not tokens-per-minute) rate limiting let Stage C send the full ~25k-token
# agent prompt library in a single call, rather than needing to batch/paginate
# it (Groq's free tier caps at 12k TPM, well under what one Stage C call
# needs). Also a stronger model than Llama 3.3 70B for the reasoning-heavy
# root-cause attribution in Stage C.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

BAD_SCORE_THRESHOLD = 0.5

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

# Shared grounding rule, appended to every judge prompt. Without this, the
# model will sometimes state an inference (e.g. guessing a price's currency
# from a URL's country domain) as if it were a verified fact, when nothing
# in the actual data confirms it — a real hallucination that can flip a
# verdict for the wrong reason. Caught via manual review of a real judgment.
GROUNDING_RULE = """
GROUNDING RULE (critical): Only state something as fact if it is directly present in the query, response, or prompts you were given. If you want to raise a possibility that isn't explicitly stated in the data (e.g. guessing a currency from a domain name, assuming a size chart, inferring intent not written anywhere) — you may mention it, but you MUST explicitly flag it as an unverified guess (e.g. "possibly...", "this cannot be confirmed from the given data, but..."), never state it as established fact, and never let an unverified guess be the primary reason for a score or verdict. If your reasoning leans on such a guess, say so explicitly."""

DRAFT_SYSTEM_PROMPT = """You are a strict quality judge for an AI shopping assistant's responses. Given a user's query and the assistant's response, score how well the response satisfies the query.

Score from 0.0 (completely fails to help) to 1.0 (excellent, fully satisfies the query).

Consider: relevance of any products shown, whether the response actually addresses what was asked, correctness, and whether the tone/content is appropriate. If the user provided their own feedback reason, weigh it heavily but verify it against the actual response yourself rather than blindly trusting it — the user's own stated reason can also be imprecise or overstated, so check it against the real response text.
""" + GROUNDING_RULE + """

Output ONLY raw JSON, no markdown fences, no commentary:
{"score": <float 0-1>, "reasoning": "<why this score, specifically what's good or bad>"}"""

CRITIQUE_SYSTEM_PROMPT = """You are reviewing your own prior quality judgment of an AI shopping assistant's response, to catch mistakes or blind spots before finalizing it.

You will be shown the original query, response, and your own draft score and reasoning. Critically re-examine your draft: did you miss anything, misjudge relevance, or reason too shallowly? Also specifically check whether your draft reasoning stated any unverified guess as fact — if so, correct it. Then give a corrected (possibly identical) score and reasoning.
""" + GROUNDING_RULE + """

Output ONLY raw JSON, no markdown fences, no commentary:
{"corrected_score": <float 0-1>, "corrected_reasoning": "<final reasoning, incorporating any correction>"}"""

ROOT_CAUSE_SYSTEM_PROMPT = """You are debugging a multi-agent AI pipeline that produced a bad response. You will be given the user's query, the bad response, an analysis of what's wrong with it, and the full text of every agent's system prompt/instructions in the pipeline that could have produced it.

Your job: identify which ONE agent's prompt is most likely responsible for this specific failure, quote the exact sentence (or short passage) from that agent's prompt you believe caused it, and explain the causal chain connecting that instruction to the observed failure.

Only point to genuine, specific wording issues in a prompt — don't invent a cause if the prompts look reasonable; in that case say so honestly and note the failure may stem from the underlying model's execution rather than the prompt wording itself.

Be decisive and concise: skim the prompts for the one or two most plausible culprits based on the failure symptom, form a judgment, and answer. Do NOT exhaustively trace step-by-step through every agent in the pipeline one by one — that wastes your budget and you will run out of room before answering. A few sentences of internal reasoning is enough.
""" + GROUNDING_RULE + """

Output ONLY raw JSON, no markdown fences, no commentary:
{"root_cause_node": "<agent name, or null if inconclusive>", "root_cause_snippet": "<quoted sentence, or null>", "root_cause_explanation": "<causal reasoning, or explanation of why inconclusive>"}"""


def _parse_json_response(text: str) -> dict:
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


STREAM_TTL_SECONDS = 3600  # cache hygiene only — response_judgments (Postgres) is the real source of truth


def _stream_key(feedback_id: int) -> str:
    return f"judge_stream:{feedback_id}"


async def _publish_stream_event(feedback_id: int, event: dict) -> None:
    """Appends one event to this feedback's Redis Stream, for the evals
    dashboard's live SSE view (see routers/evals.py). Uses a Stream, not
    Pub/Sub, specifically so a client that connects a moment after the
    pipeline started doesn't miss the opening events — Streams retain
    history for late subscribers, Pub/Sub does not."""
    key = _stream_key(feedback_id)
    await db.redis_client.xadd(key, {"data": json.dumps(event)})
    await db.redis_client.expire(key, STREAM_TTL_SECONDS)


async def _call_judge(
    system_prompt: str, user_prompt: str, feedback_id: int, stage: str,
    max_tokens: int = 2048, reasoning_max_tokens: int = 3000, wall_clock_timeout: float = 180,
) -> tuple[dict, str]:
    # Nemotron is a reasoning model — its chain-of-thought needs to be
    # explicitly enabled ('reasoning.enabled': true) to land in the separate
    # 'reasoning' delta field instead of leaking into 'content' ahead of the
    # actual JSON answer. A bare 'reasoning.max_tokens' without 'enabled'
    # doesn't reliably work for this model — both are required together.
    # Streams token-by-token (stream: true) so the evals dashboard can show
    # the model's thinking live, publishing each delta to this feedback
    # event's Redis Stream as it arrives. Returns (parsed_json, thinking_text).
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
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                    "reasoning": {"enabled": True, "max_tokens": reasoning_max_tokens},
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
                    # error — OpenRouter reports it as a normal-looking chunk
                    # with finish_reason: "error" instead. Without checking
                    # this explicitly, the loop just keeps waiting for more
                    # 'data:' lines that are never coming — a second, distinct
                    # way for a call to hang beyond the wall-clock-timeout
                    # case above. (Confirmed via OpenRouter's own docs.)
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason == "error":
                        raise RuntimeError(f"OpenRouter reported a mid-stream error (finish_reason=error) for stage '{stage}'")
                    delta = choices[0].get("delta", {})
                    reasoning_delta = delta.get("reasoning")
                    content_delta = delta.get("content")
                    if reasoning_delta:
                        thinking_parts.append(reasoning_delta)
                        await _publish_stream_event(feedback_id, {"stage": stage, "type": "thinking", "delta": reasoning_delta})
                    if content_delta:
                        content_parts.append(content_delta)
                        await _publish_stream_event(feedback_id, {"stage": stage, "type": "content", "delta": content_delta})
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
    parsed = _parse_json_response("".join(content_parts))
    await _publish_stream_event(feedback_id, {"stage": stage, "type": "stage_done"})
    return parsed, full_thinking


async def run_judge_pipeline(feedback_id: int) -> None:
    if not OPENROUTER_API_KEY:
        logger.warning("judge: OPENROUTER_API_KEY not configured, skipping feedback_id=%s", feedback_id)
        return
    try:
        feedback = await db.db_pool.fetchrow(
            "SELECT id, chat_id, message_id, rating, reason FROM response_feedback WHERE id = $1",
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

        feedback_block = f"User feedback: thumbs {feedback['rating']}"
        if feedback["reason"]:
            feedback_block += f", reason: {feedback['reason']}"

        # Stage B — draft, then self-critique/revise
        draft, draft_thinking = await _call_judge(
            DRAFT_SYSTEM_PROMPT,
            f"User query: {query_text}\n\nAssistant response: {response_text}\n\n{feedback_block}",
            feedback_id, "draft",
        )
        initial_score = draft.get("score")
        initial_reasoning = draft.get("reasoning", "")

        critique, critique_thinking = await _call_judge(
            CRITIQUE_SYSTEM_PROMPT,
            f"User query: {query_text}\n\nAssistant response: {response_text}\n\n{feedback_block}\n\n"
            f"Your draft score: {initial_score}\nYour draft reasoning: {initial_reasoning}\n\n"
            "Critique and correct your draft judgment.",
            feedback_id, "critique",
        )
        corrected_score = critique.get("corrected_score", initial_score)
        corrected_reasoning = critique.get("corrected_reasoning", initial_reasoning)

        root_cause_node = root_cause_snippet = root_cause_explanation = None
        attribution_thinking = None
        is_bad = feedback["rating"] == "down" or (
            isinstance(corrected_score, (int, float)) and corrected_score < BAD_SCORE_THRESHOLD
        )

        if is_bad:
            prompt_rows = await db.db_pool.fetch(
                """
                SELECT wn.alias, wn.agent_instructions FROM workflow_nodes wn
                JOIN workflow_versions wv ON wv.id = wn.workflow_version_id
                WHERE wv.is_active = true AND wn.node_type = 'agent' AND wn.agent_instructions IS NOT NULL
                """
            )
            if prompt_rows:
                library = "\n\n".join(f"--- {r['alias']} ---\n{r['agent_instructions']}" for r in prompt_rows)
                attribution, attribution_thinking = await _call_judge(
                    ROOT_CAUSE_SYSTEM_PROMPT,
                    f"User query: {query_text}\n\nBad response: {response_text}\n\n{feedback_block}\n\n"
                    f"Failure analysis: {corrected_reasoning}\n\nAGENT PROMPTS:\n{library}",
                    feedback_id, "attribution",
                    max_tokens=6000,  # verified sufficient (finish_reason=stop) with reasoning.enabled over the real ~25k-token prompt library
                    wall_clock_timeout=300,  # this stage's ~25k-token prompt has historically taken longer (up to ~250s observed) than Stage B's small calls
                )
                root_cause_node = attribution.get("root_cause_node")
                root_cause_snippet = attribution.get("root_cause_snippet")
                root_cause_explanation = attribution.get("root_cause_explanation")
            else:
                logger.warning("judge: no active workflow prompt library found, skipping Stage C for feedback_id=%s", feedback_id)

        await db.db_pool.execute(
            """
            INSERT INTO response_judgments
                (feedback_id, initial_score, corrected_score, judge_reasoning,
                 root_cause_node, root_cause_snippet, root_cause_explanation,
                 draft_thinking, critique_thinking, attribution_thinking)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            feedback_id, initial_score, corrected_score, corrected_reasoning,
            root_cause_node, root_cause_snippet, root_cause_explanation,
            draft_thinking, critique_thinking, attribution_thinking,
        )
        logger.info(
            "judge: feedback_id=%s initial_score=%s corrected_score=%s root_cause_node=%s",
            feedback_id, initial_score, corrected_score, root_cause_node,
        )
        await _publish_stream_event(feedback_id, {
            "type": "pipeline_done",
            "initial_score": initial_score,
            "corrected_score": corrected_score,
            "judge_reasoning": corrected_reasoning,
            "root_cause_node": root_cause_node,
            "root_cause_snippet": root_cause_snippet,
            "root_cause_explanation": root_cause_explanation,
        })
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
        await _publish_stream_event(feedback_id, {"type": "pipeline_error", "detail": str(e)})
