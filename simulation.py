"""Evals system, "Testing" tab — proactive scenario simulation. An LLM plays
a simulated shopper against the real Cartesian workflow for several turns,
then judge.py scores the full transcript against the scenario's success
criteria. Kept fully separate from chats/messages (see simulation_runs/
simulation_turns in db.py) so synthetic test traffic never touches real chat
history or the reactive evals stats. Reuses cartesian._call_cartesian_workflow
directly and unmodified — the same function real chat generation calls.
Judging (judge.py's NVIDIA credentials/model/streaming machinery) is reused
as-is, but the simulated-shopper call below deliberately does NOT share
NVIDIA's judge model — that's a heavy 120B reasoning model on a shared,
rate-limited free tier, complete overkill for "produce one line of shopper
dialogue + a boolean", and coupling it to judge's model meant a judge-side
outage took Testing down too. Runs on OpenRouter with a small model instead —
a different provider, a different rate-limit pool, an already-unused API key
in this project's .env.
"""
import os
import json
import random
import logging
import asyncio

import httpx

import db
import cartesian
import judge
import projects as projects_module

logger = logging.getLogger("columbia_backend.simulation")

# Hard ceiling on a whole run (simulated-user calls + Cartesian calls, each up
# to ~90s worst case, times max_turns, plus one final verdict call) — without
# this, a stuck run (e.g. Cartesian queue backup) hangs the background task
# forever instead of failing cleanly. See the plan's "Open risks" section.
RUN_HARD_TIMEOUT_SECONDS = 900

# Simulated shopper's own model — intentionally separate from judge.py's
# NVIDIA_MODEL (see module docstring). OpenRouter, not NVIDIA: different
# provider/rate-limit pool, so an NVIDIA outage can't block Testing too.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SIMULATED_USER_MODEL = "meta-llama/llama-3.1-8b-instruct"

_SIMULATED_USER_SYSTEM_PROMPT_SUFFIX = """

Given the conversation so far, produce the shopper's next message — concise and realistic, as a real person would type it. Stay fully in character; do not mention that this is a test or reference success criteria. If the conversation has reached a natural end (your goal was met, or it's clear it won't be), set conversation_over to true and leave message empty.
""" + judge.GROUNDING_RULE + judge.BREVITY_RULE + """

Output ONLY raw JSON, no markdown fences, no commentary:
{"conversation_over": <bool>, "message": "<next shopper message, empty string if conversation_over>"}"""


def _build_simulated_user_system_prompt(scenario_description: str) -> str:
    # Built via f-string interpolation rather than str.format() on a stored
    # template — the JSON example above has literal {...} braces that
    # str.format() would misparse as placeholders (confirmed: raised
    # KeyError: '"conversation_over"' when tested directly).
    return (
        "You are role-playing as a shopper testing an AI shopping assistant, for internal QA purposes.\n\n"
        f"SCENARIO YOU ARE PLAYING: {scenario_description}"
        + _SIMULATED_USER_SYSTEM_PROMPT_SUFFIX
    )


async def _call_simulated_user(scenario_description: str, transcript_text: str) -> dict:
    """A single, fast, non-streaming call — deliberately NOT judge's
    _call_judge: this runs once per turn (several times per run) and there's
    no dashboard value in showing its 'thinking' token-by-token, so it skips
    Redis Stream publishing and extended reasoning entirely — small
    max_tokens. Extends this session's BREVITY_RULE lesson more aggressively
    since the cost compounds per-turn, not once per pipeline. Runs on
    OpenRouter with SIMULATED_USER_MODEL (see module docstring for why this
    is deliberately not judge's NVIDIA model). Retries transient failures
    with the same backoff policy as judge._call_judge (judge.RETRY_*
    constants) — a single 503/429 shouldn't fail an entire multi-turn run."""
    system_prompt = _build_simulated_user_system_prompt(scenario_description)
    user_prompt = f"Conversation so far:\n{transcript_text or '(nothing yet — this is the opening message)'}"

    last_exc = None
    for attempt in range(judge.RETRY_MAX_ATTEMPTS):
        retry_after = None
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": SIMULATED_USER_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.4,
                        "top_p": 0.95,
                        "max_tokens": 300,
                        "stream": False,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return judge.parse_json_response(content)
        except asyncio.TimeoutError:
            last_exc = RuntimeError("simulated-user call timed out")
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (429, 500, 502, 503, 504):
                raise
            last_exc = e
            retry_after = e.response.headers.get("Retry-After")
        except httpx.TransportError as e:
            last_exc = e

        if attempt == judge.RETRY_MAX_ATTEMPTS - 1:
            raise last_exc
        delay = float(retry_after) if retry_after else min(judge.RETRY_MAX_BACKOFF, judge.RETRY_INITIAL_BACKOFF * (2 ** attempt))
        delay *= 1 + random.uniform(-0.2, 0.2)
        logger.warning(
            "simulation: simulated-user call attempt %d/%d failed (%s), retrying in %.1fs",
            attempt + 1, judge.RETRY_MAX_ATTEMPTS, last_exc, delay,
        )
        await asyncio.sleep(delay)


def _build_compiled_text(prior_turns: list[dict], latest_message: str) -> str:
    """Same shape as cartesian.py's _generate_and_save_response compiled_text,
    minus the CRM profile header — no real logged-in user backs a simulation
    run, so there's no name/profile to inject."""
    if prior_turns:
        history_str = "\n".join(
            f"[User]: {t['simulated_user_text']}\n[AI]: {t['ai_response_text'] or ''}"
            for t in prior_turns
        )
        history_block = f"--- Chat History ---\n{history_str}\n\n"
    else:
        history_block = ""
    return (
        "System Instruction: Below is the chat history of this conversation. "
        "Please use this information to answer the user's latest query at the end. "
        "Respond ONLY to the latest query and do not repeat the context or history.\n\n"
        f"{history_block}"
        f"--- Latest User Query ---\n[User]: {latest_message}"
    )


async def _run_simulation_inner(run_id: int, stream_key: str) -> None:
    run = await db.db_pool.fetchrow("SELECT * FROM simulation_runs WHERE id = $1", run_id)
    if not run:
        raise RuntimeError(f"simulation run_id={run_id} not found")
    scenario = await db.db_pool.fetchrow("SELECT * FROM simulation_scenarios WHERE id = $1", run["scenario_id"])
    if not scenario:
        raise RuntimeError(f"scenario not found for simulation run_id={run_id}")

    success_criteria = (
        json.loads(scenario["success_criteria"])
        if isinstance(scenario["success_criteria"], str)
        else scenario["success_criteria"]
    )

    # Unlike the real chat pipeline, Testing genuinely calls Cartesian with
    # this scenario's own project's credentials — simulation_scenarios never
    # touches chats/messages, so it's the one place per-project isolation is
    # real today (see projects.py's module docstring).
    try:
        cfg = await projects_module.get_project_cartesian_config(scenario["project_id"])
    except ValueError as e:
        raise RuntimeError(str(e))
    headers = {"x-client-id": cfg["client_id"], "x-client-secret": cfg["client_secret"], "Content-Type": "application/json"}
    url = f"{cfg['base_url'].rstrip('/')}{cfg['job_path']}"
    poll_url_base = url

    turns: list[dict] = []
    for turn_index in range(scenario["max_turns"]):
        await judge.publish_stream_event(stream_key, {"type": "turn_start", "turn_index": turn_index})

        transcript_text = "\n".join(
            f"[User]: {t['simulated_user_text']}\n[AI]: {t['ai_response_text'] or ''}" for t in turns
        )
        sim_user = await _call_simulated_user(scenario["user_scenario"], transcript_text)
        if sim_user.get("conversation_over") or not sim_user.get("message"):
            await judge.publish_stream_event(stream_key, {"type": "conversation_ended_early", "turn_index": turn_index})
            break

        message = sim_user["message"]
        await judge.publish_stream_event(stream_key, {"type": "user_message", "turn_index": turn_index, "text": message})

        # Same sessionId every turn (Cartesian-side continuity, mirroring how
        # a real chat's session_id stays fixed across a conversation) and a
        # synthetic, distinguishable user_id — CRM lookups will simply find
        # no match, which is expected/fine for a generic test scenario not
        # tied to a specific customer profile.
        compiled_text = _build_compiled_text(turns, message)
        payload = {"payload": {"user_query": compiled_text, "user_id": f"eval-sim-{run_id}"}}
        params = {"waitSeconds": 30, "sessionId": run["session_id"]}

        async with httpx.AsyncClient() as client:
            ai_text, raw_output = await cartesian._call_cartesian_workflow(
                client, url, poll_url_base, params, payload, headers
            )

        turn_row = await db.db_pool.fetchrow(
            """
            INSERT INTO simulation_turns (run_id, turn_index, simulated_user_text, ai_response_text, raw_output)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING id
            """,
            run_id, turn_index, message, ai_text, json.dumps(raw_output) if raw_output else None,
        )
        await db.db_pool.execute("UPDATE simulation_runs SET turn_count = $1 WHERE id = $2", turn_index + 1, run_id)

        turns.append({"simulated_user_text": message, "ai_response_text": ai_text})
        await judge.publish_stream_event(stream_key, {
            "type": "turn_done",
            "turn_index": turn_index,
            "turn_id": turn_row["id"],
            "ai_text": ai_text,
            "executed_nodes": (raw_output or {}).get("executionPath"),
        })

    transcript_for_verdict = "\n\n".join(
        f"Turn {i + 1}:\n[User]: {t['simulated_user_text']}\n[AI]: {t['ai_response_text'] or '(no response)'}"
        for i, t in enumerate(turns)
    )
    verdict_result = await judge.run_simulation_verdict(
        scenario["user_scenario"], success_criteria, transcript_for_verdict, stream_key
    )

    await db.db_pool.execute(
        """
        UPDATE simulation_runs
        SET status = 'completed', verdict = $1, verdict_reasoning = $2, verdict_thinking = $3, completed_at = now()
        WHERE id = $4
        """,
        verdict_result["verdict"], verdict_result["verdict_reasoning"], verdict_result["verdict_thinking"], run_id,
    )
    await judge.publish_stream_event(stream_key, {
        "type": "run_done",
        "verdict": verdict_result["verdict"],
        "verdict_reasoning": verdict_result["verdict_reasoning"],
    })
    logger.info("simulation: run_id=%s completed, verdict=%s, turns=%d", run_id, verdict_result["verdict"], len(turns))


async def run_simulation(run_id: int) -> None:
    """Fire-and-forget background task — same convention as
    judge.run_judge_pipeline / cartesian's chat generation (bare
    asyncio.create_task, not FastAPI BackgroundTasks, not cancelled on client
    disconnect). Triggered from routers/testing.py's run-trigger endpoint."""
    stream_key = judge.make_stream_key(judge.STREAM_PREFIX_SIM_RUN, run_id)
    try:
        await asyncio.wait_for(_run_simulation_inner(run_id, stream_key), timeout=RUN_HARD_TIMEOUT_SECONDS)
    except Exception as e:
        # str(e) is empty for some exception types (e.g. asyncio.TimeoutError)
        # — repr(e) as a fallback so the DB/stream 'detail' is never blank,
        # and logger.exception (not .error) so the full traceback actually
        # lands in logs instead of just whatever str(e) happened to be.
        detail = str(e) or repr(e)
        logger.exception("simulation: run_id=%s failed", run_id)
        try:
            await db.db_pool.execute(
                "UPDATE simulation_runs SET status = 'failed', error = $1, completed_at = now() WHERE id = $2",
                detail, run_id,
            )
        except Exception as update_err:
            logger.error("simulation: failed to record error for run_id=%s: %s", run_id, update_err)
        await judge.publish_stream_event(stream_key, {"type": "run_error", "detail": detail})
