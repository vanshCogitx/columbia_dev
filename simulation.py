"""Evals system, "Testing" tab — proactive scenario simulation. An LLM plays
a simulated shopper against the real Cartesian workflow for several turns,
then judge.py scores the full transcript against the scenario's success
criteria. Kept fully separate from chats/messages (see simulation_runs/
simulation_turns in db.py) so synthetic test traffic never touches real chat
history or the reactive evals stats. Reuses cartesian._call_cartesian_workflow
directly and unmodified — the same function real chat generation calls — and
judge.py's NVIDIA credentials/model/streaming machinery, so this module holds
no new credentials or duplicated infra.
"""
import os
import json
import logging
import asyncio

import httpx

import db
import cartesian
import judge

logger = logging.getLogger("columbia_backend.simulation")

# Hard ceiling on a whole run (simulated-user calls + Cartesian calls, each up
# to ~90s worst case, times max_turns, plus one final verdict call) — without
# this, a stuck run (e.g. Cartesian queue backup) hangs the background task
# forever instead of failing cleanly. See the plan's "Open risks" section.
RUN_HARD_TIMEOUT_SECONDS = 900

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
    max_tokens, thinking disabled. Extends this session's BREVITY_RULE lesson
    more aggressively since the cost compounds per-turn, not once per
    pipeline. Reuses judge's NVIDIA credentials/model — no new ones."""
    system_prompt = _build_simulated_user_system_prompt(scenario_description)
    user_prompt = f"Conversation so far:\n{transcript_text or '(nothing yet — this is the opening message)'}"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            judge.NVIDIA_URL,
            headers={"Authorization": f"Bearer {judge.NVIDIA_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": judge.NVIDIA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.4,
                "top_p": 0.95,
                "max_tokens": 300,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return judge.parse_json_response(content)


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

    base_url = os.getenv("CARTESIAN_BASE_URL", "https://api.cartesian.ai")
    client_id = os.getenv("CARTESIAN_CLIENT_ID")
    client_secret = os.getenv("CARTESIAN_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Cartesian API credentials are missing.")
    headers = {"x-client-id": client_id, "x-client-secret": client_secret, "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}{cartesian.CARTESIAN_JOB_PATH}"
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
        logger.error("simulation: run_id=%s failed: %s", run_id, e)
        try:
            await db.db_pool.execute(
                "UPDATE simulation_runs SET status = 'failed', error = $1, completed_at = now() WHERE id = $2",
                str(e), run_id,
            )
        except Exception as update_err:
            logger.error("simulation: failed to record error for run_id=%s: %s", run_id, update_err)
        await judge.publish_stream_event(stream_key, {"type": "run_error", "detail": str(e)})
