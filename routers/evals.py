import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import asyncpg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

import db
from models import (
    EvalsFeedResponse, EvalsFeedItem, EvalsDetailResponse,
    EvalsStatsResponse, EvalsRootCauseCount,
)

logger = logging.getLogger("columbia_backend")

router = APIRouter()

_DASHBOARD_HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "evals_dashboard.html")


@router.get("/admin/evals", response_class=HTMLResponse, tags=["evals"], summary="Evals dashboard (live feed UI)")
async def get_evals_dashboard():
    with open(_DASHBOARD_HTML_PATH) as f:
        return HTMLResponse(content=f.read())

# Prototype scope, deliberately not gated behind the customer-facing JWT login
# (see routers/chat.py's get_current_user) — this is an internal panel over
# response_feedback/response_judgments, read-only, no customer PII beyond
# what's already in those tables. Add real access control before this is
# ever exposed outside a trusted network.

_RESPONSE_SNIPPET_LEN = 200


def _to_iso(ts: datetime) -> str:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts.isoformat()


_FEED_QUERY = """
SELECT
    rf.id AS feedback_id,
    rf.chat_id,
    rf.message_id,
    rf.rating,
    rf.reason,
    rf.created_at AS feedback_created_at,
    ai_msg.text AS response_text,
    (
        SELECT text FROM messages um
        WHERE um.chat_id = rf.chat_id AND um.id < rf.message_id AND um.sender = 'user'
        ORDER BY um.id DESC LIMIT 1
    ) AS query_text,
    rj.id AS judgment_id,
    rj.created_at AS judged_at,
    rj.initial_score,
    rj.corrected_score,
    rj.judge_reasoning,
    rj.root_cause_node,
    rj.root_cause_snippet,
    rj.root_cause_explanation,
    rj.draft_thinking,
    rj.critique_thinking,
    rj.attribution_thinking,
    rj.error
FROM response_feedback rf
JOIN messages ai_msg ON ai_msg.id = rf.message_id
LEFT JOIN response_judgments rj ON rj.feedback_id = rf.id
"""


@router.get(
    "/api/evals/feed",
    response_model=EvalsFeedResponse,
    tags=["evals"],
    summary="Live feed of feedback events (evals dashboard)",
    description=(
        "Most recent feedback events first, including in-flight ones the judge hasn't "
        "finished scoring yet (judged=false). Poll this for a live-updating feed."
    )
)
async def get_evals_feed(limit: int = Query(default=50, le=200)):
    try:
        rows = await db.db_pool.fetch(_FEED_QUERY + " ORDER BY rf.created_at DESC LIMIT $1", limit)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    items = []
    for r in rows:
        response_text = r["response_text"] or ""
        snippet = response_text[:_RESPONSE_SNIPPET_LEN] + ("..." if len(response_text) > _RESPONSE_SNIPPET_LEN else "")
        items.append(EvalsFeedItem(
            feedback_id=r["feedback_id"],
            created_at=_to_iso(r["feedback_created_at"]),
            rating=r["rating"],
            reason=r["reason"],
            query=r["query_text"],
            response_snippet=snippet,
            judged=r["judgment_id"] is not None,
            corrected_score=r["corrected_score"],
            root_cause_node=r["root_cause_node"],
            error=r["error"],
        ))
    return EvalsFeedResponse(items=items)


@router.get(
    "/api/evals/feed/{feedback_id}",
    response_model=EvalsDetailResponse,
    tags=["evals"],
    summary="Full detail for one feedback event (evals dashboard)",
    description="Complete query, response, feedback reason, judge scores, and the judge's raw reasoning per stage."
)
async def get_evals_detail(feedback_id: int):
    try:
        row = await db.db_pool.fetchrow(_FEED_QUERY + " WHERE rf.id = $1", feedback_id)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    if not row:
        raise HTTPException(status_code=404, detail="Feedback event not found.")

    return EvalsDetailResponse(
        feedback_id=row["feedback_id"],
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        rating=row["rating"],
        reason=row["reason"],
        created_at=_to_iso(row["feedback_created_at"]),
        query=row["query_text"],
        response=row["response_text"],
        judged=row["judgment_id"] is not None,
        judged_at=_to_iso(row["judged_at"]) if row["judged_at"] else None,
        initial_score=row["initial_score"],
        corrected_score=row["corrected_score"],
        judge_reasoning=row["judge_reasoning"],
        root_cause_node=row["root_cause_node"],
        root_cause_snippet=row["root_cause_snippet"],
        root_cause_explanation=row["root_cause_explanation"],
        draft_thinking=row["draft_thinking"],
        critique_thinking=row["critique_thinking"],
        attribution_thinking=row["attribution_thinking"],
        error=row["error"],
    )


STREAM_READ_BLOCK_MS = 5000
STREAM_MAX_IDLE_BLOCKS = 24  # ~2 minutes total of no new messages before giving up on the live view (the background judge task itself is NOT cancelled by this — reopening the event later will show its real result once it lands)


async def _stream_judge_events(feedback_id: int):
    # Already judged (either a normal completed run, or the client
    # reconnected well after the fact) — there's nothing live to stream.
    # Synthesize a single final event straight from the DB and close,
    # rather than depending on the Redis Stream still being within its TTL.
    already = await db.db_pool.fetchrow(
        "SELECT id FROM response_judgments WHERE feedback_id = $1", feedback_id
    )
    if already:
        row = await db.db_pool.fetchrow(_FEED_QUERY + " WHERE rf.id = $1", feedback_id)
        if row["error"]:
            yield f"data: {json.dumps({'type': 'pipeline_error', 'detail': row['error']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'pipeline_done', 'initial_score': row['initial_score'], 'corrected_score': row['corrected_score'], 'judge_reasoning': row['judge_reasoning'], 'root_cause_node': row['root_cause_node'], 'root_cause_snippet': row['root_cause_snippet'], 'root_cause_explanation': row['root_cause_explanation']})}\n\n"
        return

    key = f"judge_stream:{feedback_id}"
    last_id = "0"
    idle_blocks = 0
    while True:
        result = await db.redis_client.xread({key: last_id}, block=STREAM_READ_BLOCK_MS, count=50)
        if not result:
            idle_blocks += 1
            if idle_blocks > STREAM_MAX_IDLE_BLOCKS:
                # Same 'data:'-only, type-tagged shape as every other event —
                # a bare 'event: error' SSE field is never handled by the
                # frontend's onmessage listener, so using it here silently
                # dropped this message and left the UI spinning forever.
                yield f"data: {json.dumps({'type': 'pipeline_error', 'detail': 'Timed out waiting for the judge — it may still finish in the background. Reopen this event later to check.'})}\n\n"
                return
            continue
        idle_blocks = 0
        for _stream_key, entries in result:
            for entry_id, fields in entries:
                last_id = entry_id
                data = fields.get("data", "{}")
                yield f"data: {data}\n\n"
                try:
                    parsed = json.loads(data)
                    if parsed.get("type") in ("pipeline_done", "pipeline_error"):
                        return
                except json.JSONDecodeError:
                    pass


@router.get(
    "/api/evals/stream/{feedback_id}",
    tags=["evals"],
    summary="Live SSE stream of the judge's thinking for one feedback event",
    description=(
        "Streams token-by-token thinking/content deltas per stage (draft/critique/attribution) "
        "as the judge pipeline runs. If the event is already judged, immediately sends a single "
        "final event and closes instead of replaying the whole stream."
    )
)
async def stream_evals_judge(feedback_id: int):
    return StreamingResponse(
        _stream_judge_events(feedback_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/api/evals/stats",
    response_model=EvalsStatsResponse,
    tags=["evals"],
    summary="Aggregate stats for the evals dashboard header",
)
async def get_evals_stats():
    try:
        counts = await db.db_pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_feedback,
                COUNT(*) FILTER (WHERE rating = 'up') AS up_count,
                COUNT(*) FILTER (WHERE rating = 'down') AS down_count
            FROM response_feedback
            """
        )
        avg_row = await db.db_pool.fetchrow("SELECT AVG(corrected_score) AS avg_score FROM response_judgments")
        top_causes = await db.db_pool.fetch(
            """
            SELECT root_cause_node, COUNT(*) AS count FROM response_judgments
            WHERE root_cause_node IS NOT NULL
            GROUP BY root_cause_node ORDER BY count DESC LIMIT 5
            """
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    return EvalsStatsResponse(
        total_feedback=counts["total_feedback"],
        up_count=counts["up_count"],
        down_count=counts["down_count"],
        avg_score=avg_row["avg_score"],
        top_root_causes=[
            EvalsRootCauseCount(root_cause_node=r["root_cause_node"], count=r["count"]) for r in top_causes
        ],
    )
