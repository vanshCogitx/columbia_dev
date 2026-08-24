"""Traceability — execution trace (which workflow nodes ran, and each one's
raw output) for a real chat query. Reuses message_traces (saved at
generation time for judge.py's Stage C — see cartesian.py's
_save_message_trace) and the same TraceTree.jsx component Testing already
uses for simulated turns; only the data source differs.

Scoped to response_feedback, not to messages directly, and deliberately so:
chats/messages carry no project_id at all (see db.py's projects-table
comment) — response_feedback.project_id is the only place a query is ever
attributed to a project, so it's the only queue this feature can be built
on top of. All ratings show up here (not just thumbs-down like HITL/Hybrid)
since this is about understanding execution, not triaging problems.
"""
import json
import logging
from datetime import datetime, timezone

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

import cache
import db
import projects as projects_module
from models import (
    TraceabilityFeedResponse, TraceabilityFeedItem, TraceabilityDetailResponse,
)

logger = logging.getLogger("columbia_backend")

router = APIRouter()

_TRACEABILITY_CACHE_TTL = 5
_RESPONSE_SNIPPET_LEN = 200


def _to_iso(ts: datetime) -> str:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts.isoformat()


def _feed_cache_key(project_id: int) -> str:
    return f"evalcache:traceability-feed:{project_id}"


def _parse_json_field(value):
    if value is None:
        return None
    return json.loads(value) if isinstance(value, str) else value


_TRACEABILITY_QUERY = """
SELECT
    rf.id AS feedback_id,
    rf.rating,
    rf.created_at AS feedback_created_at,
    ai_msg.text AS response_text,
    (
        SELECT text FROM messages um
        WHERE um.chat_id = rf.chat_id AND um.id < rf.message_id AND um.sender = 'user'
        ORDER BY um.id DESC LIMIT 1
    ) AS query_text,
    mt.raw_output,
    mt.execution_summary
FROM response_feedback rf
JOIN messages ai_msg ON ai_msg.id = rf.message_id
LEFT JOIN message_traces mt ON mt.message_id = rf.message_id
WHERE rf.project_id = $1
"""


@router.get(
    "/api/projects/{project_id}/monitoring/traceability",
    response_model=TraceabilityFeedResponse,
    tags=["traceability"],
    summary="Feed of queries with an execution trace available",
    description="All feedback ratings (not just thumbs-down) — this is about inspecting execution, not triaging problems. Most recent first.",
)
async def get_traceability_feed(
    limit: int = Query(default=50, le=200),
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    cache_key = _feed_cache_key(project["id"])
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return TraceabilityFeedResponse(**cached)

    try:
        rows = await db.db_pool.fetch(_TRACEABILITY_QUERY + " ORDER BY rf.created_at DESC LIMIT $2", project["id"], limit)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    items = []
    for r in rows:
        response_text = r["response_text"] or ""
        snippet = response_text[:_RESPONSE_SNIPPET_LEN] + ("..." if len(response_text) > _RESPONSE_SNIPPET_LEN else "")
        execution_summary = _parse_json_field(r["execution_summary"])
        items.append(TraceabilityFeedItem(
            feedback_id=r["feedback_id"],
            created_at=_to_iso(r["feedback_created_at"]),
            rating=r["rating"],
            query=r["query_text"],
            response_snippet=snippet,
            has_trace=r["raw_output"] is not None,
            node_count=(execution_summary or {}).get("totalNodes"),
        ))
    result = TraceabilityFeedResponse(items=items)
    await cache.set_json(cache_key, result.model_dump(), _TRACEABILITY_CACHE_TTL)
    return result


@router.get(
    "/api/projects/{project_id}/monitoring/traceability/{feedback_id}",
    response_model=TraceabilityDetailResponse,
    tags=["traceability"],
    summary="Full execution trace for one query (Traceability page)",
)
async def get_traceability_detail(
    feedback_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        row = await db.db_pool.fetchrow(_TRACEABILITY_QUERY + " AND rf.id = $2", project["id"], feedback_id)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    if not row:
        raise HTTPException(status_code=404, detail="Feedback event not found.")

    return TraceabilityDetailResponse(
        feedback_id=row["feedback_id"],
        rating=row["rating"],
        created_at=_to_iso(row["feedback_created_at"]),
        query=row["query_text"],
        response=row["response_text"],
        has_trace=row["raw_output"] is not None,
        raw_output=_parse_json_field(row["raw_output"]),
        execution_summary=_parse_json_field(row["execution_summary"]),
    )
