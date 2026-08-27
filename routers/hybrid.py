"""Hybrid — LLM verdict + human review shown side by side, with a lightweight
human sign-off. Not a computed blended score (see db.py's hybrid_approvals
comment) — approving just records that someone looked at both and confirmed
them. Only ever meaningful for a feedback event that has BOTH a
response_judgments row (LLM Judge, routers/evals.py) and a
response_human_reviews row (HITL, routers/hitl.py) — the feed below only
surfaces items where both exist.
"""
import logging
from datetime import datetime, timezone

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

import cache
import db
import projects as projects_module
from models import (
    HybridFeedResponse, HybridFeedItem, HybridDetailResponse, HybridApproveResponse,
)

logger = logging.getLogger("columbia_backend")

router = APIRouter()

# Same tradeoff as routers/evals.py's _EVALS_CACHE_TTL / routers/hitl.py's
# _HITL_CACHE_TTL — short TTL, explicit invalidation on the approve endpoint.
_HYBRID_CACHE_TTL = 5
_RESPONSE_SNIPPET_LEN = 200


def _to_iso(ts: datetime) -> str:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts.isoformat()


def _feed_cache_key(project_id: int) -> str:
    return f"evalcache:hybrid-feed:{project_id}"


# INNER JOINs on response_judgments/response_human_reviews (not LEFT JOIN) —
# this is the actual "only show items with both scores" filter; a feedback
# event missing either one simply never matches this query.
_HYBRID_FEED_QUERY = """
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
    rj.corrected_score AS llm_score,
    rj.judge_reasoning AS llm_reasoning,
    rj.root_cause_node,
    hr.score AS human_score,
    hr.reason_valid AS human_reason_valid,
    hr.notes AS human_notes,
    reviewer.name AS reviewer_name,
    reviewer.email AS reviewer_email,
    ha.id AS approval_id,
    ha.created_at AS approved_at,
    approver.name AS approver_name,
    approver.email AS approver_email
FROM response_feedback rf
JOIN messages ai_msg ON ai_msg.id = rf.message_id
JOIN response_judgments rj ON rj.feedback_id = rf.id
JOIN response_human_reviews hr ON hr.feedback_id = rf.id
LEFT JOIN users reviewer ON reviewer.id = hr.reviewer_user_id
LEFT JOIN hybrid_approvals ha ON ha.feedback_id = rf.id
LEFT JOIN users approver ON approver.id = ha.approved_by_user_id
WHERE rf.project_id = $1
"""


@router.get(
    "/api/projects/{project_id}/evaluation/hybrid",
    response_model=HybridFeedResponse,
    tags=["hybrid"],
    summary="Feed of feedback events with both an LLM verdict and a human review",
    description="Only feedback events that have both scores — that's the whole point of comparing them. Most recent first.",
)
async def get_hybrid_feed(
    limit: int = Query(default=50, le=200),
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    cache_key = _feed_cache_key(project["id"])
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return HybridFeedResponse(**cached)

    try:
        rows = await db.db_pool.fetch(_HYBRID_FEED_QUERY + " ORDER BY rf.created_at DESC LIMIT $2", project["id"], limit)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    items = []
    for r in rows:
        response_text = r["response_text"] or ""
        snippet = response_text[:_RESPONSE_SNIPPET_LEN] + ("..." if len(response_text) > _RESPONSE_SNIPPET_LEN else "")
        items.append(HybridFeedItem(
            feedback_id=r["feedback_id"],
            created_at=_to_iso(r["feedback_created_at"]),
            rating=r["rating"],
            query=r["query_text"],
            response_snippet=snippet,
            llm_score=r["llm_score"],
            human_score=r["human_score"],
            human_reason_valid=r["human_reason_valid"],
            approved=r["approval_id"] is not None,
        ))
    result = HybridFeedResponse(items=items)
    await cache.set_json(cache_key, result.model_dump(), _HYBRID_CACHE_TTL)
    return result


@router.get(
    "/api/projects/{project_id}/evaluation/hybrid/{feedback_id}",
    response_model=HybridDetailResponse,
    tags=["hybrid"],
    summary="Full detail for one feedback event (Hybrid comparison page)",
)
async def get_hybrid_detail(
    feedback_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        row = await db.db_pool.fetchrow(_HYBRID_FEED_QUERY + " AND rf.id = $2", project["id"], feedback_id)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    if not row:
        raise HTTPException(status_code=404, detail="Feedback event not found, or missing an LLM verdict / human review.")

    return HybridDetailResponse(
        feedback_id=row["feedback_id"],
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        rating=row["rating"],
        reason=row["reason"],
        created_at=_to_iso(row["feedback_created_at"]),
        query=row["query_text"],
        response=row["response_text"],
        llm_score=row["llm_score"],
        llm_reasoning=row["llm_reasoning"],
        root_cause_node=row["root_cause_node"],
        human_score=row["human_score"],
        human_reason_valid=row["human_reason_valid"],
        human_notes=row["human_notes"],
        human_reviewer_name=row["reviewer_name"] or row["reviewer_email"],
        approved=row["approval_id"] is not None,
        approved_by=row["approver_name"] or row["approver_email"],
        approved_at=_to_iso(row["approved_at"]) if row["approved_at"] else None,
    )


@router.post(
    "/api/projects/{project_id}/evaluation/hybrid/{feedback_id}/approve",
    response_model=HybridApproveResponse,
    tags=["hybrid"],
    summary="Sign off on this feedback event's LLM + human scores",
    description="A lightweight approval stamp, not a computed score — re-approving just updates who/when.",
)
async def approve_hybrid(
    feedback_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    owner_row = await db.db_pool.fetchrow(
        """
        SELECT rf.id FROM response_feedback rf
        JOIN response_judgments rj ON rj.feedback_id = rf.id
        JOIN response_human_reviews hr ON hr.feedback_id = rf.id
        WHERE rf.id = $1 AND rf.project_id = $2
        """,
        feedback_id, project["id"],
    )
    if not owner_row:
        raise HTTPException(status_code=404, detail="Feedback event not found, or missing an LLM verdict / human review.")

    try:
        row = await db.db_pool.fetchrow(
            """
            INSERT INTO hybrid_approvals (feedback_id, approved_by_user_id)
            VALUES ($1, $2)
            ON CONFLICT (feedback_id) DO UPDATE
                SET approved_by_user_id = EXCLUDED.approved_by_user_id,
                    created_at = now()
            RETURNING created_at
            """,
            feedback_id, current_user["id"],
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    logger.info("hybrid: feedback_id=%s approved by user_id=%s", feedback_id, current_user["id"])
    await cache.delete(_feed_cache_key(project["id"]))

    return HybridApproveResponse(
        feedback_id=feedback_id,
        approved_by=current_user["name"] or current_user["email"],
        approved_at=_to_iso(row["created_at"]),
    )
