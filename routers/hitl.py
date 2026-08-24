"""HITL (Human-in-the-loop) review — a second, independent opinion on a
feedback event, alongside (not overriding) the LLM judge's response_judgments
(see routers/evals.py). Reuses the exact same feedback/query/response join
shape as the LLM Judge feed — only the "verdict" side differs (a human's
reason_valid/score/notes instead of the judge's scores/root-cause). See
db.py's response_human_reviews comment for the storage model.
"""
import logging
from datetime import datetime, timezone

import asyncpg
from fastapi import APIRouter, Body, Depends, HTTPException, Query

import cache
import db
import projects as projects_module
from models import (
    HitlFeedResponse, HitlFeedItem, HitlDetailResponse,
    HitlReviewRequest, HitlReviewResponse,
)

logger = logging.getLogger("columbia_backend")

router = APIRouter()

# Same tradeoff as routers/evals.py's _EVALS_CACHE_TTL — short TTL, no
# explicit invalidation on read, but the review-submit endpoint below does
# invalidate the feed cache directly, since "I just submitted a review and
# it still shows as pending" would be a much more jarring staleness than the
# LLM-judge feed's few-seconds-behind tolerance.
_HITL_CACHE_TTL = 5
_RESPONSE_SNIPPET_LEN = 200


def _to_iso(ts: datetime) -> str:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts.isoformat()


def _feed_cache_key(project_id: int) -> str:
    return f"evalcache:hitl-feed:{project_id}"


# Only rating = 'down' — per the product decision, positive feedback rarely
# needs a human second opinion; review effort is focused on the feedback
# that actually flagged a problem.
_HITL_FEED_QUERY = """
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
    hr.id AS review_id,
    hr.created_at AS reviewed_at,
    hr.reason_valid,
    hr.score,
    hr.notes,
    hr.reviewer_user_id,
    reviewer.name AS reviewer_name,
    reviewer.email AS reviewer_email
FROM response_feedback rf
JOIN messages ai_msg ON ai_msg.id = rf.message_id
LEFT JOIN response_human_reviews hr ON hr.feedback_id = rf.id
LEFT JOIN users reviewer ON reviewer.id = hr.reviewer_user_id
WHERE rf.project_id = $1 AND rf.rating = 'down'
"""


@router.get(
    "/api/projects/{project_id}/evaluation/hitl",
    response_model=HitlFeedResponse,
    tags=["hitl"],
    summary="Feed of feedback events awaiting/having human review",
    description="Thumbs-down feedback only — see the product decision in the HITL design. Most recent first.",
)
async def get_hitl_feed(
    limit: int = Query(default=50, le=200),
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    cache_key = _feed_cache_key(project["id"])
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return HitlFeedResponse(**cached)

    try:
        rows = await db.db_pool.fetch(_HITL_FEED_QUERY + " ORDER BY rf.created_at DESC LIMIT $2", project["id"], limit)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    items = []
    for r in rows:
        response_text = r["response_text"] or ""
        snippet = response_text[:_RESPONSE_SNIPPET_LEN] + ("..." if len(response_text) > _RESPONSE_SNIPPET_LEN else "")
        items.append(HitlFeedItem(
            feedback_id=r["feedback_id"],
            created_at=_to_iso(r["feedback_created_at"]),
            rating=r["rating"],
            reason=r["reason"],
            query=r["query_text"],
            response_snippet=snippet,
            reviewed=r["review_id"] is not None,
            reason_valid=r["reason_valid"],
            score=r["score"],
        ))
    result = HitlFeedResponse(items=items)
    await cache.set_json(cache_key, result.model_dump(), _HITL_CACHE_TTL)
    return result


@router.get(
    "/api/projects/{project_id}/evaluation/hitl/{feedback_id}",
    response_model=HitlDetailResponse,
    tags=["hitl"],
    summary="Full detail for one feedback event (HITL review page)",
)
async def get_hitl_detail(
    feedback_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        row = await db.db_pool.fetchrow(_HITL_FEED_QUERY + " AND rf.id = $2", project["id"], feedback_id)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    if not row:
        raise HTTPException(status_code=404, detail="Feedback event not found.")

    return HitlDetailResponse(
        feedback_id=row["feedback_id"],
        chat_id=row["chat_id"],
        message_id=row["message_id"],
        rating=row["rating"],
        reason=row["reason"],
        created_at=_to_iso(row["feedback_created_at"]),
        query=row["query_text"],
        response=row["response_text"],
        reviewed=row["review_id"] is not None,
        reviewed_at=_to_iso(row["reviewed_at"]) if row["reviewed_at"] else None,
        reviewer_name=row["reviewer_name"] or row["reviewer_email"],
        reason_valid=row["reason_valid"],
        score=row["score"],
        notes=row["notes"],
    )


@router.post(
    "/api/projects/{project_id}/evaluation/hitl/{feedback_id}/review",
    response_model=HitlReviewResponse,
    tags=["hitl"],
    summary="Submit (or revise) a human review for one feedback event",
    description=(
        "One review per feedback event — re-submitting replaces the existing review "
        "(and its reviewer) rather than creating a second one. Stamped with whoever is logged in."
    ),
)
async def submit_hitl_review(
    feedback_id: int,
    request: HitlReviewRequest = Body(...),
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    owner_row = await db.db_pool.fetchrow(
        "SELECT id FROM response_feedback WHERE id = $1 AND project_id = $2",
        feedback_id, project["id"],
    )
    if not owner_row:
        raise HTTPException(status_code=404, detail="Feedback event not found for this project.")

    try:
        row = await db.db_pool.fetchrow(
            """
            INSERT INTO response_human_reviews (feedback_id, reviewer_user_id, reason_valid, score, notes)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (feedback_id) DO UPDATE
                SET reviewer_user_id = EXCLUDED.reviewer_user_id,
                    reason_valid = EXCLUDED.reason_valid,
                    score = EXCLUDED.score,
                    notes = EXCLUDED.notes,
                    created_at = now()
            RETURNING reason_valid, score, notes, created_at
            """,
            feedback_id, current_user["id"], request.reason_valid, request.score, request.notes,
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    logger.info("hitl: feedback_id=%s reviewed by user_id=%s reason_valid=%s", feedback_id, current_user["id"], request.reason_valid)
    await cache.delete(_feed_cache_key(project["id"]))

    return HitlReviewResponse(
        feedback_id=feedback_id,
        reason_valid=row["reason_valid"],
        score=row["score"],
        notes=row["notes"],
        reviewed_at=_to_iso(row["created_at"]),
    )
