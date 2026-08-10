import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

import asyncpg
import httpx
import asyncio
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

import db
import cartesian
from models import (
    ChatResponseModel, ChatHistoryMessage, ChatHistoryResponse, ChatMessageRequest,
    SessionProductsResponse, CartesianChatRequest,
)

logger = logging.getLogger("columbia_backend")

router = APIRouter()


def _to_iso(ts: datetime) -> str:
    """Postgres TIMESTAMPTZ comes back from asyncpg as a tz-aware datetime.
    Normalize to UTC and drop the offset for a plain ISO 8601 string."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts.isoformat()


@router.post(
    "/chat",
    tags=["workflow testing"],
    summary="Chat endpoint integrating with Cartesian Columbia workflow",
    description="Send a message to the Cartesian Columbia workflow and get the response."
)
async def chat_endpoint(request: CartesianChatRequest = Body(...)):
    base_url = os.getenv("CARTESIAN_BASE_URL", "https://api.cartesian.ai") # Change this if your base URL is different
    client_id = os.getenv("CARTESIAN_CLIENT_ID")
    client_secret = os.getenv("CARTESIAN_CLIENT_SECRET")

    if not client_id or not client_secret:
         raise HTTPException(status_code=500, detail="Cartesian API credentials (CARTESIAN_CLIENT_ID, CARTESIAN_CLIENT_SECRET) are missing in environment variables.")

    headers = {
        "x-client-id": client_id,
        "x-client-secret": client_secret,
        "Content-Type": "application/json"
    }

    payload = {
        "payload": {
            "user_query": request.text,
            "user_id": request.user_id or "test-user"
        }
    }

    params = {}
    if request.wait_seconds is not None:
        params["waitSeconds"] = request.wait_seconds
    if request.session_id:
        params["sessionId"] = request.session_id

    endpoint_path = "/exports/rest-api/6a59f1b8285cc674dbd79b87/jobs"
    url = f"{base_url.rstrip('/')}{endpoint_path}"

    async with httpx.AsyncClient() as client:
        try:
            # We set a timeout slightly larger than wait_seconds if provided
            timeout = request.wait_seconds + 10 if request.wait_seconds else 40
            response = await client.post(url, json=payload, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Failed to communicate with Cartesian API: {str(e)}")

        # Cartesian API may wrap the response in a "data" object
        resp_data = data.get("data", data)

        if resp_data.get("ok") and resp_data.get("accepted"):
            # Async processing, need to poll
            run_id = resp_data.get("runId")
            if not run_id:
                raise HTTPException(status_code=500, detail="Received accepted response but no runId")

            poll_url = f"{base_url.rstrip('/')}/exports/rest-api/6a59f1b8285cc674dbd79b87/jobs/{run_id}"

            # Poll until complete or timeout (e.g. 30 times with 2s delay = ~60s wait)
            max_retries = 30
            for _ in range(max_retries):
                await asyncio.sleep(2)
                try:
                    poll_resp = await client.get(poll_url, headers=headers, timeout=10)
                    poll_resp.raise_for_status()
                    poll_data_raw = poll_resp.json()
                    poll_data = poll_data_raw.get("data", poll_data_raw)
                except httpx.HTTPError as e:
                    raise HTTPException(status_code=502, detail=f"Failed to poll Cartesian API: {str(e)}")

                if poll_data.get("isCompleted"):
                    return poll_data

            raise HTTPException(status_code=504, detail="Timeout waiting for Cartesian workflow to complete")

        elif resp_data.get("ok") and not resp_data.get("accepted"):
            # Completed within wait window
            return resp_data

        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response from Cartesian API: {data}")


# --- Chat & Session Management Endpoints ---

@router.get(
    "/api/chats",
    response_model=List[ChatResponseModel],
    tags=["chat"],
    summary="Get all chats",
    description="Lists all active chat sessions for the logged-in user."
)
async def get_chats(current_user: dict = Depends(db.get_current_user)):
    try:
        rows = await db.db_pool.fetch(
            "SELECT id, session_id, title, created_at FROM chats WHERE user_id = $1 ORDER BY created_at DESC",
            current_user["id"],
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return [
        {
            "id": r["id"],
            "session_id": r["session_id"],
            "title": r["title"],
            "created_at": _to_iso(r["created_at"]),
        }
        for r in rows
    ]


@router.delete(
    "/api/chats/{chat_id}",
    tags=["chat"],
    summary="Delete a chat session",
    description="Deletes a chat thread and all its corresponding message history."
)
async def delete_chat(chat_id: int, current_user: dict = Depends(db.get_current_user)):
    owner_row = await db.db_pool.fetchrow(
        "SELECT id FROM chats WHERE id = $1 AND user_id = $2", chat_id, current_user["id"]
    )
    if not owner_row:
        raise HTTPException(status_code=404, detail="Chat not found or access denied.")

    try:
        await db.db_pool.execute("DELETE FROM chats WHERE id = $1", chat_id)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error deleting chat: {str(e)}")

    return {"detail": "Chat deleted successfully"}


# --- Messaging History & Dispatch Endpoints ---

@router.get(
    "/api/chats/history",
    response_model=ChatHistoryResponse,
    tags=["chat"],
    summary="Get chat history",
    description=(
        "Retrieves message history for a user by email. Pass session_id to scope to one "
        "conversation; omit it to get every message across all of that user's chats."
    )
)
async def get_chat_history(
    email: str,
    session_id: Optional[str] = None,
    limit: int = Query(default=200, le=500),
):
    norm_email = email.strip().lower()
    try:
        user_row = await db.db_pool.fetchrow("SELECT id FROM users WHERE email = $1", norm_email)
        if not user_row:
            return ChatHistoryResponse(email=norm_email, messages=[])
        user_id = user_row["id"]

        if session_id:
            chat_rows = await db.db_pool.fetch(
                "SELECT id, session_id FROM chats WHERE user_id = $1 AND session_id = $2",
                user_id, session_id,
            )
        else:
            chat_rows = await db.db_pool.fetch(
                "SELECT id, session_id FROM chats WHERE user_id = $1", user_id
            )
        if not chat_rows:
            return ChatHistoryResponse(email=norm_email, messages=[])

        session_by_chat = {r["id"]: r["session_id"] for r in chat_rows}
        chat_ids = list(session_by_chat.keys())
        # Cap per chat_id (most recent `limit` messages each), not with one
        # limit shared across every chat — otherwise one long-running chat
        # can starve the others out of the response entirely.
        rows = await db.db_pool.fetch(
            """
            SELECT chat_id, sender, text, created_at FROM (
                SELECT chat_id, sender, text, created_at,
                       ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY created_at DESC) AS rn
                FROM messages
                WHERE chat_id = ANY($1::int[])
            ) ranked
            WHERE rn <= $2
            ORDER BY created_at ASC
            """,
            chat_ids, limit,
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    messages = []
    for chat_id, sender, text, created_at in rows:
        sid = session_by_chat.get(chat_id)
        if sender == "user":
            messages.append(ChatHistoryMessage(
                role="user", content=text, session_id=sid, created_at=_to_iso(created_at),
            ))
        else:
            head, groups, tail, obj = cartesian._parse_ai_response_parts(text)
            # Only groups that actually have products get a title + marker +
            # widget — matches what live streaming does (skips empty groups,
            # and sends the group's title right before its products).
            non_empty_groups = [(title, items) for _ptype, title, items in groups if items]
            has_products = bool(non_empty_groups)
            content_type = None
            structured_data = None
            if has_products:
                content_type = (obj.get("intent") if obj else None) or "product_discovery"
                parts = []
                if head:
                    parts.append(" ".join(head))
                for i, (title, items) in enumerate(non_empty_groups, start=1):
                    if title:
                        parts.append(f"**{title}**")
                    parts.append(cartesian._product_widget_marker(i))
                if tail:
                    parts.append(" ".join(tail))
                content = "\n\n".join(parts)
                structured_data = [items for _title, items in non_empty_groups]
            else:
                content = " ".join(head + tail) or text
            messages.append(ChatHistoryMessage(
                role="assistant",
                content=content,
                content_type=content_type,
                structured_data=structured_data,
                session_id=sid,
                created_at=_to_iso(created_at),
            ))
    logger.info("chat_history: email=%s session_id=%s returned %d messages", norm_email, session_id, len(messages))
    return ChatHistoryResponse(email=norm_email, messages=messages)


@router.get(
    "/api/chats/session-products",
    response_model=SessionProductsResponse,
    tags=["chat"],
    summary="Get all products shown in a chat session",
    description=(
        "Returns every product that has been shown (via product_discovery) anywhere in this "
        "chat session, deduped by product_id — powers the frontend's '@product' popup. Backed by "
        "a per-session Redis hash cache; on a cache miss it rebuilds from the chat's saved message "
        "history in Postgres (the real source of truth) and repopulates the cache before returning."
    )
)
async def get_session_products(session_id: str, current_user: dict = Depends(db.get_current_user)):
    chat_row = await db.db_pool.fetchrow(
        "SELECT id FROM chats WHERE session_id = $1 AND user_id = $2",
        session_id, current_user["id"],
    )
    if not chat_row:
        raise HTTPException(status_code=404, detail="Chat not found or access denied.")

    key = cartesian._session_products_key(session_id)
    cached = await db.redis_client.hgetall(key)
    if cached:
        products = [json.loads(v) for v in cached.values()]
        logger.info("session-products: session_id=%s cache hit, %d products", session_id, len(products))
        return SessionProductsResponse(session_id=session_id, products=products)

    products = await cartesian._rebuild_session_products_from_db(chat_row["id"])
    if products:
        await cartesian._cache_session_products(session_id, products)
    logger.info("session-products: session_id=%s cache miss, rebuilt %d products from DB", session_id, len(products))
    return SessionProductsResponse(session_id=session_id, products=products)


@router.post(
    "/api/chats/message",
    tags=["chat"],
    summary="Create/continue a chat and send a message (SSE stream)",
    description=(
        "Combines chat creation and message sending into a single streamed endpoint. "
        "Omit session_id to start a new chat; pass an existing session_id to continue it. "
        "Streams event: chat, status, message, product_discovery (when present), conversation_id, end."
    )
)
async def post_chat_message(request: ChatMessageRequest = Body(...), current_user: dict = Depends(db.get_current_user)):
    return StreamingResponse(
        cartesian._stream_chat_message(request.session_id, request.text, current_user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
