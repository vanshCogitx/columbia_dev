import os
import uuid
import json
import re
import random
import asyncio
import contextlib
import logging
from typing import AsyncGenerator, Optional

import httpx

import db

logger = logging.getLogger("columbia_backend")

CARTESIAN_JOB_PATH = "/exports/rest-api/6a60f95529131252a1e0746a/jobs"
STATUS_MESSAGES = [
    "Illuminating your best options...",
    "Curating a shortlist for you...",
    "Synthesizing your preferences...",
    "Enumerating possible matches...",
    "Cross-referencing prices and availability...",
    "Calibrating your recommendations...",
    "Orchestrating your product search...",
    "Distilling this down to the best picks...",
    "Reconciling sizes and styles...",
    "Corroborating product details...",
    "Optimizing your results...",
    "Harmonizing your search criteria...",
    "Consolidating the top picks...",
    "Excavating a few hidden gems...",
    "Scrutinizing every detail...",
    "Evaluating the best fit...",
    "Appraising your options...",
    "Tabulating available styles...",
    "Aggregating recommendations...",
    "Formulating your shortlist...",
    "Deducing your perfect match...",
    "Discerning the best quality options...",
    "Pinpointing the exact matches...",
    "Extrapolating from your preferences...",
    "Deliberating over the finest choices...",
    "Prioritizing what matters most to you...",
    "Contextualizing your request...",
    "Triangulating the best value...",
    "Unearthing something you'll love...",
    "Finalizing your curated selection...",
]


def _product_widget_marker(index: int) -> str:
    """Position marker for the Nth product group in a history message's
    text, so the frontend can splice each group's widget into the exact
    spot it belongs (matching structured_data[index - 1]) instead of only
    ever bundling every group into one widget up front."""
    return f":::PRODUCT_WIDGET_{index}:::"


def _comparison_widget_marker() -> str:
    """Position marker for a 'comparison' intent's comparison data in a
    history message's text (same purpose as _product_widget_marker) — only
    ever one comparison per message, so no index needed."""
    return ":::PRODUCT_COMPARISON:::"


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_NON_NARRATIVE_KEYS = {"products", "intent", "status", "contentType", "type", "comparison"}


def _extract_ai_text(output_obj: dict) -> Optional[str]:
    """variables.text is the clean, flat JSON string matching the intent's
    schema. workflow_response.content is sometimes a nested dict (e.g.
    {"result": {...fields double-encoded...}}) rather than a string, which
    breaks downstream str-only consumers (DB insert, _parse_ai_response).
    Prefer text; only fall back to content if it's actually a string."""
    text = output_obj.get("variables", {}).get("text")
    if isinstance(text, str):
        return text
    content = output_obj.get("workflow_response", {}).get("content")
    if isinstance(content, str):
        return content
    return None


async def _call_cartesian_workflow(client: httpx.AsyncClient, url: str, poll_url_base: str, params: dict, payload: dict, headers: dict) -> tuple[Optional[str], Optional[dict]]:
    """Runs the full Cartesian call (initial POST, then the poll loop if the
    job was accepted for async processing). Returns (ai_text, raw_output) —
    raw_output is the full 'output' object (executionSummary, evaluatedBranches,
    etc.) alongside the extracted text, so callers can persist it for later use
    by the evals judge (see judge.py) even though ai_text is None/None if
    nothing usable came back. Runs as a background task so the caller can tick
    status messages independently of this function's cadence."""
    response = await client.post(url, json=payload, params=params, headers=headers, timeout=40)
    response.raise_for_status()
    data = response.json()
    logger.info("cartesian: initial response status=%s body=%s", response.status_code, data)
    resp_data = data.get("data", data)

    if resp_data.get("ok") and resp_data.get("accepted"):
        run_id = resp_data.get("runId")
        poll_url = f"{poll_url_base}/{run_id}"
        logger.info("cartesian: job accepted async, runId=%s, polling %s", run_id, poll_url)
        # 60 * 2s = 120s. Was 30 (60s) — the workflow swapped in has 15-20+
        # chained agent nodes (vs. the older, smaller one this limit was
        # originally sized for) and some individual nodes alone take 10+
        # seconds; 60s was observed timing out mid-execution with several
        # nodes still QUEUED, turning "slow" into an outright failure.
        max_retries = 60
        for attempt in range(max_retries):
            await asyncio.sleep(2)
            poll_resp = await client.get(poll_url, headers=headers, timeout=10)
            poll_resp.raise_for_status()
            poll_data_raw = poll_resp.json()
            poll_data = poll_data_raw.get("data", poll_data_raw)
            logger.info(
                "cartesian: poll attempt %d/%d isCompleted=%s body=%s",
                attempt + 1, max_retries, poll_data.get("isCompleted"), poll_data_raw,
            )
            if poll_data.get("isCompleted"):
                output = poll_data.get("output", {})
                return _extract_ai_text(output), output
        logger.error("cartesian: timed out after %d poll attempts", max_retries)
        return None, None

    if resp_data.get("ok") and not resp_data.get("accepted"):
        logger.info("cartesian: completed synchronously within wait window")
        output = resp_data.get("output", {})
        return _extract_ai_text(output), output

    logger.error("cartesian: no ai_text extractable from response: %s", data)
    return None, None


_PRODUCT_ARRAY_STRING_FIELDS = (
    "available_colors", "available_sizes", "available_materials",
    "available_features", "available_fits", "tags",
)


def _coerce_products_list(structured: dict) -> None:
    """Cartesian sometimes double-encodes 'products' as a JSON string instead
    of a real array. Parse it in place so _flatten_products/_normalize_product_arrays
    (which both require an actual list) don't silently no-op on it."""
    products = structured.get("products")
    if isinstance(products, str):
        try:
            parsed = json.loads(products)
        except json.JSONDecodeError:
            return
        if isinstance(parsed, list):
            structured["products"] = parsed


def _flatten_products(structured: dict) -> None:
    """Every intent except 'budget' sends products as a flat list of product
    dicts. 'budget' wraps them one level deeper as [{query, items:[...]}, ...].
    Flatten that here so the frontend's product grid renderer never has to
    branch on intent."""
    products = structured.get("products")
    if not isinstance(products, list) or not products:
        return
    if isinstance(products[0], dict) and "items" in products[0]:
        flat = []
        for group in products:
            if isinstance(group, dict) and isinstance(group.get("items"), list):
                flat.extend(item for item in group["items"] if isinstance(item, dict))
        structured["products"] = flat


def _normalize_product_arrays(structured: dict) -> None:
    """Cartesian sends list-shaped fields (colors, sizes, tags, ...) as a
    stringified JSON array (e.g. '["Black", "Blue"]') instead of a real array.
    Parse those in place so the frontend gets actual arrays."""
    for product in structured.get("products") or []:
        if not isinstance(product, dict):
            continue
        for field in _PRODUCT_ARRAY_STRING_FIELDS:
            value = product.get(field)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list):
                    product[field] = parsed


def _extract_structured_obj(ai_text: str) -> Optional[dict]:
    """Parses ai_text (optionally ```json-fenced) into the structured dict
    Cartesian sends for product replies ({"introduction","orientation",
    "opening","status","products","recovery","closing"}), normalizing the
    products field in place. Returns None for plain prose or anything that
    isn't a JSON object."""
    candidate = ai_text.strip()
    fence_match = _CODE_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    # The 'comparison' intent (and possibly others) wraps every field one
    # level deeper under a single 'result' key instead of putting them at
    # the top level like every other intent does — unwrap it here so every
    # downstream consumer can keep treating the object uniformly.
    if set(obj.keys()) == {"result"} and isinstance(obj["result"], dict):
        obj = obj["result"]
    _coerce_products_list(obj)
    _flatten_products(obj)
    _normalize_product_arrays(obj)
    return obj


def _is_in_stock(product: dict) -> bool:
    """An out-of-stock item is still a real, valid product entry (Cartesian
    just flags it) — it should be the ONE thing omitted from a list, never
    a reason to drop the whole list. Checks 'in_stock' first (the more
    reliable boolean when present), falls back to the 'availability'
    string; defaults to keeping the item if neither field is present,
    rather than hiding something we can't actually confirm is out of stock."""
    in_stock = product.get("in_stock")
    if isinstance(in_stock, bool):
        return in_stock
    availability = product.get("availability")
    if isinstance(availability, str):
        return availability.strip().lower() != "out_of_stock"
    return True


def _iter_product_groups(products) -> list[tuple[Optional[str], Optional[str], list]]:
    """Yields (product_type, title, flat_product_list) triples so the caller
    can stream each group as its own product_discovery event with a plain
    (non-nested) product array. 'title' is the group's display-friendly
    label (e.g. "Caps" vs the more technical product_type "Cap") — sent
    separately as a message right before that group's products. Handles
    both the grouped shape ([{"product_type", "title", "products": [...]},
    ...]) and an already-flat list of product dicts (e.g. budget intent,
    post-_flatten_products). Out-of-stock items are filtered out here, per
    item — not by dropping the group/list they came in — see _is_in_stock."""
    if not isinstance(products, list) or not products:
        return []
    if isinstance(products[0], dict) and isinstance(products[0].get("products"), list):
        groups = []
        for group in products:
            if not isinstance(group, dict):
                continue
            ptype = group.get("product_type")
            title = group.get("title")
            items = [p for p in group.get("products", []) if isinstance(p, dict) and _is_in_stock(p)]
            if ptype:
                for item in items:
                    item.setdefault("product_type", ptype)
            groups.append((ptype, title, items))
        return groups
    return [(None, None, [p for p in products if isinstance(p, dict) and _is_in_stock(p)])]


def _iter_comparison_items(comparison) -> list[dict]:
    """Flat list of items from a 'comparison' intent's 'comparison' field
    (e.g. two gloves being compared side by side). Streamed as its own
    comparison event — distinct from product_discovery, since this isn't a
    browsable catalog result the frontend renders as a product grid, it's a
    fixed side-by-side comparison."""
    if not isinstance(comparison, list):
        return []
    return [item for item in comparison if isinstance(item, dict)]


def _parse_ai_response_parts(ai_text: str) -> tuple[list[str], list[tuple[Optional[str], Optional[str], list]], list[str], list[dict], Optional[dict]]:
    """Streaming variant of _parse_ai_response: instead of hardcoding which
    field names count as narrative text (introduction/orientation/opening/
    recovery/closing), walks the object's own keys in order and treats any
    non-empty string value — or list of strings (e.g. 'choose' on a
    comparison reply, where several recommendation paragraphs share one key
    instead of each getting their own) — as a message — so a new or renamed
    narrative field from Cartesian is picked up automatically instead of
    silently dropped. 'products'/'comparison' are pulled out separately;
    'status'/'intent'/'contentType' are metadata, not display text.
    Returns (head, groups, tail, comparison_items, obj): head = narrative
    strings that appeared before 'products' in the object, tail = the ones
    that appeared after, obj = the raw parsed object (None for plain-prose
    responses) so callers can still pull metadata like 'intent' off it if
    needed."""
    obj = _extract_structured_obj(ai_text)
    if obj is None:
        return [ai_text], [], [], [], None
    head: list[str] = []
    tail: list[str] = []
    seen_products = False
    for key, value in obj.items():
        if key == "products":
            seen_products = True
            continue
        if key in _NON_NARRATIVE_KEYS:
            continue
        if isinstance(value, str) and value:
            (tail if seen_products else head).append(value)
        elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            (tail if seen_products else head).append("\n\n".join(value))
    if not head and not tail:
        status = obj.get("status")
        if isinstance(status, dict) and status.get("message"):
            head = [status["message"]]
        else:
            head = [ai_text]
    groups = _iter_product_groups(obj.get("products"))
    comparison_items = _iter_comparison_items(obj.get("comparison"))
    return head, groups, tail, comparison_items, obj


SESSION_PRODUCTS_TTL_SECONDS = 86400  # cache hygiene only — Postgres (messages table) is the real source of truth


def _session_products_key(session_id: str) -> str:
    return f"session_products:{session_id}"


async def _cache_session_products(session_id: str, items: list[dict]) -> None:
    """Pushes newly-shown products into this session's Redis hash, keyed by
    product_id so re-showing the same product just overwrites its field
    instead of creating a duplicate. Refreshes the TTL on every push so an
    actively-used session's cache doesn't go cold mid-conversation."""
    mapping = {p["product_id"]: json.dumps(p) for p in items if p.get("product_id")}
    if not mapping:
        return
    key = _session_products_key(session_id)
    await db.redis_client.hset(key, mapping=mapping)
    await db.redis_client.expire(key, SESSION_PRODUCTS_TTL_SECONDS)


async def _rebuild_session_products_from_db(chat_id: int) -> list[dict]:
    """Cache-miss fallback: replays every AI message ever saved for this
    chat, re-parsing each one the same way live streaming does, and
    flattens/dedupes every product group across the whole conversation.
    Postgres already holds this permanently (messages.text) — Redis is only
    ever a disposable speed layer in front of it."""
    rows = await db.db_pool.fetch(
        "SELECT text FROM messages WHERE chat_id = $1 AND sender = 'ai' ORDER BY created_at ASC",
        chat_id,
    )
    by_id: dict[str, dict] = {}
    for (text,) in rows:
        _head, groups, _tail, _comparison_items, _obj = _parse_ai_response_parts(text)
        for _product_type, _title, items in groups:
            for p in items:
                if p.get("product_id"):
                    by_id[p["product_id"]] = p
    return list(by_id.values())


async def _save_message_trace(message_id: int, raw_output: Optional[dict]) -> None:
    """Persists Cartesian's raw 'output' object (executionSummary,
    evaluatedBranches, etc.) alongside a saved AI message — needed by the
    evals judge (judge.py) later, since feedback can arrive long after the
    run is over and we can't re-fetch this from Cartesian at that point."""
    if not raw_output:
        return
    await db.db_pool.execute(
        """
        INSERT INTO message_traces (message_id, raw_output, evaluated_branches, execution_summary)
        VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb)
        ON CONFLICT (message_id) DO NOTHING
        """,
        message_id,
        json.dumps(raw_output),
        json.dumps(raw_output.get("evaluatedBranches")),
        json.dumps(raw_output.get("executionSummary")),
    )


PENDING_RESPONSE_TTL_SECONDS = 300  # safety-net expiry only, in case the process dies before the finally block runs


def _pending_response_key(chat_id: int) -> str:
    return f"chat_pending:{chat_id}"


async def is_response_pending(chat_id: int) -> bool:
    """Whether a background _generate_and_save_response task is currently
    in flight for this chat — lets a client reopening a chat mid-generation
    tell 'still working on it' apart from 'this just has no reply'."""
    return bool(await db.redis_client.exists(_pending_response_key(chat_id)))


def _pending_status_key(chat_id: int) -> str:
    return f"chat_status:{chat_id}"


async def get_pending_status(chat_id: int) -> Optional[str]:
    """Current rotating status phrase (e.g. 'Searching the catalog...') for an
    in-flight response, written by _status_ticker. Lets a client that reopens
    a chat mid-generation show the same live-feeling status text a connected
    SSE stream would have shown, instead of a generic 'loading' spinner."""
    return await db.redis_client.get(_pending_status_key(chat_id))


async def _status_ticker(chat_id: int) -> None:
    """Rotates through STATUS_MESSAGES into Redis every 3-5s for as long as
    it runs. Started alongside _generate_and_save_response and cancelled in
    its finally block — deliberately decoupled from any client connection,
    so both a live _stream_chat_message viewer and a client polling
    /api/chats/history after reconnecting read the exact same rotating text
    from here rather than each keeping their own independently-random copy."""
    status_key = _pending_status_key(chat_id)
    shuffled = random.sample(STATUS_MESSAGES, len(STATUS_MESSAGES))
    idx = 0
    while True:
        await db.redis_client.setex(status_key, PENDING_RESPONSE_TTL_SECONDS, shuffled[idx % len(shuffled)])
        idx += 1
        if idx % len(shuffled) == 0:
            shuffled = random.sample(STATUS_MESSAGES, len(STATUS_MESSAGES))
        await asyncio.sleep(random.uniform(3, 5))


async def _generate_and_save_response(
    chat_id: int, session_id: str, title: str, text: str, current_user: dict
) -> dict:
    """Does all of the real work for one turn: saves the user message, calls
    the Cartesian workflow, parses the response, and saves it. Deliberately
    NOT a generator and NOT tied to the client's HTTP connection — it's run
    as an independent asyncio task by _stream_chat_message (see there for
    why: a client disconnecting mid-request — tab switch, backgrounding,
    network blip — must not abandon an in-flight response partway through).
    Returns {"error": str} on failure, or {"head", "groups", "tail"} on
    success, for the caller to turn into SSE events."""
    pending_key = _pending_response_key(chat_id)
    status_key = _pending_status_key(chat_id)
    await db.redis_client.setex(pending_key, PENDING_RESPONSE_TTL_SECONDS, "1")
    ticker_task = asyncio.create_task(_status_ticker(chat_id))
    try:
        # 2. Save user message
        user_msg_row = await db.db_pool.fetchrow(
            "INSERT INTO messages (chat_id, sender, text) VALUES ($1, $2, $3) RETURNING id",
            chat_id, 'user', text,
        )
        user_msg_id = user_msg_row["id"]

        # 3. Auto-title from first message
        if title == "New Chat":
            new_title = text[:40] + ("..." if len(text) > 40 else "")
            await db.db_pool.execute("UPDATE chats SET title = $1 WHERE id = $2", new_title, chat_id)

        # 4. Compile history + profile into the prompt sent to Cartesian
        history_rows = await db.db_pool.fetch(
            "SELECT sender, text FROM messages WHERE chat_id = $1 AND id < $2 ORDER BY created_at ASC",
            chat_id, user_msg_id,
        )
        profile_header = f"--- User Profile ---\n- Name: {current_user['name']}\n\n"
        if history_rows:
            history_str = "\n".join(
                f"[{'User' if sender == 'user' else 'AI'}]: {msg}" for sender, msg in history_rows
            )
            history_block = f"--- Chat History ---\n{history_str}\n\n"
        else:
            history_block = ""
        compiled_text = (
            "System Instruction: Below is the context of the logged-in user and the chat history of this conversation. "
            "Please use this information to answer the user's latest query at the end. "
            "Respond ONLY to the latest query and do not repeat the context or history.\n\n"
            f"{profile_header}{history_block}"
            f"--- Latest User Query ---\n[User]: {text}"
        )
        logger.info("chat_message: compiled prompt (%d chars): %r", len(compiled_text), compiled_text)

        # 5. Call Cartesian workflow
        base_url = os.getenv("CARTESIAN_BASE_URL", "https://api.cartesian.ai")
        client_id = os.getenv("CARTESIAN_CLIENT_ID")
        client_secret = os.getenv("CARTESIAN_CLIENT_SECRET")
        if not client_id or not client_secret:
            logger.error("chat_message: Cartesian API credentials missing in environment")
            return {"error": "Cartesian API credentials are missing."}

        headers = {
            "x-client-id": client_id,
            "x-client-secret": client_secret,
            "Content-Type": "application/json",
        }
        params = {"waitSeconds": 30, "sessionId": session_id}
        payload = {"payload": {"user_query": compiled_text, "user_id": current_user["email"]}}
        url = f"{base_url.rstrip('/')}{CARTESIAN_JOB_PATH}"
        poll_url_base = f"{base_url.rstrip('/')}{CARTESIAN_JOB_PATH}"

        logger.info("cartesian: POST %s params=%s", url, params)

        try:
            async with httpx.AsyncClient() as client:
                ai_text, raw_output = await _call_cartesian_workflow(client, url, poll_url_base, params, payload, headers)
        except httpx.HTTPError as e:
            logger.error("cartesian: HTTP error communicating with Cartesian: %s", e)
            return {"error": f"AI service communication error: {str(e)}"}

        if ai_text is None:
            logger.error("cartesian: no ai_text extractable from response")
            return {"error": "Unexpected response from the AI service."}

        logger.info("cartesian: extracted ai_text (%d chars): %r", len(ai_text), ai_text)

        # 6. Save AI response
        ai_msg_row = await db.db_pool.fetchrow(
            "INSERT INTO messages (chat_id, sender, text) VALUES ($1, $2, $3) RETURNING id",
            chat_id, 'ai', ai_text,
        )
        await _save_message_trace(ai_msg_row["id"], raw_output)

        head, groups, tail, comparison_items, obj = _parse_ai_response_parts(ai_text)
        product_count = sum(len(items) for _, _, items in groups)
        response_type = obj.get("type") if obj else None
        logger.info(
            "chat_message: parsed response head=%d groups=%d product_count=%d tail=%d comparison_items=%d type=%r",
            len(head), len(groups), product_count, len(tail), len(comparison_items), response_type,
        )
        return {
            "head": head, "groups": groups, "tail": tail, "comparison_items": comparison_items,
            "type": response_type, "message_id": ai_msg_row["id"],
        }
    except Exception as e:
        # Belt-and-braces: this runs detached from any client connection, so
        # an unhandled exception here would otherwise just vanish into an
        # "exception was never retrieved" asyncio warning instead of being
        # visible anywhere.
        logger.error("chat_message: _generate_and_save_response failed: %s", e)
        return {"error": f"Internal error while generating the response: {str(e)}"}
    finally:
        ticker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker_task
        await db.redis_client.delete(pending_key)
        await db.redis_client.delete(status_key)


async def _stream_chat_message(session_id: Optional[str], text: str, current_user: dict) -> AsyncGenerator[str, None]:
    logger.info(
        "chat_message: request received user=%s(%s) session_id=%s text=%r",
        current_user["email"], current_user["id"], session_id, text,
    )
    # 1. Resolve chat: create one if session_id omitted, else verify ownership
    if session_id is None:
        session_id = str(uuid.uuid4())
        title = "New Chat"
        row = await db.db_pool.fetchrow(
            "INSERT INTO chats (user_id, session_id, title) VALUES ($1, $2, $3) RETURNING id",
            current_user["id"], session_id, title,
        )
        chat_id = row["id"]
        logger.info("chat_message: created new chat_id=%s session_id=%s", chat_id, session_id)
    else:
        chat_row = await db.db_pool.fetchrow(
            "SELECT id, title FROM chats WHERE session_id = $1 AND user_id = $2",
            session_id, current_user["id"],
        )
        if not chat_row:
            logger.warning("chat_message: session_id=%s not found for user_id=%s", session_id, current_user["id"])
            yield f"event: error\ndata: {json.dumps({'detail': 'Chat not found or access denied.'})}\n\n"
            return
        chat_id, title = chat_row["id"], chat_row["title"]
        logger.info("chat_message: continuing chat_id=%s session_id=%s", chat_id, session_id)

    yield f"event: chat\ndata: {json.dumps({'v': {'id': chat_id, 'session_id': session_id, 'title': title}})}\n\n"

    # 2-6 (save user message, call Cartesian, save AI response) run as an
    # independent task, created now so it's already underway. If the client
    # disconnects (tab switch, backgrounding, network blip) while we're
    # still waiting below, this task is NOT cancelled — it keeps running and
    # still saves the response. Only our ability to *stream it live* is
    # lost; the data itself is safe, and a later /api/chats/history call
    # will show the completed response.
    work_task = asyncio.create_task(
        _generate_and_save_response(chat_id, session_id, title, text, current_user)
    )

    yield f"event: status\ndata: {json.dumps({'v': random.choice(STATUS_MESSAGES)})}\n\n"

    # Status ticks are read from the same Redis-backed ticker that
    # _generate_and_save_response starts (see _status_ticker) rather than
    # generated locally here — so a live viewer and a client that
    # disconnects and later polls /api/chats/history's pending_status field
    # see the identical rotating text, not two independently-random copies.
    # Polled frequently (short timeout) but only yielded on change, so the
    # visible cadence still matches the ticker's own 3-5s rotation.
    last_status = None
    while not work_task.done():
        done, _ = await asyncio.wait({work_task}, timeout=1.5)
        if work_task in done:
            break
        current_status = await get_pending_status(chat_id)
        if current_status and current_status != last_status:
            yield f"event: status\ndata: {json.dumps({'v': current_status})}\n\n"
            last_status = current_status

    # shield: even if *this* generator gets cancelled right at this exact
    # await (client disconnects at this instant), work_task must not be —
    # it's already saving/about to save the response.
    result = await asyncio.shield(work_task)

    if "error" in result:
        yield f"event: error\ndata: {json.dumps({'detail': result['error']})}\n\n"
        return

    head, groups, tail = result["head"], result["groups"], result["tail"]
    comparison_items = result.get("comparison_items") or []

    # add_to_cart is a distinct UX moment (confirming items just added):
    # the confirmation text goes out as its own add_to_cart event, and the
    # items themselves reuse the normal product_discovery event/shape —
    # same as any other product-bearing reply — so the frontend renders
    # them with its existing product widget. Everything else (including
    # out_of_scope) falls through to the normal narrative + product_discovery
    # streaming below, unchanged.
    if result.get("type") == "add_to_cart":
        message = " ".join(head) if head else None
        yield f"event: add_to_cart\ndata: {json.dumps({'v': {'message': message}})}\n\n"
        if comparison_items:
            yield f"event: comparison\ndata: {json.dumps({'v': comparison_items})}\n\n"
        for _product_type, group_title, items in groups:
            if items:
                if group_title:
                    title_text = f"\n\n**{group_title}**"
                    yield f"event: message\ndata: {json.dumps({'v': title_text})}\n\n"
                yield f"event: product_discovery\ndata: {json.dumps({'v': items})}\n\n"
                await _cache_session_products(session_id, items)
        if tail:
            tail_text = " ".join(tail)
            yield f"event: message\ndata: {json.dumps({'v': tail_text})}\n\n"
        yield f"event: conversation_id\ndata: {json.dumps({'v': session_id})}\n\n"
        # The real, saved database id of the AI message just generated — sent
        # as its own event so the frontend has a real id to attach to this
        # message (e.g. for the thumbs up/down feedback call) instead of having
        # to fabricate a client-side placeholder (a real bug this was tracked
        # down from: the frontend was sending a Date.now() timestamp as
        # message_id, which crashed POST /api/chats/feedback's int4 column).
        yield f"event: msg_id\ndata: {json.dumps({'v': result['message_id']})}\n\n"
        yield f"event: end\ndata: {json.dumps({'v': {}})}\n\n"
        logger.info("chat_message: stream complete (add_to_cart) chat_id=%s session_id=%s", chat_id, session_id)
        return

    if comparison_items:
        # Comparison replies read as: here are the two products, THEN the
        # narrative ("choose X for Y, opt for Z for W" + a closing line) —
        # comparison first matches that reading order, and every narrative
        # field (choose, closing, ...) gets its own event: message instead
        # of being mashed into one paragraph via " ".join, so the closing
        # line doesn't visually disappear into the recommendation text.
        yield f"event: comparison\ndata: {json.dumps({'v': comparison_items})}\n\n"
        for text in head:
            yield f"event: message\ndata: {json.dumps({'v': text})}\n\n"
    elif head:
        head_text = " ".join(head)
        yield f"event: message\ndata: {json.dumps({'v': head_text})}\n\n"
    for _product_type, group_title, items in groups:
        if items:
            if group_title:
                title_text = f"\n\n**{group_title}**"
                yield f"event: message\ndata: {json.dumps({'v': title_text})}\n\n"
            yield f"event: product_discovery\ndata: {json.dumps({'v': items})}\n\n"
            await _cache_session_products(session_id, items)
    if tail:
        if comparison_items:
            for text in tail:
                yield f"event: message\ndata: {json.dumps({'v': text})}\n\n"
        else:
            tail_text = " ".join(tail)
            yield f"event: message\ndata: {json.dumps({'v': tail_text})}\n\n"

    yield f"event: conversation_id\ndata: {json.dumps({'v': session_id})}\n\n"
    # The real, saved database id of the AI message just generated — sent
    # as its own event so the frontend has a real id to attach to this
    # message (e.g. for the thumbs up/down feedback call) instead of having
    # to fabricate a client-side placeholder (a real bug this was tracked
    # down from: the frontend was sending a Date.now() timestamp as
    # message_id, which crashed POST /api/chats/feedback's int4 column).
    yield f"event: msg_id\ndata: {json.dumps({'v': result['message_id']})}\n\n"
    yield f"event: end\ndata: {json.dumps({'v': {}})}\n\n"
    logger.info("chat_message: stream complete chat_id=%s session_id=%s", chat_id, session_id)
