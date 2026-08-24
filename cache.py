"""Thin cache-aside helpers over Redis for read-heavy, poll-heavy dashboard
endpoints (see projects.py, routers/evals.py, routers/testing.py). Short TTLs
are the primary correctness mechanism — these endpoints are already polled by
the frontend on a matching cadence (usePolling), so a few seconds of
staleness is the existing UX, not a regression. A couple of high-value
mutations (project create/set-live) also explicitly invalidate, since
staleness there would be jarring rather than just "wait for the next poll".

Not for anything write-path-critical or exact — this is purely a latency
optimization layered on top of Postgres as the source of truth, never a
replacement for it (see db.py for why the reverse pattern, DB features
depending on Redis being up, is intentionally avoided elsewhere too).
"""
import json
import logging

import db

logger = logging.getLogger("columbia_backend.cache")


async def get_json(key: str):
    """Returns the cached value, or None on a miss OR a Redis error — a cache
    failure should degrade to 'go hit Postgres', never surface as a 500."""
    try:
        raw = await db.redis_client.get(key)
    except Exception as e:
        logger.warning("cache: get failed for key=%s: %s", key, e)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def set_json(key: str, value, ttl_seconds: int) -> None:
    try:
        await db.redis_client.setex(key, ttl_seconds, json.dumps(value, default=str))
    except Exception as e:
        logger.warning("cache: set failed for key=%s: %s", key, e)


async def delete(*keys: str) -> None:
    if not keys:
        return
    try:
        await db.redis_client.delete(*keys)
    except Exception as e:
        logger.warning("cache: delete failed for keys=%s: %s", keys, e)
