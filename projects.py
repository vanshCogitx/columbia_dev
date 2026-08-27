"""Multi-project support for the evals dashboard — a "project" is a distinct
Cartesian workflow (its own export id + credentials). Deliberately minimal:
only Testing (simulation.py) actually calls Cartesian using a project's own
credentials via get_project_cartesian_config. The live chat pipeline
(cartesian.py's _generate_and_save_response) is untouched by this module and
never references projects in any way — chats/messages have no project
column at all. The one link between real chat and a project is is_live: the
project currently flagged live is snapshotted onto response_feedback.
project_id at feedback-creation time (see routers/chat.py's post_feedback),
so real feedback can still be attributed to a project without chats itself
ever depending on one existing.
"""
import json
import logging
import os
from datetime import timezone
from typing import Optional

from fastapi import HTTPException

import cache
import db

logger = logging.getLogger("columbia_backend.projects")

# Cache-aside for project reads (see cache.py) — these are polled constantly
# by the dashboard (project switcher, every project-scoped page's
# get_project_or_404 check) and rarely change, so a short TTL absorbs almost
# all of that traffic without ever risking staleness beyond a few seconds.
_LIST_CACHE_KEY = "evalcache:projects:list"
_PROJECT_CACHE_TTL = 10


def _project_cache_key(project_id: int) -> str:
    return f"evalcache:project:{project_id}"


async def get_project_cartesian_config(project_id: int) -> dict:
    """Returns {'client_id', 'client_secret', 'base_url', 'job_path'} for a
    project. Raises ValueError if the project doesn't exist. base_url falls
    back to the global CARTESIAN_BASE_URL env default when the project has
    no override, matching the fallback _generate_and_save_response already
    used before this module existed."""
    row = await db.db_pool.fetchrow(
        "SELECT cartesian_client_id, cartesian_client_secret, cartesian_base_url, cartesian_export_id "
        "FROM projects WHERE id = $1",
        project_id,
    )
    if not row:
        raise ValueError(f"project_id={project_id} not found")
    base_url = row["cartesian_base_url"] or os.getenv("CARTESIAN_BASE_URL", "https://api.cartesian.ai")
    return {
        "client_id": row["cartesian_client_id"],
        "client_secret": row["cartesian_client_secret"],
        "base_url": base_url,
        "job_path": f"/exports/rest-api/{row['cartesian_export_id']}/jobs",
    }


async def get_project_or_404(project_id: int) -> dict:
    """FastAPI Depends-friendly existence check — used by routers/evals.py
    and routers/testing.py so every project-scoped endpoint 404s early
    instead of returning an empty result for a bad project_id. Reuses
    get_project's cache entry (same key) when present instead of issuing its
    own query — this is the dependency almost every project-scoped request
    pays for, so keeping it cache-aware matters more than any single route's
    own caching."""
    cached = await cache.get_json(_project_cache_key(project_id))
    if cached is not None:
        return {"id": cached["id"], "name": cached["name"]}
    row = await db.db_pool.fetchrow("SELECT id, name FROM projects WHERE id = $1", project_id)
    if not row:
        raise HTTPException(status_code=404, detail="Project not found.")
    return dict(row)


def _row_to_dict(row, node_count: int = 0, edge_count: int = 0) -> dict:
    created_at = row["created_at"]
    if created_at.tzinfo is not None:
        created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    return {
        "id": row["id"],
        "name": row["name"],
        "cartesian_export_id": row["cartesian_export_id"],
        "cartesian_base_url": row["cartesian_base_url"],
        "is_live": row["is_live"],
        "created_at": created_at.isoformat(),
        "node_count": node_count,
        "edge_count": edge_count,
    }


async def list_projects() -> list[dict]:
    cached = await cache.get_json(_LIST_CACHE_KEY)
    if cached is not None:
        return cached
    rows = await db.db_pool.fetch(
        """
        SELECT p.id, p.name, p.cartesian_export_id, p.cartesian_base_url, p.is_live, p.created_at,
               (SELECT COUNT(*) FROM workflow_nodes wn
                JOIN workflow_versions wv ON wv.id = wn.workflow_version_id
                WHERE wv.project_id = p.id AND wv.is_active = true) AS node_count,
               (SELECT COUNT(*) FROM workflow_edges we
                JOIN workflow_versions wv ON wv.id = we.workflow_version_id
                WHERE wv.project_id = p.id AND wv.is_active = true) AS edge_count
        FROM projects p
        ORDER BY p.created_at ASC
        """
    )
    result = [_row_to_dict(r, r["node_count"], r["edge_count"]) for r in rows]
    await cache.set_json(_LIST_CACHE_KEY, result, _PROJECT_CACHE_TTL)
    return result


async def get_project(project_id: int) -> Optional[dict]:
    key = _project_cache_key(project_id)
    cached = await cache.get_json(key)
    if cached is not None:
        return cached
    row = await db.db_pool.fetchrow(
        """
        SELECT p.id, p.name, p.cartesian_export_id, p.cartesian_base_url, p.is_live, p.created_at,
               (SELECT COUNT(*) FROM workflow_nodes wn
                JOIN workflow_versions wv ON wv.id = wn.workflow_version_id
                WHERE wv.project_id = p.id AND wv.is_active = true) AS node_count,
               (SELECT COUNT(*) FROM workflow_edges we
                JOIN workflow_versions wv ON wv.id = we.workflow_version_id
                WHERE wv.project_id = p.id AND wv.is_active = true) AS edge_count
        FROM projects p
        WHERE p.id = $1
        """,
        project_id,
    )
    if not row:
        return None
    result = _row_to_dict(row, row["node_count"], row["edge_count"])
    await cache.set_json(key, result, _PROJECT_CACHE_TTL)
    return result


async def get_live_project_id() -> Optional[int]:
    """Snapshot target for response_feedback.project_id (see routers/chat.py's
    post_feedback) — returns None if no project is currently live, in which
    case feedback just saves with project_id NULL rather than failing."""
    row = await db.db_pool.fetchrow("SELECT id FROM projects WHERE is_live = true LIMIT 1")
    return row["id"] if row else None


async def set_live_project(project_id: int) -> dict:
    """Flags exactly one project as live at a time — unsets every other
    project first, in the same transaction, so idx_projects_one_live's
    partial-unique constraint can never be violated mid-update."""
    async with db.db_pool.acquire() as conn:
        async with conn.transaction():
            exists = await conn.fetchval("SELECT 1 FROM projects WHERE id = $1", project_id)
            if not exists:
                raise ValueError(f"project_id={project_id} not found")
            all_ids = [r["id"] for r in await conn.fetch("SELECT id FROM projects")]
            await conn.execute("UPDATE projects SET is_live = false WHERE is_live = true")
            await conn.execute("UPDATE projects SET is_live = true WHERE id = $1", project_id)
    logger.info("projects: set_live_project project_id=%s", project_id)
    # is_live flips on (at least) two rows here (the old live project and the
    # new one) — staleness on that specific field would be actively
    # misleading rather than just "a few seconds behind", so this is one of
    # the few writes that explicitly invalidates rather than waiting out the
    # TTL. Cheap: this is a rare, deliberate action, not a hot path.
    await cache.delete(_LIST_CACHE_KEY, *(_project_cache_key(pid) for pid in all_ids))
    return await get_project(project_id)


async def create_project(
    name: str, cartesian_client_id: str, cartesian_client_secret: str,
    cartesian_export_id: str, cartesian_base_url: Optional[str],
    workflow_json: dict, created_by_user_id: int,
) -> dict:
    """Creates a project row and ingests its workflow JSON (workflow_versions/
    nodes/edges) in one transaction — the API equivalent of running
    tools/ingest_workflow.py by hand, so the dashboard's "New Project" form
    doesn't require CLI access. Reuses the exact same node/edge field
    extraction as that script."""
    nodes = workflow_json.get("nodes", [])
    edges = workflow_json.get("edges", [])
    workflow_name = workflow_json.get("name", name)
    description = workflow_json.get("description")

    async with db.db_pool.acquire() as conn:
        async with conn.transaction():
            project_row = await conn.fetchrow(
                """
                INSERT INTO projects (name, cartesian_export_id, cartesian_client_id, cartesian_client_secret, cartesian_base_url, created_by_user_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                name, cartesian_export_id, cartesian_client_id, cartesian_client_secret, cartesian_base_url, created_by_user_id,
            )
            project_id = project_row["id"]

            # Scoped by project_id, unlike tools/ingest_workflow.py's
            # historical unconditional version of this line — a brand new
            # project has nothing to deactivate yet, but this keeps the
            # invariant correct if this project is ever re-ingested later.
            await conn.execute(
                "UPDATE workflow_versions SET is_active = false WHERE is_active = true AND project_id = $1",
                project_id,
            )

            version_row = await conn.fetchrow(
                """
                INSERT INTO workflow_versions (name, description, raw_json, is_active, project_id)
                VALUES ($1, $2, $3::jsonb, true, $4)
                RETURNING id
                """,
                workflow_name, description, json.dumps(workflow_json), project_id,
            )
            version_id = version_row["id"]

            for n in nodes:
                config = n.get("config", {}) or {}
                await conn.execute(
                    """
                    INSERT INTO workflow_nodes
                        (workflow_version_id, node_id, node_type, alias, agent_instructions,
                         model, provider, temperature, raw_config)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                    """,
                    version_id, n.get("id"), n.get("type"), n.get("alias"),
                    config.get("agentInstructions"), config.get("model"),
                    config.get("provider"), config.get("temperature"), json.dumps(config),
                )

            for e in edges:
                await conn.execute(
                    """
                    INSERT INTO workflow_edges (workflow_version_id, source_node_id, target_node_id, source_handle)
                    VALUES ($1, $2, $3, $4)
                    """,
                    version_id, e.get("source"), e.get("target"), e.get("sourceHandle"),
                )

    logger.info(
        "projects: created project_id=%s name=%r with %d nodes, %d edges",
        project_id, name, len(nodes), len(edges),
    )
    await cache.delete(_LIST_CACHE_KEY)
    return await get_project(project_id)
