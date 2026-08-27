import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import asyncpg
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

import cache
import db
import judge
import simulation
import projects as projects_module
from models import (
    ScenarioCreateRequest, ScenarioResponse, ScenarioListResponse,
    RunTriggerResponse, RunListResponse, RunListItem,
    RunDetailResponse, SimulationTurnItem,
    TurnFeedbackRequest, TurnFeedbackResponse, TurnJudgmentDetailResponse,
    WorkflowGraphResponse, WorkflowGraphNode, WorkflowGraphEdge,
)

logger = logging.getLogger("columbia_backend")

router = APIRouter()

# See cache.py's module docstring / routers/evals.py's _EVALS_CACHE_TTL for
# the same short-TTL-no-explicit-invalidation tradeoff. workflow-graph gets a
# longer TTL since it only changes on a re-ingest (rare, deliberate).
_TESTING_CACHE_TTL = 5
_WORKFLOW_GRAPH_CACHE_TTL = 60

# Every endpoint here is scoped to a project via a
# /api/projects/{project_id}/testing/... path prefix and
# Depends(projects_module.get_project_or_404), same convention as
# routers/evals.py. Testing is the one place per-project Cartesian
# credentials are actually used (see simulation.py) — simulation_scenarios
# is a "root" table (see db.py), so scoping every other query here is a
# single-hop join back to s.project_id.


def _to_iso(ts: datetime) -> str:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts.isoformat()


def _as_list(value) -> list:
    return json.loads(value) if isinstance(value, str) else (value or [])


def _as_dict(value) -> Optional[dict]:
    if value is None:
        return None
    return json.loads(value) if isinstance(value, str) else value


# --- Scenarios ---

@router.post(
    "/api/projects/{project_id}/testing/scenarios",
    response_model=ScenarioResponse,
    tags=["testing"],
    summary="Create a test scenario",
)
async def create_scenario(
    request: ScenarioCreateRequest = Body(...),
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        row = await db.db_pool.fetchrow(
            """
            INSERT INTO simulation_scenarios (name, user_scenario, success_criteria, max_turns, mock_tools, created_by_user_id, project_id)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7)
            RETURNING id, name, user_scenario, success_criteria, max_turns, mock_tools, created_at
            """,
            request.name, request.user_scenario, json.dumps(request.success_criteria),
            request.max_turns, request.mock_tools, current_user["id"], project["id"],
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    return ScenarioResponse(
        id=row["id"], name=row["name"], user_scenario=row["user_scenario"],
        success_criteria=_as_list(row["success_criteria"]), max_turns=row["max_turns"],
        mock_tools=row["mock_tools"], created_at=_to_iso(row["created_at"]),
    )


_SCENARIO_LIST_QUERY = """
SELECT s.id, s.name, s.user_scenario, s.success_criteria, s.max_turns, s.mock_tools, s.created_at,
       lr.status AS last_run_status, lr.verdict AS last_run_verdict
FROM simulation_scenarios s
LEFT JOIN LATERAL (
    SELECT status, verdict FROM simulation_runs WHERE scenario_id = s.id ORDER BY started_at DESC LIMIT 1
) lr ON true
WHERE s.project_id = $1
"""


@router.get(
    "/api/projects/{project_id}/testing/scenarios",
    response_model=ScenarioListResponse,
    tags=["testing"],
    summary="List test scenarios, with each one's most recent run status",
)
async def list_scenarios(
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    # Short TTL, no explicit invalidation on create/delete/trigger-run — same
    # tradeoff as evals feed/stats (see cache.py docstring): this is already
    # polled every few seconds, so a few seconds of staleness after a
    # mutation is the existing UX, not a new regression.
    cache_key = f"evalcache:scenarios:{project['id']}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ScenarioListResponse(**cached)

    try:
        rows = await db.db_pool.fetch(_SCENARIO_LIST_QUERY + " ORDER BY s.created_at DESC", project["id"])
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    result = ScenarioListResponse(items=[
        ScenarioResponse(
            id=r["id"], name=r["name"], user_scenario=r["user_scenario"],
            success_criteria=_as_list(r["success_criteria"]), max_turns=r["max_turns"],
            mock_tools=r["mock_tools"], created_at=_to_iso(r["created_at"]),
            last_run_status=r["last_run_status"], last_run_verdict=r["last_run_verdict"],
        ) for r in rows
    ])
    await cache.set_json(cache_key, result.model_dump(), _TESTING_CACHE_TTL)
    return result


@router.get(
    "/api/projects/{project_id}/testing/scenarios/{scenario_id}",
    response_model=ScenarioResponse,
    tags=["testing"],
    summary="Get a single scenario",
)
async def get_scenario(
    scenario_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        row = await db.db_pool.fetchrow(_SCENARIO_LIST_QUERY + " AND s.id = $2", project["id"], scenario_id)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
    if not row:
        raise HTTPException(status_code=404, detail="Scenario not found.")

    return ScenarioResponse(
        id=row["id"], name=row["name"], user_scenario=row["user_scenario"],
        success_criteria=_as_list(row["success_criteria"]), max_turns=row["max_turns"],
        mock_tools=row["mock_tools"], created_at=_to_iso(row["created_at"]),
        last_run_status=row["last_run_status"], last_run_verdict=row["last_run_verdict"],
    )


@router.delete(
    "/api/projects/{project_id}/testing/scenarios/{scenario_id}",
    tags=["testing"],
    summary="Delete a scenario (cascades its runs and turns)",
)
async def delete_scenario(
    scenario_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        deleted = await db.db_pool.fetchval(
            "DELETE FROM simulation_scenarios WHERE id = $1 AND project_id = $2 RETURNING id",
            scenario_id, project["id"],
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
    if not deleted:
        raise HTTPException(status_code=404, detail="Scenario not found.")
    return {"success": True, "id": scenario_id}


# --- Runs ---

@router.post(
    "/api/projects/{project_id}/testing/scenarios/{scenario_id}/run",
    response_model=RunTriggerResponse,
    tags=["testing"],
    summary="Trigger a scenario run",
    description=(
        "Starts a simulated multi-turn conversation against the real Cartesian workflow "
        "in the background and returns immediately — poll GET .../testing/runs/{run_id} "
        "or open its /stream to watch progress. Runs can take several minutes."
    ),
)
async def trigger_run(
    scenario_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    scenario = await db.db_pool.fetchrow(
        "SELECT id FROM simulation_scenarios WHERE id = $1 AND project_id = $2", scenario_id, project["id"]
    )
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found.")

    session_id = f"sim_{uuid.uuid4()}"
    row = await db.db_pool.fetchrow(
        "INSERT INTO simulation_runs (scenario_id, session_id, status) VALUES ($1, $2, 'running') RETURNING id, status",
        scenario_id, session_id,
    )
    run_id = row["id"]
    logger.info("testing: triggered run_id=%s for scenario_id=%s project_id=%s", run_id, scenario_id, project["id"])

    # Fire-and-forget background task — same convention as
    # judge.run_judge_pipeline / cartesian's chat generation, not FastAPI
    # BackgroundTasks, not cancelled on client disconnect.
    asyncio.create_task(simulation.run_simulation(run_id))

    return RunTriggerResponse(run_id=run_id, status=row["status"])


@router.get(
    "/api/projects/{project_id}/testing/scenarios/{scenario_id}/runs",
    response_model=RunListResponse,
    tags=["testing"],
    summary="Run history for a scenario",
)
async def list_scenario_runs(
    scenario_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        rows = await db.db_pool.fetch(
            """
            SELECT r.id AS run_id, r.scenario_id, s.name AS scenario_name, r.status, r.turn_count,
                   r.verdict, r.started_at, r.completed_at
            FROM simulation_runs r
            JOIN simulation_scenarios s ON s.id = r.scenario_id
            WHERE r.scenario_id = $1 AND s.project_id = $2
            ORDER BY r.started_at DESC
            """,
            scenario_id, project["id"],
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    return RunListResponse(items=[_row_to_run_list_item(r) for r in rows])


@router.get(
    "/api/projects/{project_id}/testing/runs",
    response_model=RunListResponse,
    tags=["testing"],
    summary="Most recent runs across all scenarios in this project",
)
async def list_runs(
    limit: int = Query(default=50, le=200),
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        rows = await db.db_pool.fetch(
            """
            SELECT r.id AS run_id, r.scenario_id, s.name AS scenario_name, r.status, r.turn_count,
                   r.verdict, r.started_at, r.completed_at
            FROM simulation_runs r
            JOIN simulation_scenarios s ON s.id = r.scenario_id
            WHERE s.project_id = $1
            ORDER BY r.started_at DESC LIMIT $2
            """,
            project["id"], limit,
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    return RunListResponse(items=[_row_to_run_list_item(r) for r in rows])


def _row_to_run_list_item(r) -> RunListItem:
    return RunListItem(
        run_id=r["run_id"], scenario_id=r["scenario_id"], scenario_name=r["scenario_name"],
        status=r["status"], turn_count=r["turn_count"], verdict=r["verdict"],
        started_at=_to_iso(r["started_at"]), completed_at=_to_iso(r["completed_at"]) if r["completed_at"] else None,
    )


_TURNS_QUERY = """
SELECT t.id, t.turn_index, t.simulated_user_text, t.ai_response_text, t.raw_output,
       tf.id AS feedback_id, tf.rating AS feedback_rating,
       tj.id AS judgment_id, tj.corrected_score, tj.root_cause_node
FROM simulation_turns t
LEFT JOIN LATERAL (
    SELECT id, rating FROM simulation_turn_feedback WHERE turn_id = t.id ORDER BY id DESC LIMIT 1
) tf ON true
LEFT JOIN LATERAL (
    SELECT id, corrected_score, root_cause_node FROM simulation_turn_judgments
    WHERE turn_feedback_id = tf.id ORDER BY id DESC LIMIT 1
) tj ON true
WHERE t.run_id = $1
ORDER BY t.turn_index
"""


@router.get(
    "/api/projects/{project_id}/testing/runs/{run_id}",
    response_model=RunDetailResponse,
    tags=["testing"],
    summary="Run detail: scenario recap, every turn (with its raw trace, feedback, and judgment), and the verdict",
)
async def get_run_detail(
    run_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    try:
        run = await db.db_pool.fetchrow(
            """
            SELECT r.id AS run_id, r.scenario_id, s.name AS scenario_name, s.user_scenario,
                   s.success_criteria, s.max_turns, r.status, r.turn_count, r.verdict,
                   r.verdict_reasoning, r.verdict_thinking, r.error, r.started_at, r.completed_at
            FROM simulation_runs r
            JOIN simulation_scenarios s ON s.id = r.scenario_id
            WHERE r.id = $1 AND s.project_id = $2
            """,
            run_id, project["id"],
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        turn_rows = await db.db_pool.fetch(_TURNS_QUERY, run_id)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    return RunDetailResponse(
        run_id=run["run_id"], scenario_id=run["scenario_id"], scenario_name=run["scenario_name"],
        user_scenario=run["user_scenario"], success_criteria=_as_list(run["success_criteria"]),
        max_turns=run["max_turns"], status=run["status"], turn_count=run["turn_count"],
        verdict=run["verdict"], verdict_reasoning=run["verdict_reasoning"], verdict_thinking=run["verdict_thinking"],
        error=run["error"], started_at=_to_iso(run["started_at"]),
        completed_at=_to_iso(run["completed_at"]) if run["completed_at"] else None,
        turns=[
            SimulationTurnItem(
                id=t["id"], turn_index=t["turn_index"], simulated_user_text=t["simulated_user_text"],
                ai_response_text=t["ai_response_text"], raw_output=_as_dict(t["raw_output"]),
                feedback_id=t["feedback_id"], feedback_rating=t["feedback_rating"],
                judged=t["judgment_id"] is not None, corrected_score=t["corrected_score"],
                root_cause_node=t["root_cause_node"],
            ) for t in turn_rows
        ],
    )


STREAM_READ_BLOCK_MS = 5000
STREAM_MAX_IDLE_BLOCKS = 24  # ~2 minutes of no new events before giving up on the live view — the background run itself is NOT cancelled by this


async def _stream_run_events(project_id: int, run_id: int):
    run = await db.db_pool.fetchrow(
        """
        SELECT r.status, r.verdict, r.verdict_reasoning, r.error
        FROM simulation_runs r JOIN simulation_scenarios s ON s.id = r.scenario_id
        WHERE r.id = $1 AND s.project_id = $2
        """,
        run_id, project_id,
    )
    if not run:
        yield f"data: {json.dumps({'type': 'run_error', 'detail': 'Run not found for this project.'})}\n\n"
        return
    if run["status"] in ("completed", "failed"):
        if run["status"] == "failed":
            yield f"data: {json.dumps({'type': 'run_error', 'detail': run['error']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'run_done', 'verdict': run['verdict'], 'verdict_reasoning': run['verdict_reasoning']})}\n\n"
        return

    key = judge.make_stream_key(judge.STREAM_PREFIX_SIM_RUN, run_id)
    last_id = "0"
    idle_blocks = 0
    while True:
        result = await db.redis_client.xread({key: last_id}, block=STREAM_READ_BLOCK_MS, count=50)
        if not result:
            idle_blocks += 1
            if idle_blocks > STREAM_MAX_IDLE_BLOCKS:
                yield f"data: {json.dumps({'type': 'run_error', 'detail': 'Timed out waiting for the run — it may still finish in the background. Reopen this run later to check.'})}\n\n"
                return
            continue
        idle_blocks = 0
        for _key, entries in result:
            for entry_id, fields in entries:
                last_id = entry_id
                data = fields.get("data", "{}")
                yield f"data: {data}\n\n"
                try:
                    parsed = json.loads(data)
                    if parsed.get("type") in ("run_done", "run_error"):
                        return
                except json.JSONDecodeError:
                    pass


@router.get(
    "/api/projects/{project_id}/testing/runs/{run_id}/stream",
    tags=["testing"],
    summary="Live SSE stream of a run's turn-by-turn progress and final verdict",
)
async def stream_run(
    project_id: int,
    run_id: int,
    current_user: dict = Depends(db.get_current_user_query),
    project: dict = Depends(projects_module.get_project_or_404),
):
    return StreamingResponse(
        _stream_run_events(project_id, run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Turn feedback (thumbs up/down on individual simulated responses) ---

@router.post(
    "/api/projects/{project_id}/testing/turns/{turn_id}/feedback",
    response_model=TurnFeedbackResponse,
    tags=["testing"],
    summary="Rate a simulated turn's AI response (same Layer 2 judge pipeline as real feedback)",
)
async def submit_turn_feedback(
    turn_id: int,
    request: TurnFeedbackRequest = Body(...),
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'.")

    turn = await db.db_pool.fetchrow(
        """
        SELECT t.id FROM simulation_turns t
        JOIN simulation_runs r ON r.id = t.run_id
        JOIN simulation_scenarios s ON s.id = r.scenario_id
        WHERE t.id = $1 AND s.project_id = $2
        """,
        turn_id, project["id"],
    )
    if not turn:
        raise HTTPException(status_code=404, detail="Turn not found.")

    row = await db.db_pool.fetchrow(
        "INSERT INTO simulation_turn_feedback (turn_id, rating, reason) VALUES ($1, $2, $3) RETURNING id",
        turn_id, request.rating, request.reason,
    )
    feedback_id = row["id"]
    logger.info("testing: turn_id=%s feedback_id=%s rating=%s", turn_id, feedback_id, request.rating)

    asyncio.create_task(judge.run_simulation_judge_pipeline(feedback_id))

    return TurnFeedbackResponse(id=feedback_id, turn_id=turn_id, rating=request.rating)


@router.get(
    "/api/projects/{project_id}/testing/turn-feedback/{turn_feedback_id}",
    response_model=TurnJudgmentDetailResponse,
    tags=["testing"],
    summary="Judge-only detail for one simulated-turn feedback event",
    description="Mirrors GET .../evals/feed/{feedback_id}'s judge fields, for the shared judge-verdict panel component.",
)
async def get_turn_feedback_detail(
    turn_feedback_id: int,
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    feedback = await db.db_pool.fetchrow(
        """
        SELECT tf.id FROM simulation_turn_feedback tf
        JOIN simulation_turns t ON t.id = tf.turn_id
        JOIN simulation_runs r ON r.id = t.run_id
        JOIN simulation_scenarios s ON s.id = r.scenario_id
        WHERE tf.id = $1 AND s.project_id = $2
        """,
        turn_feedback_id, project["id"],
    )
    if not feedback:
        raise HTTPException(status_code=404, detail="Turn feedback event not found.")

    row = await db.db_pool.fetchrow(
        """
        SELECT initial_score, corrected_score, judge_reasoning, root_cause_node,
               root_cause_snippet, root_cause_explanation, draft_thinking,
               critique_thinking, attribution_thinking, error
        FROM simulation_turn_judgments WHERE turn_feedback_id = $1
        """,
        turn_feedback_id,
    )
    if not row:
        return TurnJudgmentDetailResponse(turn_feedback_id=turn_feedback_id, judged=False)

    return TurnJudgmentDetailResponse(
        turn_feedback_id=turn_feedback_id, judged=True,
        initial_score=row["initial_score"], corrected_score=row["corrected_score"],
        judge_reasoning=row["judge_reasoning"], root_cause_node=row["root_cause_node"],
        root_cause_snippet=row["root_cause_snippet"], root_cause_explanation=row["root_cause_explanation"],
        draft_thinking=row["draft_thinking"], critique_thinking=row["critique_thinking"],
        attribution_thinking=row["attribution_thinking"], error=row["error"],
    )


async def _stream_turn_judge_events(project_id: int, turn_feedback_id: int):
    owned = await db.db_pool.fetchrow(
        """
        SELECT tf.id FROM simulation_turn_feedback tf
        JOIN simulation_turns t ON t.id = tf.turn_id
        JOIN simulation_runs r ON r.id = t.run_id
        JOIN simulation_scenarios s ON s.id = r.scenario_id
        WHERE tf.id = $1 AND s.project_id = $2
        """,
        turn_feedback_id, project_id,
    )
    if not owned:
        yield f"data: {json.dumps({'type': 'pipeline_error', 'detail': 'Turn feedback event not found for this project.'})}\n\n"
        return

    already = await db.db_pool.fetchrow(
        "SELECT initial_score, corrected_score, judge_reasoning, root_cause_node, root_cause_snippet, root_cause_explanation, error "
        "FROM simulation_turn_judgments WHERE turn_feedback_id = $1",
        turn_feedback_id,
    )
    if already:
        if already["error"]:
            yield f"data: {json.dumps({'type': 'pipeline_error', 'detail': already['error']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'pipeline_done', 'initial_score': already['initial_score'], 'corrected_score': already['corrected_score'], 'judge_reasoning': already['judge_reasoning'], 'root_cause_node': already['root_cause_node'], 'root_cause_snippet': already['root_cause_snippet'], 'root_cause_explanation': already['root_cause_explanation']})}\n\n"
        return

    key = judge.make_stream_key(judge.STREAM_PREFIX_SIMULATION_JUDGE, turn_feedback_id)
    last_id = "0"
    idle_blocks = 0
    while True:
        result = await db.redis_client.xread({key: last_id}, block=STREAM_READ_BLOCK_MS, count=50)
        if not result:
            idle_blocks += 1
            if idle_blocks > STREAM_MAX_IDLE_BLOCKS:
                yield f"data: {json.dumps({'type': 'pipeline_error', 'detail': 'Timed out waiting for the judge — it may still finish in the background. Reopen this turn later to check.'})}\n\n"
                return
            continue
        idle_blocks = 0
        for _key, entries in result:
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
    "/api/projects/{project_id}/testing/turn-feedback/{turn_feedback_id}/stream",
    tags=["testing"],
    summary="Live SSE stream of the judge's thinking for one simulated-turn feedback event",
)
async def stream_turn_judge(
    project_id: int,
    turn_feedback_id: int,
    current_user: dict = Depends(db.get_current_user_query),
    project: dict = Depends(projects_module.get_project_or_404),
):
    return StreamingResponse(
        _stream_turn_judge_events(project_id, turn_feedback_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Workflow graph (for the trace-tree UI) ---

@router.get(
    "/api/projects/{project_id}/testing/workflow-graph",
    response_model=WorkflowGraphResponse,
    tags=["testing"],
    summary="Active workflow's nodes and edges, for rendering a turn's execution as a tree",
)
async def get_workflow_graph(
    current_user: dict = Depends(db.get_current_user),
    project: dict = Depends(projects_module.get_project_or_404),
):
    # Static per active workflow version — only changes on re-ingest (a rare,
    # deliberate action) — so a longer TTL than the other evals-cache entries
    # is safe, matching the frontend's own 60s poll interval for this call.
    cache_key = f"evalcache:workflow-graph:{project['id']}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return WorkflowGraphResponse(**cached)

    try:
        nodes, edges = await asyncio.gather(
            db.db_pool.fetch(
                """
                SELECT wn.node_id, wn.node_type, wn.alias, wn.model, wn.provider
                FROM workflow_nodes wn
                JOIN workflow_versions wv ON wv.id = wn.workflow_version_id
                WHERE wv.is_active = true AND wv.project_id = $1
                """,
                project["id"],
            ),
            db.db_pool.fetch(
                """
                SELECT we.source_node_id, we.target_node_id, we.source_handle
                FROM workflow_edges we
                JOIN workflow_versions wv ON wv.id = we.workflow_version_id
                WHERE wv.is_active = true AND wv.project_id = $1
                """,
                project["id"],
            ),
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    result = WorkflowGraphResponse(
        nodes=[WorkflowGraphNode(node_id=n["node_id"], node_type=n["node_type"], alias=n["alias"], model=n["model"], provider=n["provider"]) for n in nodes],
        edges=[WorkflowGraphEdge(source_node_id=e["source_node_id"], target_node_id=e["target_node_id"], source_handle=e["source_handle"]) for e in edges],
    )
    await cache.set_json(cache_key, result.model_dump(), _WORKFLOW_GRAPH_CACHE_TTL)
    return result
