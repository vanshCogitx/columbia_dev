import json
import logging

import asyncpg
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

import db
import projects as projects_module
from models import ProjectResponse, ProjectListResponse

logger = logging.getLogger("columbia_backend")

router = APIRouter()


def _to_response(p: dict) -> ProjectResponse:
    # p["created_at"] is already an ISO string — projects.py's _row_to_dict
    # converts it there (once, at the cache-population boundary) since a
    # cached round-trip through Redis/JSON can't carry a real datetime.
    return ProjectResponse(
        id=p["id"], name=p["name"], cartesian_export_id=p["cartesian_export_id"],
        cartesian_base_url=p["cartesian_base_url"], is_live=p["is_live"], created_at=p["created_at"],
        node_count=p["node_count"], edge_count=p["edge_count"],
    )


@router.get(
    "/api/projects",
    response_model=ProjectListResponse,
    tags=["projects"],
    summary="List projects",
)
async def list_projects(current_user: dict = Depends(db.get_current_user)):
    items = await projects_module.list_projects()
    return ProjectListResponse(items=[_to_response(p) for p in items])


@router.get(
    "/api/projects/{project_id}",
    response_model=ProjectResponse,
    tags=["projects"],
    summary="Get a single project",
)
async def get_project(project_id: int, current_user: dict = Depends(db.get_current_user)):
    p = await projects_module.get_project(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found.")
    return _to_response(p)


@router.post(
    "/api/projects",
    response_model=ProjectResponse,
    tags=["projects"],
    summary="Create a project (ingests its workflow JSON in the same request)",
    description=(
        "Multipart form: name, cartesian_client_id, cartesian_client_secret, "
        "cartesian_export_id, cartesian_base_url (optional), workflow_json (file). "
        "Equivalent to running tools/ingest_workflow.py by hand, done via the API instead."
    ),
)
async def create_project(
    name: str = Form(...),
    cartesian_client_id: str = Form(...),
    cartesian_client_secret: str = Form(...),
    cartesian_export_id: str = Form(...),
    cartesian_base_url: str = Form(None),
    workflow_json: UploadFile = None,
    current_user: dict = Depends(db.get_current_user),
):
    if workflow_json is None:
        raise HTTPException(status_code=422, detail="workflow_json file is required.")
    try:
        raw = await workflow_json.read()
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="workflow_json must be valid JSON.")

    try:
        p = await projects_module.create_project(
            name=name,
            cartesian_client_id=cartesian_client_id,
            cartesian_client_secret=cartesian_client_secret,
            cartesian_export_id=cartesian_export_id,
            cartesian_base_url=cartesian_base_url or None,
            workflow_json=parsed,
            created_by_user_id=current_user["id"],
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    return _to_response(p)


@router.post(
    "/api/projects/{project_id}/set-live",
    response_model=ProjectResponse,
    tags=["projects"],
    summary="Mark a project as the live one",
    description=(
        "Flags this project as the one real Columbia chat traffic is attributed to (see "
        "projects.get_live_project_id / response_feedback.project_id). Unsets any other "
        "project's live flag in the same transaction — exactly one project is live at a time."
    ),
)
async def set_live_project(project_id: int, current_user: dict = Depends(db.get_current_user)):
    try:
        p = await projects_module.set_live_project(project_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Project not found.")
    return _to_response(p)
