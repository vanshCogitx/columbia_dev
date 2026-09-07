"""Exposes the vendored Google Product Search tool (see google_product_search/
and google_product_search_mount.py) as real routes on this app's own router,
so they appear on the main "Columbia Inventory Agent Toolset & RAG API"
Swagger page — same endpoints, same request/response shapes, same behavior
as the tool's own standalone src/main.py, just registered here via
include_router() instead of mounted as a separate sub-app (which would only
ever show up on its own, separate /docs page — see
google_product_search_mount.py's docstring for why the tool's own code
can't just be imported directly here without the isolation it provides).

The two request models below are a deliberate exact duplicate of the ones
defined inline in google_product_search/src/main.py — necessary because
FastAPI needs them defined as real classes in this module for the OpenAPI
schema to generate correctly here, but the fields/validation are copied
verbatim so the request contract is identical to the original.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from google_product_search_mount import load_google_product_search_tool

google_product_search, get_tool_schema, SearchResponse, BatchSearchResponse = load_google_product_search_tool()

router = APIRouter(tags=["google product search"])


class GoogleProductSearchRequest(BaseModel):
    """Payload for POST /tools/google-product-search/search (single query,
    matches the tool's own legacy /search endpoint)."""
    query: str = Field(
        ...,
        description="Natural-language product search query containing requirements like brand, product type, etc."
    )
    limit: int = Field(5, ge=1, le=5, description="Maximum number of products to return.")


class GoogleProductBatchSearchRequest(BaseModel):
    """Payload for POST /tools/google-product-search/api/search. Accepts
    either a single 'text' query or a list of 'queries'; if both are
    provided, 'text' is prepended to 'queries' (matches the tool's own
    unified /api/search endpoint)."""
    text: str | None = Field(None, description="A single natural-language product search query.")
    queries: list[str] | None = Field(
        None, description="A list of natural-language product search queries to execute concurrently."
    )
    limit: int = Field(5, ge=1, le=5, description="Maximum number of products to return per query.")

    @model_validator(mode="after")
    def validate_has_query(self):
        if not self.text and not self.queries:
            raise ValueError("Either 'text' or 'queries' must be provided.")
        return self


@router.post(
    "/tools/google-product-search/search",
    response_model=SearchResponse,
    summary="Search Google for products (single query)",
    description="Searches Google, scrapes matching product pages, and returns structured product data.",
)
async def google_product_search_endpoint(request: GoogleProductSearchRequest):
    try:
        return google_product_search(query=request.query, limit=request.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/tools/google-product-search/api/search",
    summary="Search Google for products (single or batch)",
    description=(
        "Unified search endpoint. Single query: {\"text\": \"...\"}. "
        "Batch: {\"queries\": [\"...\", \"...\"]}. Returns SearchResponse for a "
        "single query, BatchSearchResponse for a batch."
    ),
)
async def google_product_search_batch_endpoint(request: GoogleProductBatchSearchRequest):
    try:
        return google_product_search(query=request.text, queries=request.queries, limit=request.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/tools/google-product-search/tool-schema",
    summary="OpenAI function-calling schema for the google_product_search tool",
)
async def google_product_search_tool_schema_endpoint():
    return get_tool_schema()


@router.get("/tools/google-product-search/health", summary="Health check")
async def google_product_search_health_endpoint():
    return {"status": "healthy", "version": "2.0.0"}
