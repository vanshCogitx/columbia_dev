"""
Main entry point for Google Product Search tool.

Provides both a FastAPI HTTP server and a CLI for running queries directly.
Supports single-query and batch-query execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Ensure src directory is in Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import uvicorn

from models.product import SearchResponse, BatchSearchResponse
from tool.google_product_search import google_product_search, get_tool_schema

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("google_product_search")

app = FastAPI(
    title="Google Product Search Tool",
    description="AI Agent tool for dynamic product discovery and parsing via Google Search and Playwright",
    version="2.0.0",
)


# ── Legacy single-query endpoint ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Payload for POST /search endpoint."""
    query: str = Field(
        ...,
        description="Natural-language product search query containing requirements like brand, product type, etc."
    )
    limit: int = Field(
        5,
        ge=1,
        le=5,
        description="Maximum number of products to return."
    )


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """Expose the google_product_search tool over HTTP (single query, backward compatible)."""
    logger.info("Received POST /search request for query: '%s'", request.query)
    try:
        result = google_product_search(query=request.query, limit=request.limit)
        return result
    except Exception as exc:
        logger.exception("HTTP search endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Unified search endpoint (supports single + batch) ────────────────────────

class TextSearchRequest(BaseModel):
    """Payload for POST /api/search endpoint.
    
    Accepts either a single 'text' query or a list of 'queries'.
    If both are provided, 'text' is prepended to 'queries'.
    """
    text: str | None = Field(
        None,
        description="A single natural-language product search query."
    )
    queries: list[str] | None = Field(
        None,
        description="A list of natural-language product search queries to execute concurrently."
    )
    limit: int = Field(
        5,
        ge=1,
        le=5,
        description="Maximum number of products to return per query."
    )

    @model_validator(mode="after")
    def validate_has_query(self):
        if not self.text and not self.queries:
            raise ValueError("Either 'text' or 'queries' must be provided.")
        return self


@app.post("/api/search")
async def api_search_endpoint(request: TextSearchRequest):
    """Unified search endpoint accepting single or batch queries.
    
    Single query: {"text": "red fleece jacket", "limit": 5}
    Batch query:  {"queries": ["red fleece jacket", "hiking shoes"], "limit": 5}
    Mixed:        {"text": "pants", "queries": ["jacket", "shoes"], "limit": 3}
    
    Returns SearchResponse for single queries, BatchSearchResponse for batch.
    """
    logger.info(
        "Received POST /api/search request — text=%s, queries=%s",
        repr(request.text), repr(request.queries)
    )
    try:
        result = google_product_search(
            query=request.text,
            queries=request.queries,
            limit=request.limit,
        )
        return result
    except Exception as exc:
        logger.exception("HTTP /api/search endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tool-schema")
async def tool_schema_endpoint():
    """Return the OpenAI function calling schema for the tool."""
    return get_tool_schema()


@app.get("/health")
async def health_endpoint():
    """Simple health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="Google Product Search Tool: Dynamic Google Search & Product Page Scraper"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        nargs="+",
        help="One or more natural-language product queries to execute."
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=5,
        help="Max number of products to return per query (1-5, default 5)."
    )
    parser.add_argument(
        "-s", "--serve",
        action="store_true",
        help="Start the FastAPI HTTP server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default 0.0.0.0)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number (default 8000)."
    )

    args = parser.parse_args()

    if args.serve:
        logger.info("Starting FastAPI server on %s:%d", args.host, args.port)
        uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
    elif args.query:
        # Load dotenv if present
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        if len(args.query) == 1:
            logger.info("Running search for: '%s'", args.query[0])
            result = google_product_search(query=args.query[0], limit=args.limit)
        else:
            logger.info("Running batch search for %d queries", len(args.query))
            result = google_product_search(queries=args.query, limit=args.limit)

        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
