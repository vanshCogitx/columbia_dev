import asyncio
import logging
from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException

import db
from models import (
    SearchRequest, ProductFamilyResult, CatalogFacetsResponse,
    CatalogSearchRequestV2, CatalogSearchRequestV3, CatalogProductResultV2,
)

logger = logging.getLogger("columbia_backend")

router = APIRouter()


# --- API Tool Endpoints ---

@router.post(
    "/tools/search-products",
    response_model=List[ProductFamilyResult],
    dependencies=[Depends(db.verify_api_key)],
    tags=["search tools"],
    summary="Unified search tool (Semantic search, direct ID lookup, and structured filters)",
    description=(
        "The primary search tool for the agent. Performs direct product ID lookup if family_id is supplied; "
        "performs local SQL structured lookup if query is omitted; and performs hybrid vector search "
        "with local variant filter joins if query is provided."
    )
)
async def search_products(request: SearchRequest = Body(...)):
    logger.info("TOOL search-products: request=%s", request.model_dump(exclude_none=True))
    try:
        results = await db.engine.search(
            query=request.query,
            family_id=request.family_id,
            category_level_1=request.category_level_1,
            category_level_2=request.category_level_2,
            product_type=request.product_type,
            sport=request.sport,
            price_min=request.price_min,
            price_max=request.price_max,
            color=request.color,
            size=request.size,
            in_stock_only=request.in_stock_only,
            top_k=request.top_k
        )
        logger.info(
            "TOOL search-products: %d families returned: %s",
            len(results), [r["family_id"] for r in results],
        )
        return results
    except Exception as e:
        logger.error("TOOL search-products: failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/tools/search-products-v2",
    response_model=List[CatalogProductResultV2],
    dependencies=[Depends(db.verify_api_key)],
    tags=["search tools"],
    summary="Flat catalog semantic search (V2)",
    description=(
        "Semantic search tool for the flat product catalog (product_catalog_2_3_unique dataset). "
        "Results are retrieved entirely from Pinecone — no database joins. "
        "A natural language 'query' is always required. Optional structured filters "
        "(category, sport, color, size, price range) are applied as Pinecone metadata pre-filters "
        "with progressive relaxation to maximise result coverage. "
        "Uses the same Google Gemma embedding model as the primary search tool."
    )
)
async def search_products_v2(request: CatalogSearchRequestV2 = Body(...)):
    logger.info("TOOL search-products-v2: request=%s", request.model_dump(exclude_none=True))
    try:
        results = await db.catalog_engine_v2.search(
            query=request.query,
            category_level_1=request.category_level_1,
            category_level_2=request.category_level_2,
            product_type=request.product_type,
            sport=request.sport,
            price_min=request.price_min,
            price_max=request.price_max,
            color=request.color,
            size=request.size,
            top_k=request.top_k,
        )
        logger.info(
            "TOOL search-products-v2: %d products returned: %s",
            len(results), [r["product_id"] for r in results],
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("TOOL search-products-v2: failed: %s", e)
        raise HTTPException(status_code=500, detail=f"V2 search failed: {str(e)}")


@router.post(
    "/tools/search-products-v3",
    response_model=List[List[CatalogProductResultV2]],
    dependencies=[Depends(db.verify_api_key)],
    tags=["search tools"],
    summary="Batched flat catalog semantic search (V3)",
    description=(
        "Batched version of search-products-v2 — accepts multiple independent queries in a "
        "single call and returns one result list per query, in the same order they were sent. "
        "Each query supports the same fields as V2 (a natural language 'query' plus optional "
        "structured filters). Queries run concurrently; if one query fails, only that query's "
        "slot comes back as an empty list — the rest of the batch still returns normally."
    )
)
async def search_products_v3(request: CatalogSearchRequestV3 = Body(...)):
    logger.info("TOOL search-products-v3: %d queries", len(request.queries))

    async def _run_one(q: CatalogSearchRequestV2, index: int) -> list:
        try:
            results = await db.catalog_engine_v2.search(
                query=q.query,
                category_level_1=q.category_level_1,
                category_level_2=q.category_level_2,
                product_type=q.product_type,
                sport=q.sport,
                price_min=q.price_min,
                price_max=q.price_max,
                color=q.color,
                size=q.size,
                top_k=q.top_k,
            )
            logger.info(
                "TOOL search-products-v3: query[%d] %d products returned: %s",
                index, len(results), [r["product_id"] for r in results],
            )
            return results
        except Exception as e:
            logger.error("TOOL search-products-v3: query[%d] failed: %s", index, e)
            return []

    all_results = await asyncio.gather(*(_run_one(q, i) for i, q in enumerate(request.queries)))
    return list(all_results)


@router.get(
    "/tools/catalog-facets",
    response_model=CatalogFacetsResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["search tools"],
    summary="Get valid values for catalog filters",
    description=(
        "Returns the exact unique values and categories present in the database, along with the "
        "minimum and maximum prices. Agents should call this tool first to understand the valid values "
        "they can use as structured filters (e.g. valid sports, categories, product types) to avoid empty results."
    )
)
async def get_catalog_facets():
    logger.info("TOOL catalog-facets: request received")
    try:
        facets = await db.engine.get_facets()
        logger.info("TOOL catalog-facets: response=%s", facets)
        return facets
    except Exception as e:
        logger.error("TOOL catalog-facets: failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve facets: {str(e)}")
