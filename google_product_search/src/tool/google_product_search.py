"""
Google Product Search Tool Interface.

This module provides the primary, single-function interface ('google_product_search')
for the AI agent to search and scrape products dynamically. Supports both single-query
and batch-query execution with bounded concurrent processing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from config import get_config
from models.product import Product, SearchResponse, BatchSearchResponse
from search.google_client import GoogleSearchClient
from search.result_parser import extract_product_urls
from scraper.product_scraper import ProductScraper

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "google_product_search",
        "description": (
            "Search Google dynamically for products based on one or more natural-language queries. "
            "Use this tool when the user needs dynamically discovered products from the web "
            "or when products may not exist in the internal product knowledge base. "
            "The tool searches Google, visits individual product pages, scrapes product information, "
            "and returns structured JSON with up to 5 products per query. "
            "You may submit multiple queries in a single call to search for different products concurrently. "
            "Provide the complete natural-language product requirement in each query. "
            "Do not provide URLs, CSS selectors, or search-engine parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single natural-language product search query. Use 'queries' for multiple queries."
                },
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of natural-language product search queries to execute concurrently. Each query independently returns up to 'limit' products."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of products to return per query.",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 5
                }
            },
            "required": []
        }
    }
}


def get_tool_schema() -> dict[str, Any]:
    """Return the OpenAI-compatible tool definition."""
    return TOOL_SCHEMA


async def _process_single_query(
    query: str,
    limit: int,
    scraper: ProductScraper,
    browser: Any,
    query_semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Process a single query within the batch pipeline.

    Bounded by query_semaphore to limit concurrent query execution.
    Uses a shared browser instance for Playwright scraping.
    """
    async with query_semaphore:
        query_start = time.time()
        logger.info("Query started: '%s'", query)

        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty query received")
            return SearchResponse(
                query=query,
                count=0,
                products=[],
                error="Query must be a non-empty string."
            ).model_dump()

        limit = min(max(limit, 1), 5)

        try:
            # 1. Google Search (offload synchronous requests.get to thread pool)
            search_start = time.time()
            client = GoogleSearchClient()
            search_results = await asyncio.to_thread(
                client.search_products, query, limit
            )
            search_duration = time.time() - search_start
            logger.info(
                "Google search completed for '%s': %d results in %.2fs",
                query, len(search_results), search_duration
            )

            # 2. Product URL extraction & filtering
            product_urls_with_meta = extract_product_urls(search_results, limit=limit)
            logger.info(
                "Query '%s': %d search results → %d product URLs",
                query, len(search_results), len(product_urls_with_meta)
            )

            if not product_urls_with_meta:
                logger.info("No product-page URLs found for query: %s", query)
                return SearchResponse(
                    query=query,
                    count=0,
                    products=[],
                    error=None
                ).model_dump()

            # 3. Scrape details using shared browser
            scrape_start = time.time()
            raw_scraped_products = await scraper.scrape_products(
                product_urls_with_meta, browser=browser
            )
            scrape_duration = time.time() - scrape_start
            logger.info(
                "Query '%s': scraped %d products in %.2fs",
                query, len(raw_scraped_products), scrape_duration
            )

            # 4. Normalize into Product models
            products: list[Product] = []
            for raw in raw_scraped_products:
                url = raw.get("product_url", "")
                serp_meta = raw.get("serp_meta", {})
                normalized = Product.from_raw(raw, url, serp_meta)
                if normalized:
                    products.append(normalized)
                else:
                    logger.warning("Failed to normalize product from URL: %s", url)

            query_duration = time.time() - query_start
            logger.info(
                "Query '%s' completed: %d products in %.2fs",
                query, len(products), query_duration
            )

            return SearchResponse(
                query=query,
                count=len(products),
                products=products,
                error=None
            ).model_dump()

        except Exception as exc:
            query_duration = time.time() - query_start
            logger.exception(
                "Query '%s' failed after %.2fs: %s", query, query_duration, exc
            )
            return SearchResponse(
                query=query,
                count=0,
                products=[],
                error=f"Query failure: {exc}"
            ).model_dump()


async def async_google_batch_search(
    queries: list[str], limit: int = 5
) -> dict[str, Any]:
    """Execute multiple product search queries concurrently.

    Uses a shared Playwright browser instance and bounded concurrency at
    both the query level and the product-scraping level.

    Args:
        queries: List of natural-language product search queries.
        limit: Maximum number of products per query (1-5).

    Returns:
        BatchSearchResponse dict with results for each query.
    """
    from playwright.async_api import async_playwright

    config = get_config()
    batch_start = time.time()
    logger.info("Batch search started: %d queries", len(queries))

    query_semaphore = asyncio.Semaphore(config.MAX_QUERY_CONCURRENCY)
    scraper = ProductScraper(
        headless=config.PLAYWRIGHT_HEADLESS,
        timeout_ms=config.PLAYWRIGHT_TIMEOUT_MS,
        max_concurrent=config.MAX_PRODUCT_CONCURRENCY,
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=config.PLAYWRIGHT_HEADLESS)
        try:
            tasks = [
                _process_single_query(q, limit, scraper, browser, query_semaphore)
                for q in queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await browser.close()

    # Convert exceptions to error responses
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Batch query %d failed: %s", i, result)
            final_results.append(SearchResponse(
                query=queries[i],
                count=0,
                products=[],
                error=f"Batch processing error: {result}"
            ).model_dump())
        else:
            final_results.append(result)

    batch_duration = time.time() - batch_start
    logger.info("Batch completed: %d queries in %.2fs", len(queries), batch_duration)

    return BatchSearchResponse(results=[
        SearchResponse(**r) for r in final_results
    ]).model_dump()


def google_product_search(
    query: str | None = None,
    queries: list[str] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search Google for products, scrape individual product pages, and normalize them.

    Supports both single-query and batch-query execution.

    Args:
        query: Single natural-language query (backward compatible).
        queries: List of queries for batch execution.
        limit: Maximum number of products to return per query (1-5).

    Returns:
        For single query: SearchResponse dict.
        For batch queries: BatchSearchResponse dict with 'results' list.
    """
    config = get_config()
    is_batch = (queries is not None)

    # Normalize inputs
    if queries and query:
        # Both provided — merge them
        all_queries = [query] + queries
    elif queries:
        all_queries = queries
    elif query:
        all_queries = [query]
    else:
        return SearchResponse(
            query="",
            count=0,
            products=[],
            error="Either 'query' or 'queries' must be provided."
        ).model_dump()

    # Filter out empty strings
    all_queries = [q.strip() for q in all_queries if q and q.strip()]
    if not all_queries:
        return SearchResponse(
            query="",
            count=0,
            products=[],
            error="All queries were empty."
        ).model_dump()

    # Validate batch size
    if len(all_queries) > config.MAX_BATCH_QUERIES:
        return SearchResponse(
            query="",
            count=0,
            products=[],
            error=f"Batch size {len(all_queries)} exceeds maximum of {config.MAX_BATCH_QUERIES}."
        ).model_dump()

    limit = min(max(limit, 1), 5)

    # Single query — backward compatible path
    if not is_batch:
        logger.info(
            "Starting google_product_search with query='%s', limit=%d",
            all_queries[0], limit
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    async_google_batch_search(all_queries, limit)
                )
                batch_result = future.result()
        else:
            batch_result = asyncio.run(
                async_google_batch_search(all_queries, limit)
            )

        # Unwrap single result from batch response
        if batch_result.get("results"):
            return batch_result["results"][0]
        return SearchResponse(
            query=all_queries[0],
            count=0,
            products=[],
            error="No results returned."
        ).model_dump()

    # Batch query path
    logger.info(
        "Starting batch google_product_search with %d queries, limit=%d",
        len(all_queries), limit
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                async_google_batch_search(all_queries, limit)
            )
            return future.result()
    else:
        return asyncio.run(
            async_google_batch_search(all_queries, limit)
        )
