"""
Search result parser: URL extraction and product-page filtering.

Filters out category pages, search pages, blogs, and other non-product URLs
from SERP results, keeping only actual individual product pages.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)

# URL path patterns that indicate a NON-product page
_EXCLUDE_PATTERNS: list[str] = [
    "/category/",
    "/categories/",
    "/search?",
    "/search/",
    "/blog/",
    "/article/",
    "/articles/",
    "/news/",
    "/collections/",
    "/collection/",
    "/tag/",
    "/tags/",
    "/help/",
    "/about/",
    "/contact/",
    "/faq/",
    "/reviews/",
    "/compare/",
    "/wishlist/",
    "/cart/",
    "/checkout/",
    "/account/",
    "/login/",
    "/register/",
    "/sitemap",
    "/privacy",
    "/terms",
    "/return",
    "/shipping",
    "/track",
]

# URL path patterns that strongly indicate a product page
_INCLUDE_PATTERNS: list[str] = [
    "/product/",
    "/products/",
    "/p/",
    "/dp/",
    "/ip/",
    "/buy/",
    "/item/",
    "/pd/",
    "/-p-",
    "/gp/product/",
    "/itm/",
]

# Tracking query params to strip
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "gclsrc", "fbclid", "ref", "ref_", "tag", "camp",
    "source", "medium", "campaign",
}


def normalize_url(url: str) -> str:
    """Strip tracking parameters and normalise a URL for deduplication."""
    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query, keep_blank_values=False)

        # Remove tracking params
        cleaned_params = {
            k: v
            for k, v in query_params.items()
            if k.lower() not in _TRACKING_PARAMS
        }

        cleaned_query = urlencode(cleaned_params, doseq=True) if cleaned_params else ""

        return urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.params,
            cleaned_query,
            "",  # drop fragment
        ))
    except Exception:
        return url.lower()


def is_product_url(url: str) -> bool:
    """Heuristic check: does *url* look like an individual product page?

    Returns True for likely product pages, False for category/blog/search pages.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        full_url = url.lower()
    except Exception:
        return False

    # --- Exclusion checks ---
    if path == "/search" or path.endswith("/search"):
        return False

    for pattern in _EXCLUDE_PATTERNS:
        if pattern in path or pattern in full_url:
            return False

    # Homepage or single-segment section page
    segments = [s for s in path.split("/") if s]
    if not segments:
        return False  # homepage
    if len(segments) == 1 and not re.search(r"[A-Z0-9]{5,}", segments[0], re.IGNORECASE):
        # Single path segment without a SKU-like ID → likely a section page
        return False

    # --- Inclusion checks ---
    for pattern in _INCLUDE_PATTERNS:
        if pattern in path:
            return True

    # Amazon ASIN pattern: /dp/B0... or /gp/product/B0...
    if re.search(r"/dp/[A-Z0-9]{10}", full_url, re.IGNORECASE):
        return True

    # Flipkart pattern: /p/itm...
    if re.search(r"/p/itm[a-z0-9]+", full_url, re.IGNORECASE):
        return True

    # Product slug heuristic: last path segment with hyphens or long alphanumeric
    if len(segments) >= 2:
        last_segment = segments[-1]
        # Contains hyphens (typical product slug)
        if "-" in last_segment and len(last_segment) > 10:
            return True
        # Looks like a SKU / product ID
        if re.search(r"[A-Z0-9]{5,}", last_segment, re.IGNORECASE):
            return True

    # No strong signal either way — include it (let the scraper decide)
    return True


def extract_product_urls(
    search_results: list[dict],
    limit: int = 5,
) -> list[dict]:
    """Filter and deduplicate search results, keeping only product-page URLs.

    Args:
        search_results: Raw result dicts from GoogleSearchClient.
        limit: Maximum number of product URLs to return (hard cap 5).

    Returns:
        A list of result dicts with confirmed product-page URLs.
    """
    limit = min(max(limit, 1), 5)
    seen: set[str] = set()
    filtered: list[dict] = []

    for result in search_results:
        url = result.get("url", "")
        if not url:
            continue

        norm = normalize_url(url)
        if norm in seen:
            continue

        if not is_product_url(url):
            logger.debug("Excluded non-product URL: %s", url)
            continue

        seen.add(norm)
        filtered.append(result)

        if len(filtered) >= limit:
            break

    logger.info("Filtered to %d product URLs from %d results", len(filtered), len(search_results))
    return filtered
