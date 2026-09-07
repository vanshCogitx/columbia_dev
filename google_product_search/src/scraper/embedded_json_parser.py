"""
Embedded JSON product data extractor.

Finds product data embedded in JavaScript variables and inline JSON blobs
common in modern SPAs and server-rendered e-commerce sites.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_NEXT_DATA_RE = re.compile(
    r'<script\b[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE
)

# Regex patterns for common JS variable assignments containing product data
_JS_VAR_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "INITIAL_STATE",
        re.compile(
            r"window\.__INITIAL_STATE__\s*=\s*(\{.+?\})\s*;",
            re.DOTALL,
        ),
    ),
    (
        "PRELOADED_STATE",
        re.compile(
            r"window\.__PRELOADED_STATE__\s*=\s*(\{.+?\})\s*;",
            re.DOTALL,
        ),
    ),
    (
        "APP_STATE",
        re.compile(
            r"window\.__APP_STATE__\s*=\s*(\{.+?\})\s*;",
            re.DOTALL,
        ),
    ),
    (
        "NEXT_DATA_VAR",
        re.compile(
            r"window\.__NEXT_DATA__\s*=\s*(\{.+?\})\s*;",
            re.DOTALL,
        ),
    ),
    (
        "productData",
        re.compile(
            r"(?:var|let|const)\s+productData\s*=\s*(\{.+?\})\s*;",
            re.DOTALL,
        ),
    ),
    (
        "product_var",
        re.compile(
            r"(?:var|let|const)\s+product\s*=\s*(\{.+?\})\s*;",
            re.DOTALL,
        ),
    ),
]

# Product-identifying keys
_NAME_KEYS = {"name", "title", "productName", "product_name"}
_SUPPORTING_KEYS = {"price", "sku", "brand", "description", "mrp", "images", "variants"}


def extract_embedded_product(html_content: str) -> dict[str, Any] | None:
    """Extract product data from embedded JavaScript on the page.

    Args:
        html_content: Full HTML page source.

    Returns:
        A dict of product fields, or None if nothing found.
    """
    # --- Strategy 1: __NEXT_DATA__ script tag ---
    product = _try_next_data_tag(html_content)
    if product:
        return product

    # --- Strategy 2: JS variable assignments ---
    for label, pattern in _JS_VAR_PATTERNS:
        match = pattern.search(html_content)
        if not match:
            continue
        try:
            data = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            logger.debug("Malformed JSON in %s, skipping", label)
            continue

        product = find_product_data(data)
        if product:
            logger.debug("Found product data in %s", label)
            return product

    # --- Strategy 3: dataLayer ecommerce ---
    product = _try_datalayer(html_content)
    if product:
        return product

    # --- Strategy 4: Generic "product": {...} pattern ---
    product = _try_generic_product_json(html_content)
    if product:
        return product

    return None


def find_product_data(
    data: Any,
    depth: int = 0,
    max_depth: int = 8,
) -> dict[str, Any] | None:
    """Recursively search for a product-like dict in nested data.

    A dict is considered product-like if it has a name/title key AND at least
    one supporting field (price, sku, brand, description).

    Args:
        data: Any JSON-deserialized value.
        depth: Current recursion depth.
        max_depth: Maximum depth to traverse.

    Returns:
        The first product-like dict found, or None.
    """
    if depth > max_depth:
        return None

    if isinstance(data, dict):
        # Check if this dict looks like a product
        has_name = bool(_NAME_KEYS & set(data.keys()))
        has_support = bool(_SUPPORTING_KEYS & set(data.keys()))
        if has_name and has_support:
            return data

        # Recurse into values
        for key, value in data.items():
            result = find_product_data(value, depth + 1, max_depth)
            if result:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_product_data(item, depth + 1, max_depth)
            if result:
                return result

    return None


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _try_next_data_tag(html_content: str) -> dict[str, Any] | None:
    """Extract product from Next.js __NEXT_DATA__ script tag."""
    if not html_content:
        return None
    try:
        match = _NEXT_DATA_RE.search(html_content)
        if not match:
            return None

        text = match.group(1)
        if not text or not text.strip():
            return None

        data = json.loads(text)
        # Next.js stores page props in props.pageProps
        page_props = data.get("props", {}).get("pageProps", {})
        if page_props:
            product = find_product_data(page_props)
            if product:
                logger.debug("Found product in __NEXT_DATA__ pageProps")
                return product

        # Fallback: search entire structure
        return find_product_data(data)

    except Exception as exc:
        logger.debug("__NEXT_DATA__ extraction failed: %s", exc)
        return None


def _try_datalayer(html_content: str) -> dict[str, Any] | None:
    """Extract product from Google Tag Manager dataLayer."""
    try:
        # Find dataLayer.push calls
        pattern = re.compile(
            r"dataLayer\.push\((\{.+?\})\)",
            re.DOTALL,
        )
        for match in pattern.finditer(html_content):
            try:
                data = json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                continue

            # GA4 ecommerce structure
            ecommerce = data.get("ecommerce", {})
            if not ecommerce:
                continue

            # detail.products or items
            products = (
                ecommerce.get("detail", {}).get("products")
                or ecommerce.get("items")
                or ecommerce.get("products")
            )
            if isinstance(products, list) and products:
                product = products[0]
                if isinstance(product, dict):
                    logger.debug("Found product in dataLayer ecommerce")
                    return product

    except Exception as exc:
        logger.debug("dataLayer extraction failed: %s", exc)

    return None


def _try_generic_product_json(html_content: str) -> dict[str, Any] | None:
    """Last resort: find "product":{...} patterns in script blocks."""
    try:
        # Look for "product": { or 'product': { in script tags
        pattern = re.compile(
            r'"product"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
            re.DOTALL,
        )
        for match in pattern.finditer(html_content):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and (_NAME_KEYS & set(data.keys())):
                    logger.debug("Found product via generic JSON pattern")
                    return data
            except (json.JSONDecodeError, ValueError):
                continue

    except Exception as exc:
        logger.debug("Generic product JSON extraction failed: %s", exc)

    return None
