"""
HTML fallback parser for product page scraping.

Uses BeautifulSoup to extract product data from common HTML patterns,
OpenGraph meta tags, and semantic markup when structured data is unavailable.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Currency symbol → code mapping
_CURRENCY_MAP: dict[str, str] = {
    "₹": "INR",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
}


def extract_html_product(html_content: str, url: str) -> dict[str, Any] | None:
    """Extract product data from HTML using common patterns and meta tags.

    Args:
        html_content: Full HTML page source.
        url: The page URL (used for resolving relative image URLs).

    Returns:
        A dict of product fields, or None if not even a name could be found.
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception as exc:
            logger.warning("Failed to parse HTML: %s", exc)
            return None

    result: dict[str, Any] = {}

    result["name"] = _extract_name(soup)
    if not result["name"]:
        return None

    result["price"] = _extract_price(soup)
    result["currency"] = _extract_currency(soup, result.get("price"))
    result["original_price"] = _extract_original_price(soup)
    result["description"] = _extract_description(soup)
    result["brand"] = _extract_brand(soup)
    result["image"] = _extract_image(soup, url)
    result["rating"] = _extract_rating(soup)
    result["review_count"] = _extract_review_count(soup)
    result["availability"] = _extract_availability(soup)
    result["sizes"] = _extract_sizes(soup)
    result["color"] = _extract_color(soup)

    return result


# ------------------------------------------------------------------
# Individual field extractors
# ------------------------------------------------------------------

def _extract_name(soup: BeautifulSoup) -> str | None:
    """Extract product name from meta tags, microdata, or heading tags."""
    # OpenGraph title
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return _clean_text(og_title["content"])

    # Schema.org itemprop
    itemprop_name = soup.find(attrs={"itemprop": "name"})
    if itemprop_name:
        text = itemprop_name.get("content") or itemprop_name.get_text()
        if text and text.strip():
            return _clean_text(text)

    # Common CSS selectors for product titles
    for selector in [
        "h1.product-title",
        "h1.product-name",
        "h1.pdp-title",
        "h1.product__title",
        "h1#productTitle",
        "[data-testid='product-title']",
        ".product-header h1",
    ]:
        elem = soup.select_one(selector)
        if elem and elem.get_text(strip=True):
            return _clean_text(elem.get_text())

    # First h1
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return _clean_text(h1.get_text())

    # Title tag (cleaned)
    title = soup.find("title")
    if title and title.get_text(strip=True):
        text = title.get_text(strip=True)
        # Remove common suffixes like " | Store Name" or " - Brand"
        text = re.split(r"\s*[|–—-]\s*", text)[0].strip()
        return text if text else None

    return None


def _extract_price(soup: BeautifulSoup) -> str | None:
    """Extract current/sale price."""
    # Meta tags
    for prop in ("product:price:amount", "og:price:amount"):
        meta = soup.find("meta", attrs={"property": prop})
        if meta and meta.get("content"):
            return meta["content"]

    # Itemprop
    price_elem = soup.find(attrs={"itemprop": "price"})
    if price_elem:
        val = price_elem.get("content") or price_elem.get_text()
        if val and val.strip():
            return val.strip()

    # Common selectors
    for selector in [
        ".current-price",
        ".sale-price",
        ".offer-price",
        ".product-price",
        ".selling-price",
        "#priceblock_ourprice",
        "#priceblock_dealprice",
        ".a-price .a-offscreen",
        "[data-testid='product-price']",
        ".pdp-price",
        ".price-current",
    ]:
        elem = soup.select_one(selector)
        if elem and elem.get_text(strip=True):
            return elem.get_text(strip=True)

    # Generic .price class (be more careful to avoid picking up unrelated prices)
    price_container = soup.select_one(".price")
    if price_container:
        text = price_container.get_text(strip=True)
        if re.search(r"[\d₹$€£]", text):
            return text

    return None


def _extract_currency(soup: BeautifulSoup, price_text: Any = None) -> str | None:
    """Extract currency code."""
    # Meta tags
    for prop in ("product:price:currency", "og:price:currency"):
        meta = soup.find("meta", attrs={"property": prop})
        if meta and meta.get("content"):
            return meta["content"].upper()

    # Itemprop
    curr = soup.find(attrs={"itemprop": "priceCurrency"})
    if curr:
        val = curr.get("content") or curr.get_text()
        if val and val.strip():
            return val.strip().upper()

    # Detect from price string
    if price_text:
        text = str(price_text)
        for sym, code in _CURRENCY_MAP.items():
            if sym in text:
                return code
        if "Rs" in text:
            return "INR"

    return None


def _extract_original_price(soup: BeautifulSoup) -> str | None:
    """Extract original/MRP price."""
    for selector in [
        ".original-price",
        ".was-price",
        ".list-price",
        ".mrp",
        ".price-was",
        ".price-regular",
        ".price-compare",
        ".old-price",
        "[data-testid='original-price']",
    ]:
        elem = soup.select_one(selector)
        if elem and elem.get_text(strip=True):
            return elem.get_text(strip=True)

    # Look for <s> or <del> inside price containers
    for container_sel in (".price-container", ".price-block", ".price-box", ".product-price"):
        container = soup.select_one(container_sel)
        if container:
            for tag_name in ("s", "del", "strike"):
                struck = container.find(tag_name)
                if struck and struck.get_text(strip=True):
                    return struck.get_text(strip=True)

    return None


def _extract_description(soup: BeautifulSoup) -> str | None:
    """Extract product description."""
    # Meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        return meta_desc["content"].strip()

    # OG description
    og_desc = soup.find("meta", attrs={"property": "og:description"})
    if og_desc and og_desc.get("content"):
        return og_desc["content"].strip()

    # Itemprop
    desc_elem = soup.find(attrs={"itemprop": "description"})
    if desc_elem and desc_elem.get_text(strip=True):
        return desc_elem.get_text(strip=True)

    # Common selectors
    for selector in [
        ".product-description",
        ".pdp-description",
        "#productDescription",
        ".product-detail__description",
        "[data-testid='product-description']",
    ]:
        elem = soup.select_one(selector)
        if elem and elem.get_text(strip=True):
            return elem.get_text(strip=True)

    # First substantial paragraph
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text and len(text) > 50:
            return text

    return None


def _extract_brand(soup: BeautifulSoup) -> str | None:
    """Extract product brand."""
    # Itemprop
    brand_elem = soup.find(attrs={"itemprop": "brand"})
    if brand_elem:
        if isinstance(brand_elem, Tag):
            val = brand_elem.get("content") or brand_elem.get_text()
            if val and val.strip():
                return val.strip()

    # Common selectors
    for selector in [
        ".brand",
        ".product-brand",
        ".brand-name",
        "a[href*='/brand/']",
        "[data-testid='product-brand']",
    ]:
        elem = soup.select_one(selector)
        if elem and elem.get_text(strip=True):
            return elem.get_text(strip=True)

    # Meta tag
    meta_brand = soup.find("meta", attrs={"property": "product:brand"})
    if meta_brand and meta_brand.get("content"):
        return meta_brand["content"].strip()

    return None


def _extract_image(soup: BeautifulSoup, base_url: str) -> str | None:
    """Extract main product image URL."""
    # OG image
    og_img = soup.find("meta", attrs={"property": "og:image"})
    if og_img and og_img.get("content"):
        return _make_absolute(og_img["content"], base_url)

    # Itemprop
    img_elem = soup.find(attrs={"itemprop": "image"})
    if img_elem:
        src = img_elem.get("src") or img_elem.get("content") or img_elem.get("href")
        if src:
            return _make_absolute(src, base_url)

    # Common selectors
    for selector in [
        ".product-image img",
        ".pdp-image img",
        "#main-image",
        ".product-gallery img",
        ".gallery-image img",
        "[data-testid='product-image'] img",
    ]:
        elem = soup.select_one(selector)
        if elem:
            src = elem.get("src") or elem.get("data-src")
            if src:
                return _make_absolute(src, base_url)

    return None


def _extract_rating(soup: BeautifulSoup) -> float | None:
    """Extract product rating."""
    # Itemprop
    rating_elem = soup.find(attrs={"itemprop": "ratingValue"})
    if rating_elem:
        val = rating_elem.get("content") or rating_elem.get_text()
        return _parse_float(val)

    # Common selectors
    for selector in [".rating-value", ".star-rating", "[data-rating]"]:
        elem = soup.select_one(selector)
        if elem:
            val = elem.get("data-rating") or elem.get("content") or elem.get_text()
            parsed = _parse_float(val)
            if parsed is not None:
                return parsed

    return None


def _extract_review_count(soup: BeautifulSoup) -> int | None:
    """Extract review/rating count."""
    # Itemprop
    rc_elem = soup.find(attrs={"itemprop": "reviewCount"})
    if rc_elem:
        val = rc_elem.get("content") or rc_elem.get_text()
        return _parse_int_from_text(val)

    rc_elem = soup.find(attrs={"itemprop": "ratingCount"})
    if rc_elem:
        val = rc_elem.get("content") or rc_elem.get_text()
        return _parse_int_from_text(val)

    # Common selectors
    for selector in [".review-count", ".ratings-count", ".rating-count"]:
        elem = soup.select_one(selector)
        if elem:
            return _parse_int_from_text(elem.get_text())

    return None


def _extract_availability(soup: BeautifulSoup) -> str | None:
    """Extract stock/availability status."""
    # Itemprop
    avail_elem = soup.find(attrs={"itemprop": "availability"})
    if avail_elem:
        val = avail_elem.get("content") or avail_elem.get("href") or avail_elem.get_text()
        if val:
            return val.strip()

    # Common selectors
    for selector in [".stock-status", ".availability", ".in-stock", ".out-of-stock"]:
        elem = soup.select_one(selector)
        if elem and elem.get_text(strip=True):
            return elem.get_text(strip=True)

    return None


def _extract_sizes(soup: BeautifulSoup) -> list[str]:
    """Extract available sizes."""
    sizes: list[str] = []

    # Size selector dropdowns
    for selector in [
        ".size-selector option",
        "select[name*='size'] option",
        ".size-dropdown option",
    ]:
        options = soup.select(selector)
        for opt in options:
            val = opt.get("value") or opt.get_text(strip=True)
            if val and val.lower() not in ("", "select", "select size", "choose size", "choose"):
                sizes.append(val.strip())
        if sizes:
            return sizes

    # Size swatches / list items
    for selector in [
        ".size-list li",
        ".size-swatch",
        "[data-size]",
        ".size-option",
        ".size-chip",
    ]:
        elems = soup.select(selector)
        for elem in elems:
            val = elem.get("data-size") or elem.get("data-value") or elem.get_text(strip=True)
            if val:
                sizes.append(val.strip())
        if sizes:
            return sizes

    return sizes


def _extract_color(soup: BeautifulSoup) -> str | None:
    """Extract selected/displayed color."""
    for selector in [
        ".color-name",
        ".selected-color",
        ".color-label",
        ".pdp-color",
        "[data-testid='selected-color']",
    ]:
        elem = soup.select_one(selector)
        if elem and elem.get_text(strip=True):
            return elem.get_text(strip=True)

    # Data attributes
    for selector in ["[data-color]", "[data-colour]"]:
        elem = soup.select_one(selector)
        if elem:
            val = elem.get("data-color") or elem.get("data-colour")
            if val:
                return val.strip()

    return None


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _make_absolute(url: str, base_url: str) -> str:
    """Convert a potentially relative URL to absolute."""
    if url.startswith(("http://", "https://", "//")):
        if url.startswith("//"):
            return "https:" + url
        return url
    return urljoin(base_url, url)


def _parse_float(text: Any) -> float | None:
    """Try to parse a float from text."""
    if text is None:
        return None
    try:
        # Extract first number from text
        match = re.search(r"[\d.]+", str(text))
        if match:
            return float(match.group())
    except (ValueError, TypeError):
        pass
    return None


def _parse_int_from_text(text: Any) -> int | None:
    """Extract an integer from text like '(123 reviews)'."""
    if text is None:
        return None
    try:
        match = re.search(r"[\d,]+", str(text))
        if match:
            return int(match.group().replace(",", ""))
    except (ValueError, TypeError):
        pass
    return None
