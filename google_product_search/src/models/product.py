"""
Product and SearchResponse models with normalisation helpers.

Defines the canonical product schema and provides robust field-mapping,
price-parsing, and availability-normalisation utilities.
"""

from __future__ import annotations

import html as html_module
import logging
import re
from typing import Any

# Conversion rates to USD (approximate)
_CONVERSION_RATES = {
    "INR": 0.012,   # 1 INR ≈ 0.012 USD
    "EUR": 1.10,    # 1 EUR ≈ 1.10 USD
    "GBP": 1.25,    # 1 GBP ≈ 1.25 USD
    "JPY": 0.0091,  # 1 JPY ≈ 0.0091 USD
    "KRW": 0.00085, # 1 KRW ≈ 0.00085 USD
}

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS: dict[str, str] = {
    "₹": "INR",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₩": "KRW",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PRICE_RE = re.compile(r"[\d,]+\.?\d*")

_AVAILABILITY_MAP: dict[str, str] = {
    "https://schema.org/instock": "In Stock",
    "http://schema.org/instock": "In Stock",
    "instock": "In Stock",
    "in stock": "In Stock",
    "in_stock": "In Stock",
    "https://schema.org/outofstock": "Out of Stock",
    "http://schema.org/outofstock": "Out of Stock",
    "outofstock": "Out of Stock",
    "out of stock": "Out of Stock",
    "out_of_stock": "Out of Stock",
    "https://schema.org/preorder": "Pre-Order",
    "http://schema.org/preorder": "Pre-Order",
    "preorder": "Pre-Order",
    "pre-order": "Pre-Order",
    "https://schema.org/limitedavailability": "Limited Stock",
    "http://schema.org/limitedavailability": "Limited Stock",
    "limitedavailability": "Limited Stock",
    "limited availability": "Limited Stock",
    "https://schema.org/discontinued": "Discontinued",
    "http://schema.org/discontinued": "Discontinued",
    "discontinued": "Discontinued",
}


def parse_price(value: Any) -> float | None:
    """Extract a numeric price from various representations.

    Handles currency symbols, commas, whitespace, and string prefixes like 'Rs.'.
    Returns *None* when the value cannot be parsed.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    text = value.strip()
    if not text:
        return None

    # Remove known currency symbols and prefixes
    for sym in _CURRENCY_SYMBOLS:
        text = text.replace(sym, "")
    for prefix in ("Rs.", "Rs", "INR", "USD", "EUR", "GBP", "MRP", "Price:"):
        text = text.replace(prefix, "")

    text = text.replace(",", "").replace(" ", "").strip()

    match = _PRICE_RE.search(text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    return None


def detect_currency(value: Any) -> str | None:
    """Detect currency code from a price string or explicit currency value."""
    if value is None:
        return None

    text = str(value).strip().upper()

    # Direct code match
    if text in ("INR", "USD", "EUR", "GBP", "JPY", "KRW"):
        return text

    # Symbol detection
    raw = str(value)
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in raw:
            return code

    if "Rs" in raw:
        return "INR"

    return None


def clean_html(text: str | None) -> str | None:
    """Strip HTML tags, decode entities, and collapse whitespace."""
    if not text:
        return None

    cleaned = _HTML_TAG_RE.sub(" ", text)
    cleaned = html_module.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else None


def normalize_availability(value: Any) -> str | None:
    """Normalise availability strings to a canonical form."""
    if value is None:
        return None

    if isinstance(value, bool):
        return "In Stock" if value else "Out of Stock"

    text = str(value).strip()
    if not text:
        return None

    lookup = text.lower()
    if lookup in _AVAILABILITY_MAP:
        return _AVAILABILITY_MAP[lookup]

    # Partial matching
    lower = lookup
    if "in stock" in lower or "instock" in lower:
        return "In Stock"
    if "out of stock" in lower or "outofstock" in lower or "sold out" in lower:
        return "Out of Stock"
    if "pre" in lower and "order" in lower:
        return "Pre-Order"
    if "limited" in lower:
        return "Limited Stock"

    return text  # Return raw string if no mapping


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _get_first(data: dict, keys: list[str], default: Any = None) -> Any:
    """Return the first non-None value found in *data* for the given *keys*."""
    for key in keys:
        val = data.get(key)
        if val is not None and val != "" and val != []:
            return val
    return default


def _get_nested(data: dict, path: str, default: Any = None) -> Any:
    """Traverse *data* using a dot-separated *path* (e.g. 'brand.name')."""
    parts = path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default
        if current is None:
            return default
    return current


def _extract_image(value: Any) -> str | None:
    """Normalise image field to a single URL string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("url") or first.get("contentUrl")
    if isinstance(value, dict):
        return value.get("url") or value.get("contentUrl")
    return None


def _extract_sizes(data: dict) -> list[str]:
    """Extract sizes list from raw data."""
    sizes = _get_first(data, ["sizes", "available_sizes", "size_options", "availableSizes"])
    if isinstance(sizes, list):
        return [str(s).strip() for s in sizes if s]
    if isinstance(sizes, str):
        # "S, M, L, XL" or "S/M/L/XL"
        parts = re.split(r"[,/|]", sizes)
        return [p.strip() for p in parts if p.strip()]
    return []


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Product(BaseModel):
    """Canonical product schema."""

    name: str
    brand: str | None = None
    price: float | None = None
    description: str | None = None
    color: str | None = None
    sizes: list[str] = Field(default_factory=list)
    availability: str | None = None
    image_url: str | None = None
    product_url: str

    @classmethod
    def from_raw(
        cls,
        raw_data: dict[str, Any],
        product_url: str,
        serp_meta: dict[str, Any] | None = None,
    ) -> Product | None:
        """Build a Product from heterogeneous raw data.

        Returns *None* if a product name cannot be determined.
        """
        if not raw_data:
            raw_data = {}
        if serp_meta is None:
            serp_meta = {}

        data = {**raw_data}  # work on a copy

        # --- Name ---
        name = _get_first(data, ["name", "title", "productName", "product_name", "heading"])
        if not name:
            name = serp_meta.get("title")
        if not name:
            return None
        name = clean_html(str(name)) or str(name)

        # --- Brand ---
        brand_raw = _get_first(data, ["brand", "manufacturer", "brand_name"])
        if isinstance(brand_raw, dict):
            brand = brand_raw.get("name")
        elif brand_raw:
            brand = str(brand_raw).strip()
        else:
            brand = _get_nested(data, "brand.name")
        if not brand:
            brand = serp_meta.get("source")  # SERP source is often the brand

        if brand:
            brand_lower = brand.lower()
            if "columbia" in brand_lower or "columbiasportswear" in brand_lower:
                brand = "Columbia"

        # --- Price ---
        price_raw = _get_first(data, [
            "price", "current_price", "sale_price", "offer_price",
            "extracted_price", "selling_price",
        ])
        if price_raw is None:
            price_raw = _get_nested(data, "offers.price") or _get_nested(data, "offers.lowPrice")
        # Detect currency code from raw price string if possible; fall back to explicit priceCurrency field
        currency_code = detect_currency(price_raw) or _get_first(data, ["priceCurrency", "currency", "price_currency"])  # noqa: E501

        price = parse_price(price_raw)
        if price is None:
            # Fallback to SERP metadata for price
            price = parse_price(serp_meta.get("price") or serp_meta.get("extracted_price"))
            if price is None:
                currency_code = None
        # If we have price but no currency_code, try serp_meta for currency
        if price is not None and not currency_code:
            currency_code = serp_meta.get("priceCurrency") or serp_meta.get("currency")
        # Convert to USD if needed
        if price is not None and currency_code and currency_code != "USD":
            rate = _CONVERSION_RATES.get(currency_code)
            if rate:
                price = round(price * rate, 2)

        # --- Description ---
        desc = _get_first(data, ["description", "short_description", "productDescription", "shortDescription"])
        desc = clean_html(str(desc)) if desc else None
        if desc and len(desc) > 500:
            desc = desc[:497] + "..."

        # --- Color ---
        color = _get_first(data, ["color", "colour", "colorName", "color_name"])
        if isinstance(color, dict):
            color = color.get("name") or color.get("label")
        if color:
            color = str(color).strip()

        # --- Sizes ---
        sizes = _extract_sizes(data)

        # --- Availability ---
        avail_raw = _get_first(data, ["availability", "stock_status", "inStock", "stockStatus"])
        availability = normalize_availability(avail_raw)

        # --- Image URL ---
        image_url = _extract_image(
            _get_first(data, ["image", "image_url", "imageUrl", "primaryImage", "thumbnail"])
        )
        if not image_url:
            image_url = serp_meta.get("thumbnail")

        if image_url and image_url.startswith("http://"):
            image_url = "https://" + image_url[7:]

        try:
            return cls(
                name=name,
                brand=brand,
                price=price,
                description=desc,
                color=color,
                sizes=sizes,
                availability=availability,
                image_url=image_url,
                product_url=product_url,
            )
        except Exception as exc:
            logger.warning("Failed to create Product from raw data: %s", exc)
            return None


class SearchResponse(BaseModel):
    """Structured response from the google_product_search tool."""

    query: str
    count: int
    products: list[Product]
    source: str | None = "google_custom_search"
    error: str | None = None


class BatchSearchResponse(BaseModel):
    """Structured response for batch execution of the google_product_search tool."""
    results: list[SearchResponse]
