"""
JSON-LD / schema.org product data extractor.

Parses <script type="application/ld+json"> blocks to find Product structured
data. Handles @graph wrappers, nested offers, aggregateRating, and variant
data.
"""

from __future__ import annotations

import json
import logging
import re
import html
from typing import Any

logger = logging.getLogger(__name__)

_LD_JSON_RE = re.compile(
    r'<script\b[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE
)


def extract_jsonld_product(html_content: str) -> dict[str, Any] | None:
    """Extract product data from JSON-LD structured data in the HTML.

    Args:
        html_content: Full HTML page source.

    Returns:
        A flat dict of product fields, or None if no Product found.
    """
    if not html_content:
        return None

    matches = _LD_JSON_RE.findall(html_content)
    for text in matches:
        if not text or not text.strip():
            continue

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            try:
                data = json.loads(html.unescape(text))
            except Exception:
                continue

        product = _find_product_in_jsonld(data)
        if product:
            ld_type = product.get("@type", "")
            if isinstance(ld_type, list):
                type_list = [t.lower() for t in ld_type if isinstance(t, str)]
            elif isinstance(ld_type, str):
                type_list = [ld_type.lower()]
            else:
                type_list = []

            if "productgroup" in type_list:
                return _flatten_product_group(product)
            else:
                return _flatten_product(product)

    return None


def _find_product_in_jsonld(data: Any) -> dict | None:
    """Recursively search JSON-LD data for a Product or ProductGroup typed object."""
    if isinstance(data, list):
        for item in data:
            result = _find_product_in_jsonld(item)
            if result:
                return result
        return None

    if not isinstance(data, dict):
        return None

    # Check @graph wrapper
    if "@graph" in data:
        return _find_product_in_jsonld(data["@graph"])

    # Check @type
    ld_type = data.get("@type", "")
    if isinstance(ld_type, list):
        type_list = [t.lower() for t in ld_type if isinstance(t, str)]
    elif isinstance(ld_type, str):
        type_list = [ld_type.lower()]
    else:
        type_list = []

    if "product" in type_list or "productgroup" in type_list:
        return data

    # Some pages nest Product inside other objects
    for key in ("mainEntity", "mainEntityOfPage", "itemListElement"):
        if key in data:
            result = _find_product_in_jsonld(data[key])
            if result:
                return result

    return None


def _flatten_product_group(pg: dict[str, Any]) -> dict[str, Any]:
    """Convert a schema.org ProductGroup object into a flat dict."""
    result: dict[str, Any] = {}
    result["name"] = pg.get("name")
    
    brand = pg.get("brand")
    if isinstance(brand, dict):
        result["brand"] = brand.get("name")
    elif isinstance(brand, str):
        result["brand"] = brand
    else:
        result["brand"] = None

    result["description"] = pg.get("description")

    # Extract from variants
    variants = pg.get("hasVariant", [])
    if not isinstance(variants, list):
        variants = [variants]

    # Find first variant to extract base details
    if variants:
        # Prefer in-stock variant if available
        in_stock_variant = None
        for v in variants:
            if isinstance(v, dict):
                offer = v.get("offers")
                if isinstance(offer, dict):
                    avail = offer.get("availability") or ""
                    if "InStock" in str(avail):
                        in_stock_variant = v
                        break

        ref_variant = in_stock_variant or (variants[0] if isinstance(variants[0], dict) else {})

        # SKU
        result["sku"] = ref_variant.get("sku") or ref_variant.get("gtin") or pg.get("productGroupID")

        # Image
        image = ref_variant.get("image")
        if isinstance(image, str):
            result["image"] = image
        elif isinstance(image, list) and image:
            result["image"] = image[0]
        else:
            result["image"] = pg.get("image")

        # Parse offers from ref_variant
        if "offers" in ref_variant:
            temp_prod = {"offers": ref_variant["offers"]}
            _parse_offers(temp_prod, result)

        # Collect sizes from variants
        sizes = []
        for v in variants:
            if not isinstance(v, dict):
                continue
            v_name = v.get("name") or ""
            # Extract size from variant name like "Columbia Men Red Backbowl II Fleece Jacket - S"
            if " - " in v_name:
                parts = v_name.split(" - ")
                size_candidate = parts[-1].strip()
                if re.match(r"^(XS|S|M|L|XL|XXL|XXXL|\d{1,3})$", size_candidate, re.IGNORECASE):
                    sizes.append(size_candidate)
            elif v.get("sku"):
                sku_parts = v["sku"].split("-")
                if len(sku_parts) > 1:
                    size_candidate = sku_parts[-1].strip()
                    if re.match(r"^(XS|S|M|L|XL|XXL|XXXL|\d{1,3})$", size_candidate, re.IGNORECASE):
                        sizes.append(size_candidate)

        result["sizes"] = list(dict.fromkeys(sizes))
    else:
        result["sku"] = pg.get("productGroupID")
        result["image"] = pg.get("image")
        result["sizes"] = []
        result["price"] = None
        result["priceCurrency"] = None
        result["availability"] = None

    # Aggregate Rating
    agg_rating = pg.get("aggregateRating")
    if isinstance(agg_rating, dict):
        try:
            result["ratingValue"] = float(agg_rating.get("ratingValue", 0))
        except (ValueError, TypeError):
            result["ratingValue"] = None
        try:
            result["reviewCount"] = int(
                agg_rating.get("reviewCount") or agg_rating.get("ratingCount") or 0
            )
        except (ValueError, TypeError):
            result["reviewCount"] = None
    else:
        result["ratingValue"] = None
        result["reviewCount"] = None

    return result


def _flatten_product(product: dict[str, Any]) -> dict[str, Any]:
    """Convert a schema.org Product object into a flat dict."""
    result: dict[str, Any] = {}

    # Name
    result["name"] = product.get("name")

    # Brand
    brand = product.get("brand")
    if isinstance(brand, dict):
        result["brand"] = brand.get("name")
    elif isinstance(brand, str):
        result["brand"] = brand
    else:
        result["brand"] = None

    # Description
    result["description"] = product.get("description")

    # SKU
    result["sku"] = product.get("sku") or product.get("mpn") or product.get("gtin13")

    # Image
    image = product.get("image")
    if isinstance(image, list):
        if image:
            first = image[0]
            result["image"] = first.get("url") if isinstance(first, dict) else first
        else:
            result["image"] = None
    elif isinstance(image, dict):
        result["image"] = image.get("url") or image.get("contentUrl")
    elif isinstance(image, str):
        result["image"] = image
    else:
        result["image"] = None

    # Color
    color = product.get("color")
    if isinstance(color, dict):
        result["color"] = color.get("name")
    elif isinstance(color, str):
        result["color"] = color
    else:
        result["color"] = _extract_from_additional_properties(product, "color")

    # Sizes from additionalProperty
    sizes = _extract_from_additional_properties(product, "size")
    if isinstance(sizes, list):
        result["sizes"] = sizes
    elif isinstance(sizes, str):
        result["sizes"] = [sizes]
    else:
        result["sizes"] = []

    # Offers → price, currency, availability, original_price
    _parse_offers(product, result)

    # Aggregate Rating
    agg_rating = product.get("aggregateRating")
    if isinstance(agg_rating, dict):
        try:
            result["ratingValue"] = float(agg_rating.get("ratingValue", 0))
        except (ValueError, TypeError):
            result["ratingValue"] = None
        try:
            result["reviewCount"] = int(
                agg_rating.get("reviewCount") or agg_rating.get("ratingCount") or 0
            )
        except (ValueError, TypeError):
            result["reviewCount"] = None
    else:
        result["ratingValue"] = None
        result["reviewCount"] = None

    return result


def _parse_offers(product: dict, result: dict) -> None:
    """Extract pricing and availability from offers."""
    offers = product.get("offers")
    if not offers:
        result.setdefault("price", None)
        result.setdefault("priceCurrency", None)
        result.setdefault("availability", None)
        return

    # Normalise to a list
    if isinstance(offers, dict):
        offer_type = offers.get("@type", "")
        if isinstance(offer_type, str) and offer_type.lower() == "aggregateoffer":
            # AggregateOffer — try lowPrice
            result["price"] = _safe_float(offers.get("lowPrice") or offers.get("price"))
            result["original_price"] = _safe_float(offers.get("highPrice"))
            result["priceCurrency"] = offers.get("priceCurrency")
            result["availability"] = _clean_availability(offers.get("availability"))
            # Check for nested offers array inside AggregateOffer
            inner = offers.get("offers")
            if isinstance(inner, list) and inner:
                _extract_sizes_from_offers(inner, result)
            return
        offer_list = [offers]
    elif isinstance(offers, list):
        offer_list = offers
    else:
        result.setdefault("price", None)
        result.setdefault("priceCurrency", None)
        result.setdefault("availability", None)
        return

    # Find the first offer with a price
    best_offer = None
    sizes_from_offers: list[str] = []
    for offer in offer_list:
        if not isinstance(offer, dict):
            continue
        price = offer.get("price")
        if price is not None and best_offer is None:
            best_offer = offer
        # Collect sizes from variant offers
        size = offer.get("size") or offer.get("name")
        if size and isinstance(size, str):
            # Only add if it looks like a size
            if re.match(r"^(XS|S|M|L|XL|XXL|XXXL|\d{1,3})$", size.strip(), re.IGNORECASE):
                sizes_from_offers.append(size.strip())

    if best_offer:
        result["price"] = _safe_float(best_offer.get("price"))
        result["priceCurrency"] = best_offer.get("priceCurrency")
        result["availability"] = _clean_availability(best_offer.get("availability"))

        # Check for priceSpecification for original price
        price_spec = best_offer.get("priceSpecification")
        if isinstance(price_spec, dict):
            result["original_price"] = _safe_float(
                price_spec.get("price") or price_spec.get("maxPrice")
            )
    else:
        result.setdefault("price", None)
        result.setdefault("priceCurrency", None)
        result.setdefault("availability", None)

    if sizes_from_offers and not result.get("sizes"):
        result["sizes"] = sizes_from_offers


def _extract_sizes_from_offers(offers: list, result: dict) -> None:
    """Extract size options from a list of variant offers."""
    sizes: list[str] = []
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        size = offer.get("size") or offer.get("name")
        if size and isinstance(size, str):
            sizes.append(size.strip())
    if sizes and not result.get("sizes"):
        result["sizes"] = sizes


def _extract_from_additional_properties(
    product: dict, prop_name: str
) -> Any:
    """Search additionalProperty array for a named property."""
    props = product.get("additionalProperty", [])
    if not isinstance(props, list):
        return None

    values: list[str] = []
    for prop in props:
        if not isinstance(prop, dict):
            continue
        name = (prop.get("name") or "").lower()
        if name == prop_name.lower():
            val = prop.get("value")
            if val:
                values.append(str(val))

    if not values:
        return None
    return values if len(values) > 1 else values[0]


def _clean_availability(value: Any) -> str | None:
    """Strip schema.org URL prefix from availability values."""
    if not value:
        return None
    text = str(value).strip()
    # Remove schema.org prefix
    text = re.sub(r"https?://schema\.org/", "", text)
    return text if text else None


def _safe_float(value: Any) -> float | None:
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None
