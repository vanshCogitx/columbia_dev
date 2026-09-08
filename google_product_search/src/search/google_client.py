"""
Google Search client using Google Custom Search JSON API.

Provides product search via Google Custom Search API.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests

from config import get_config

logger = logging.getLogger(__name__)

# Known e-commerce domains for site-scoping heuristic
_KNOWN_DOMAINS: dict[str, str] = {
    "amazon": "amazon.in",
    "flipkart": "flipkart.com",
    "myntra": "myntra.com",
    "ajio": "ajio.com",
    "columbia": "columbiasportswear.co.in",
    "nike": "nike.com",
    "adidas": "adidas.co.in",
    "puma": "puma.com",
    "decathlon": "decathlon.in",
    "tatacliq": "tatacliq.com",
    "nykaa": "nykaafashion.com",
    "meesho": "meesho.com",
}

_SITE_RE = re.compile(r"site:(\S+)", re.IGNORECASE)


def detect_site_domain(query: str) -> str | None:
    """Detect if query references a specific domain or brand site.

    Returns the domain string if found (e.g. 'amazon.in'), else None.
    """
    # Explicit site: operator
    match = _SITE_RE.search(query)
    if match:
        return match.group(1).lower()

    # Check for known brand names in the query
    query_lower = query.lower()
    for brand, domain in _KNOWN_DOMAINS.items():
        # Match whole word to avoid false positives
        if re.search(rf"\b{re.escape(brand)}\b", query_lower):
            return domain

    return None


def _clean_query_for_search(query: str) -> str:
    """Remove site: operators from the query string for API use."""
    return _SITE_RE.sub("", query).strip()


class GoogleSearchClient:
    """Google Custom Search-based product search client.

    Retrieves search results using the Google Custom Search JSON API.
    """

    def __init__(self, api_key: str | None = None, cx: str | None = None) -> None:
        config = get_config()
        self._api_key = api_key or config.GOOGLE_API_KEY
        self._cx = cx or config.GOOGLE_CX
        self._country = config.SEARCH_COUNTRY
        self._language = config.SEARCH_LANGUAGE

    def search_products(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search Google Custom Search for products.

        Args:
            query: Natural-language product query.
            limit: Desired number of results (capped at 5).

        Returns:
            A list of search result dicts mapped to the internal representation.
        """
        if not self._api_key or not self._cx:
            raise ValueError("Google API Key or CX (Search Engine ID) is not configured.")

        clean_query = _clean_query_for_search(query)
        # Force Columbia Sportswear site
        domain = "columbiasportswear.co.in"
        search_query = f"site:{domain} {clean_query}"
        logger.info("Scoped search to default domain: %s", domain)

        params = {
            "key": self._api_key,
            "cx": self._cx,
            "q": search_query,
            "num": 10,  # Request up to 10 items to allow downstream filtering/deduplication
        }

        if self._country:
            params["gl"] = self._country
        if self._language:
            params["hl"] = self._language

        try:
            logger.info("Sending request to Google Custom Search API for query: '%s'", search_query)
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=10,
            )

            if response.status_code != 200:
                try:
                    error_detail = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    error_detail = response.text
                raise Exception(f"Google Custom Search API error (status {response.status_code}): {error_detail}")

            data = response.json()
            items = data.get("items", [])

            results: list[dict[str, Any]] = []
            for item in items:
                link = item.get("link")
                if not link:
                    continue

                # Extract potential thumbnail URL from pagemap
                thumbnail = None
                pagemap = item.get("pagemap", {})
                if pagemap:
                    cse_thumb = pagemap.get("cse_thumbnail")
                    if cse_thumb and isinstance(cse_thumb, list) and cse_thumb[0].get("src"):
                        thumbnail = cse_thumb[0].get("src")
                    else:
                        cse_img = pagemap.get("cse_image")
                        if cse_img and isinstance(cse_img, list) and cse_img[0].get("src"):
                            thumbnail = cse_img[0].get("src")

                results.append({
                    "url": link,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("displayLink", ""),
                    "thumbnail": thumbnail,
                })

            logger.info(
                "Google Custom Search returned %d items, mapped to %d results for query: %s",
                len(items),
                len(results),
                search_query,
            )
            return results

        except Exception as exc:
            logger.error("Google Custom Search request failed: %s", exc)
            raise
