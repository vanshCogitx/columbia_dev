"""
Unit and integration tests for the Google Product Search tool.
"""

from __future__ import annotations

import os
import sys
import pytest

# Add src folder to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import requests
from models.product import Product, SearchResponse, parse_price, normalize_availability
from search.result_parser import is_product_url, normalize_url
from search.google_client import GoogleSearchClient
from scraper.jsonld_parser import extract_jsonld_product
from scraper.html_parser import extract_html_product
from tool.google_product_search import TOOL_SCHEMA, google_product_search


def test_tool_schema():
    """Verify tool schema matches expected OpenAI/standard function description."""
    assert TOOL_SCHEMA["type"] == "function"
    func = TOOL_SCHEMA["function"]
    assert func["name"] == "google_product_search"
    assert "query" in func["parameters"]["properties"]
    assert "queries" in func["parameters"]["properties"]
    assert func["parameters"]["properties"]["queries"]["type"] == "array"


def test_parse_price():
    """Test price parser utility."""
    assert parse_price(10) == 10.0
    assert parse_price("123.45") == 123.45
    assert parse_price("₹4,499.00") == 4499.00
    assert parse_price("Rs. 499") == 499.0
    assert parse_price("Price: $99.99 USD") == 99.99
    assert parse_price(None) is None
    assert parse_price("not a price") is None


def test_normalize_availability():
    """Test stock status normalization."""
    assert normalize_availability("https://schema.org/InStock") == "In Stock"
    assert normalize_availability("InStock") == "In Stock"
    assert normalize_availability("in stock") == "In Stock"
    assert normalize_availability(True) == "In Stock"
    
    assert normalize_availability("https://schema.org/OutOfStock") == "Out of Stock"
    assert normalize_availability("out of stock") == "Out of Stock"
    assert normalize_availability(False) == "Out of Stock"
    
    assert normalize_availability("https://schema.org/LimitedAvailability") == "Limited Stock"
    assert normalize_availability("Pre-Order") == "Pre-Order"
    assert normalize_availability("random string") == "random string"


def test_url_filtering():
    """Test product-page identification heuristics."""
    # Product URLs
    assert is_product_url("https://amazon.in/dp/B08XYZ1234/ref=something") is True
    assert is_product_url("https://example.com/products/hiking-jacket-p-12345") is True
    assert is_product_url("https://example.com/p/men-fleece-jacket") is True
    
    # Non-Product URLs
    assert is_product_url("https://example.com/") is False  # homepage
    assert is_product_url("https://example.com/category/mens-clothing") is False
    assert is_product_url("https://example.com/blog/best-fleece-jackets") is False
    assert is_product_url("https://example.com/search?q=fleece") is False
    assert is_product_url("https://example.com/cart") is False


def test_url_normalization():
    """Test stripping of tracking params."""
    dirty = "https://example.com/product/123?utm_source=google&gclid=abc&ref=xyz&size=M"
    # size is a valid parameter to keep, others should be stripped
    clean = normalize_url(dirty)
    assert "utm_source" not in clean
    assert "gclid" not in clean
    assert "ref" not in clean
    assert "size=M" in clean


def test_product_model_from_raw():
    """Test raw product mapping & normalization."""
    raw = {
        "title": "Columbia Fleece Jacket",
        "brand_name": "Columbia",
        "price": "₹4,499.00",
        "mrp": "4,999.00",
        "description": "<p>Warm fleece jacket.</p>",
        "color": "Red",
        "sizes": ["S", "M", "L"],
        "availability": "https://schema.org/InStock",
        "image": "https://example.com/img.jpg"
    }
    prod = Product.from_raw(raw, "https://example.com/product/1")
    assert prod is not None
    assert prod.name == "Columbia Fleece Jacket"
    assert prod.brand == "Columbia"
    # Expect price converted to USD (INR -> USD conversion rate 0.012)
    expected_usd = round(4499.0 * 0.012, 2)
    assert prod.price == expected_usd
    assert prod.description == "Warm fleece jacket."
    assert prod.color == "Red"
    assert prod.sizes == ["S", "M", "L"]
    assert prod.availability == "In Stock"
    assert prod.image_url == "https://example.com/img.jpg"
    assert prod.product_url == "https://example.com/product/1"


def test_product_model_missing_name():
    """Verify that a product without a name is skipped (returns None)."""
    raw = {
        "price": 1000
    }
    prod = Product.from_raw(raw, "https://example.com/product/1")
    assert prod is None


def test_jsonld_parser():
    """Test schema.org JSON-LD extraction."""
    html = """
    <html>
      <body>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Product",
          "name": "Nike Air Zoom",
          "brand": {
            "@type": "Brand",
            "name": "Nike"
          },
          "offers": {
            "@type": "Offer",
            "price": "7999.00",
            "priceCurrency": "INR",
            "availability": "https://schema.org/InStock"
          }
        }
        </script>
      </body>
    </html>
    """
    data = extract_jsonld_product(html)
    assert data is not None
    assert data["name"] == "Nike Air Zoom"
    assert data["brand"] == "Nike"
    assert data["price"] == 7999.0
    assert data["priceCurrency"] == "INR"


def test_html_parser():
    """Test fallback BeautifulSoup extraction."""
    html = """
    <html>
      <head>
        <meta property="og:title" content="Adidas Running Shoes" />
        <meta property="og:description" content="Very comfortable running shoes" />
        <meta property="og:image" content="/images/shoes.png" />
      </head>
      <body>
        <span itemprop="price" content="5999">5999</span>
        <div class="stock-status">Out of Stock</div>
      </body>
    </html>
    """
    data = extract_html_product(html, "https://adidas.co.in/p/shoes")
    assert data is not None
    assert data["name"] == "Adidas Running Shoes"
    assert data["description"] == "Very comfortable running shoes"
    assert data["image"] == "https://adidas.co.in/images/shoes.png"
    assert data["price"] == "5999"
    assert data["availability"] == "Out of Stock"


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CX"),
    reason="GOOGLE_API_KEY or GOOGLE_CX not configured"
)
def test_live_search():
    """Test end-to-end live search (needs real API key)."""
    res_dict = google_product_search("Columbia men's fleece jacket", limit=2)
    response = SearchResponse(**res_dict)
    
    assert response.query == "Columbia men's fleece jacket"
    assert response.error is None
    assert response.source == "google_custom_search"
    # We should get at least some products unless search is rate-limited or fails
    if response.count > 0:
        assert len(response.products) <= 2
        for prod in response.products:
            assert prod.name is not None
            assert prod.product_url is not None
            # Check price/currency if available
            if prod.price is not None:
                assert isinstance(prod.price, float)


def test_google_api_response_parsing(monkeypatch):
    """Test parsing of Google Custom Search JSON response."""
    mock_response = {
        "items": [
            {
                "title": "Product 1",
                "link": "https://example.com/product/product-1",
                "snippet": "Snippet 1",
                "displayLink": "example.com",
                "pagemap": {
                    "cse_thumbnail": [{"src": "https://example.com/thumb1.jpg"}]
                }
            },
            {
                "title": "Product 2",
                "link": "https://example.com/product/product-2",
                "snippet": "Snippet 2",
                "displayLink": "example.com",
                "pagemap": {
                    "cse_image": [{"src": "https://example.com/img2.jpg"}]
                }
            }
        ]
    }
    
    class MockResponse:
        status_code = 200
        def json(self):
            return mock_response
            
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())
    
    client = GoogleSearchClient(api_key="fake_key", cx="fake_cx")
    results = client.search_products("test query")
    
    assert len(results) == 2
    assert results[0]["url"] == "https://example.com/product/product-1"
    assert results[0]["title"] == "Product 1"
    assert results[0]["snippet"] == "Snippet 1"
    assert results[0]["source"] == "example.com"
    assert results[0]["thumbnail"] == "https://example.com/thumb1.jpg"
    
    assert results[1]["url"] == "https://example.com/product/product-2"
    assert results[1]["thumbnail"] == "https://example.com/img2.jpg"


def test_config_google_keys():
    """Verify that GOOGLE_API_KEY and GOOGLE_CX are present in the Config class."""
    from config import Config
    assert "GOOGLE_API_KEY" in Config.model_fields
    assert "GOOGLE_CX" in Config.model_fields
    assert "SERPAPI_KEY" not in Config.model_fields


def test_google_product_search_limit(monkeypatch):
    """Verify limit validation and clamping."""
    import asyncio
    from scraper.product_scraper import ProductScraper

    class MockClient:
        def search_products(self, query, limit):
            return [
                {"url": f"https://example.com/product/{i}", "title": f"Prod {i}"}
                for i in range(10)
            ]

    async def mock_scrape(self, product_results, browser=None):
        return [{"name": r["title"], "product_url": r["url"]} for r in product_results]

    monkeypatch.setattr("tool.google_product_search.GoogleSearchClient", lambda *args, **kwargs: MockClient())
    monkeypatch.setattr(ProductScraper, "scrape_products", mock_scrape)

    res_dict = google_product_search(query="test query", limit=3)
    assert res_dict["count"] == 3
    assert len(res_dict["products"]) == 3
    assert res_dict["source"] == "google_custom_search"

    res_dict = google_product_search(query="test query", limit=5)
    assert res_dict["count"] == 5
    assert len(res_dict["products"]) == 5

    res_dict = google_product_search(query="test query", limit=10)
    assert res_dict["count"] == 5
    assert len(res_dict["products"]) == 5


@pytest.mark.asyncio
async def test_concurrent_scraping(monkeypatch):
    """Verify that product scraper initiates multiple concurrent scrape tasks."""
    import asyncio
    from scraper.product_scraper import ProductScraper
    
    class MockPage:
        async def route(self, pattern, handler):
            pass
        async def goto(self, url, timeout, wait_until):
            await asyncio.sleep(0.05)  # Simulate navigation delay
            return None
        async def wait_for_selector(self, selector, timeout):
            raise Exception()
        async def wait_for_function(self, fn, timeout):
            raise Exception()
        async def content(self):
            return "<html><body><h1>Product</h1></body></html>"
        async def close(self):
            pass

    class MockBrowserContext:
        async def new_page(self):
            return MockPage()
        async def close(self):
            pass

    class MockBrowser:
        async def new_context(self, **kwargs):
            return MockBrowserContext()
        async def close(self):
            pass

    class MockChromium:
        async def launch(self, **kwargs):
            return MockBrowser()

    class MockPlaywright:
        chromium = MockChromium()
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr("scraper.product_scraper.async_playwright", lambda: MockPlaywright())
    
    scraper = ProductScraper(max_concurrent=3)
    start_time = asyncio.get_event_loop().time()
    results = await scraper.scrape_products([
        {"url": "https://example.com/1", "skip_http": True},
        {"url": "https://example.com/2", "skip_http": True},
        {"url": "https://example.com/3", "skip_http": True},
    ])
    end_time = asyncio.get_event_loop().time()
    
    assert len(results) == 3
    assert (end_time - start_time) < 0.2


@pytest.mark.asyncio
async def test_navigation_timeout_graceful_recovery(monkeypatch):
    """Verify that a page.goto timeout still attempts parsing on loaded HTML."""
    from scraper.product_scraper import ProductScraper
    
    class MockPage:
        async def route(self, pattern, handler):
            pass
        async def goto(self, url, timeout, wait_until):
            raise TimeoutError("Navigation timeout")
        async def wait_for_selector(self, selector, timeout):
            raise Exception("Not found")
        async def wait_for_function(self, fn, timeout):
            raise Exception("Not found")
        async def content(self):
            return '<html><body><script type="application/ld+json">{"@type": "Product", "name": "Timed-out Product"}</script></body></html>'
        async def close(self):
            pass

    class MockBrowserContext:
        async def new_page(self):
            return MockPage()

    scraper = ProductScraper()
    result = await scraper._scrape_single_product(MockBrowserContext(), {"url": "https://example.com/timeout-test"})
    
    assert result is not None
    assert result["name"] == "Timed-out Product"


@pytest.mark.asyncio
async def test_jsonld_extraction_after_commit(monkeypatch):
    """Verify JSON-LD is extracted when wait_until='commit' is used."""
    from scraper.product_scraper import ProductScraper
    
    class MockPage:
        def __init__(self):
            self.wait_until_used = None
        async def route(self, pattern, handler):
            pass
        async def goto(self, url, timeout, wait_until):
            self.wait_until_used = wait_until
            return None
        async def wait_for_selector(self, selector, timeout):
            assert selector == 'script[type="application/ld+json"]'
            return None
        async def wait_for_function(self, fn, timeout):
            pass
        async def content(self):
            return '<html><body><script type="application/ld+json">{"@type": "Product", "name": "JSON-LD Product"}</script></body></html>'
        async def close(self):
            pass

    class MockBrowserContext:
        async def new_page(self):
            return MockPage()

    scraper = ProductScraper()
    result = await scraper._scrape_single_product(MockBrowserContext(), {"url": "https://example.com/jsonld-test"})
    
    assert result is not None
    assert result["name"] == "JSON-LD Product"


@pytest.mark.asyncio
async def test_fallback_to_embedded_json(monkeypatch):
    """Verify fallback to embedded JSON when JSON-LD is absent."""
    from scraper.product_scraper import ProductScraper
    
    class MockPage:
        async def route(self, pattern, handler):
            pass
        async def goto(self, url, timeout, wait_until):
            return None
        async def wait_for_selector(self, selector, timeout):
            raise Exception("Absent")
        async def wait_for_function(self, fn, timeout):
            return None
        async def content(self):
            return '<html><body><script id="__NEXT_DATA__">{"props": {"pageProps": {"product": {"name": "Embedded Product", "price": 100}}}}</script></body></html>'
        async def close(self):
            pass

    class MockBrowserContext:
        async def new_page(self):
            return MockPage()

    scraper = ProductScraper()
    result = await scraper._scrape_single_product(MockBrowserContext(), {"url": "https://example.com/embedded-test"})
    
    assert result is not None
    assert result["name"] == "Embedded Product"
    assert result["price"] == 100.0


@pytest.mark.asyncio
async def test_fallback_to_html_parser(monkeypatch):
    """Verify fallback to HTML parser when both JSON-LD and embedded JSON are absent."""
    from scraper.product_scraper import ProductScraper
    
    class MockPage:
        async def route(self, pattern, handler):
            pass
        async def goto(self, url, timeout, wait_until):
            return None
        async def wait_for_selector(self, selector, timeout):
            raise Exception("Absent")
        async def wait_for_function(self, fn, timeout):
            raise Exception("Absent")
        async def content(self):
            return '<html><head><meta property="og:title" content="HTML Product" /></head><body><h1>HTML Product</h1><span itemprop="price">150</span></body></html>'
        async def close(self):
            pass

    class MockBrowserContext:
        async def new_page(self):
            return MockPage()

    scraper = ProductScraper()
    result = await scraper._scrape_single_product(MockBrowserContext(), {"url": "https://example.com/html-test"})
    
    assert result is not None
    assert result["name"] == "HTML Product"
    assert result["price"] == "150"


@pytest.mark.asyncio
async def test_resource_blocking(monkeypatch):
    """Verify images, fonts, and media are blocked, while scripts and stylesheets are allowed."""
    from scraper.product_scraper import ProductScraper, _BLOCKED_RESOURCE_TYPES
    
    assert "image" in _BLOCKED_RESOURCE_TYPES
    assert "media" in _BLOCKED_RESOURCE_TYPES
    assert "font" in _BLOCKED_RESOURCE_TYPES
    assert "stylesheet" not in _BLOCKED_RESOURCE_TYPES
    assert "script" not in _BLOCKED_RESOURCE_TYPES
    
    class MockRequest:
        def __init__(self, resource_type):
            self.resource_type = resource_type
            
    class MockRoute:
        def __init__(self, request):
            self.request = request
            self.aborted = False
            self.continued = False
        def abort(self):
            self.aborted = True
        def continue_(self):
            self.continued = True
            
    class MockPage:
        def __init__(self):
            self.routes = []
        async def route(self, pattern, handler):
            self.routes.append((pattern, handler))
        async def goto(self, url, timeout, wait_until):
            return None
        async def wait_for_selector(self, selector, timeout):
            raise Exception()
        async def wait_for_function(self, fn, timeout):
            raise Exception()
        async def content(self):
            return '<html><body><h1>Product</h1></body></html>'
        async def close(self):
            pass

    class MockBrowserContext:
        def __init__(self, page_instance):
            self.page_instance = page_instance
        async def new_page(self):
            return self.page_instance

    scraper = ProductScraper()
    page = MockPage()
    ctx = MockBrowserContext(page)
    
    routes = []
    async def mock_route(pattern, handler):
        routes.append((pattern, handler))
    page.route = mock_route
    
    await scraper._scrape_single_product(ctx, {"url": "https://example.com"})
    
    assert len(routes) > 0
    pattern, handler = routes[0]
    
    for res_type in ["image", "media", "font"]:
        route = MockRoute(MockRequest(res_type))
        handler(route)
        assert route.aborted is True
        assert route.continued is False
        
    for res_type in ["stylesheet", "script", "document", "xhr"]:
        route = MockRoute(MockRequest(res_type))
        handler(route)
        assert route.aborted is False
        assert route.continued is True


def test_output_schema_unchanged():
    """Verify normalized output schema matches expected fields."""
    from models.product import Product
    fields = Product.model_fields
    assert "name" in fields
    assert "brand" in fields
    assert "price" in fields
    assert "description" in fields
    assert "color" in fields
    assert "sizes" in fields
    assert "availability" in fields
    assert "image_url" in fields
    assert "product_url" in fields

    assert fields["name"].annotation == str
    assert fields["brand"].annotation == str | None
    assert fields["price"].annotation == float | None
    assert fields["description"].annotation == str | None
    assert fields["color"].annotation == str | None
    assert fields["sizes"].annotation == list[str]
    assert fields["availability"].annotation == str | None
    assert fields["image_url"].annotation == str | None
    assert fields["product_url"].annotation == str


def test_batch_search_basic(monkeypatch):
    """Verify batch execution of multiple queries."""
    from scraper.product_scraper import ProductScraper

    class MockClient:
        def search_products(self, query, limit):
            return [
                {"url": f"https://example.com/{query.replace(' ', '-')}/{i}", "title": f"{query} Prod {i}"}
                for i in range(3)
            ]

    async def mock_scrape(self, product_results, browser=None):
        return [{"name": r["title"], "product_url": r["url"]} for r in product_results]

    monkeypatch.setattr("tool.google_product_search.GoogleSearchClient", lambda *args, **kwargs: MockClient())
    monkeypatch.setattr(ProductScraper, "scrape_products", mock_scrape)

    result = google_product_search(
        queries=["red jacket", "blue shoes"],
        limit=3
    )

    assert "results" in result
    assert len(result["results"]) == 2
    assert result["results"][0]["query"] == "red jacket"
    assert result["results"][1]["query"] == "blue shoes"
    assert result["results"][0]["count"] == 3
    assert result["results"][1]["count"] == 3


def test_batch_partial_query_failure(monkeypatch):
    """Verify that a failure in one query does not crash the other queries."""
    from scraper.product_scraper import ProductScraper

    class MockClient:
        def search_products(self, query, limit):
            if "fail" in query:
                raise RuntimeError("Simulated API failure")
            return [
                {"url": f"https://example.com/products/item-{i}", "title": f"Prod {i}"}
                for i in range(2)
            ]

    async def mock_scrape(self, product_results, browser=None):
        return [{"name": r["title"], "product_url": r["url"]} for r in product_results]

    monkeypatch.setattr("tool.google_product_search.GoogleSearchClient", lambda *args, **kwargs: MockClient())
    monkeypatch.setattr(ProductScraper, "scrape_products", mock_scrape)

    result = google_product_search(
        queries=["good query", "fail query", "another good"],
        limit=2
    )

    assert "results" in result
    assert len(result["results"]) == 3
    assert result["results"][0]["count"] == 2
    assert result["results"][0]["error"] is None
    assert result["results"][1]["count"] == 0
    assert result["results"][1]["error"] is not None
    assert result["results"][2]["count"] == 2
    assert result["results"][2]["error"] is None


def test_batch_validation():
    """Verify batch size validations and empty query rejections."""
    # No queries at all
    result = google_product_search()
    assert result["error"] is not None
    assert result["count"] == 0

    # All empty queries
    result = google_product_search(queries=["", "  ", ""])
    assert result["error"] is not None

    # Exceed max batch size
    from config import get_config
    config = get_config()
    too_many = [f"query {i}" for i in range(config.MAX_BATCH_QUERIES + 5)]
    result = google_product_search(queries=too_many)
    assert result["error"] is not None
    assert "exceeds" in result["error"].lower() or "maximum" in result["error"].lower()


def test_single_query_backward_compatibility(monkeypatch):
    """Verify that single query inputs continue to work and return backward-compatible response."""
    from scraper.product_scraper import ProductScraper

    class MockClient:
        def search_products(self, query, limit):
            return [{"url": "https://example.com/prod1", "title": "Product 1"}]

    async def mock_scrape(self, product_results, browser=None):
        return [{"name": "Product 1", "product_url": "https://example.com/prod1"}]

    monkeypatch.setattr("tool.google_product_search.GoogleSearchClient", lambda *args, **kwargs: MockClient())
    monkeypatch.setattr(ProductScraper, "scrape_products", mock_scrape)

    # Single query via 'query' parameter
    result = google_product_search(query="test product")
    assert "query" in result
    assert "count" in result
    assert "products" in result
    assert "results" not in result  # Should NOT be a batch response
    assert result["query"] == "test product"
    assert result["count"] == 1


def test_batch_response_schema(monkeypatch):
    """Verify the batch response schema is correct."""
    from scraper.product_scraper import ProductScraper
    from models.product import BatchSearchResponse, SearchResponse

    class MockClient:
        def search_products(self, query, limit):
            return [{"url": "https://example.com/p1", "title": "P1"}]

    async def mock_scrape(self, product_results, browser=None):
        return [{"name": "P1", "product_url": "https://example.com/p1"}]

    monkeypatch.setattr("tool.google_product_search.GoogleSearchClient", lambda *args, **kwargs: MockClient())
    monkeypatch.setattr(ProductScraper, "scrape_products", mock_scrape)

    result = google_product_search(queries=["q1", "q2"])
    assert "results" in result
    # Validate it can be parsed as BatchSearchResponse
    batch = BatchSearchResponse(**result)
    assert len(batch.results) == 2
    for r in batch.results:
        assert isinstance(r, SearchResponse)


def test_fastapi_endpoints(monkeypatch):
    """Verify FastAPI search endpoints for text-only, queries-only, mixed, and error payloads."""
    from fastapi.testclient import TestClient
    from main import app
    from scraper.product_scraper import ProductScraper

    class MockClient:
        def search_products(self, query, limit):
            return [{"url": "https://example.com/products/item-1", "title": f"Result for {query}"}]

    async def mock_scrape(self, product_results, browser=None):
        return [{"name": r["title"], "product_url": r["url"]} for r in product_results]

    monkeypatch.setattr("tool.google_product_search.GoogleSearchClient", lambda *args, **kwargs: MockClient())
    monkeypatch.setattr(ProductScraper, "scrape_products", mock_scrape)

    client = TestClient(app)

    # 1. Text-only request (returns SearchResponse)
    resp = client.post("/api/search", json={"text": "shoes", "limit": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "products" in data
    assert "results" not in data
    assert data["query"] == "shoes"
    assert len(data["products"]) == 1

    # 2. Queries-only request (returns BatchSearchResponse)
    resp = client.post("/api/search", json={"queries": ["boots", "pants"], "limit": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["query"] == "boots"
    assert data["results"][1]["query"] == "pants"

    # 3. Single query in queries array (returns BatchSearchResponse)
    resp = client.post("/api/search", json={"queries": ["hat"], "limit": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["query"] == "hat"

    # 4. Mixed text and queries request (returns BatchSearchResponse)
    resp = client.post("/api/search", json={"text": "shirt", "queries": ["jacket", "socks"], "limit": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 3
    assert data["results"][0]["query"] == "shirt"
    assert data["results"][1]["query"] == "jacket"
    assert data["results"][2]["query"] == "socks"

    # 5. Missing both text and queries (returns 422 Unprocessable Entity)
    resp = client.post("/api/search", json={"limit": 2})
    assert resp.status_code == 422

    # 6. Legacy /search endpoint backward compatibility (returns SearchResponse)
    resp = client.post("/search", json={"query": "gloves", "limit": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "products" in data
    assert "results" not in data
    assert data["query"] == "gloves"


# --- HTTP-First Scraping Architecture Tests ---

class MockHTTPResponse:
    def __init__(self, status_code, text, content=None):
        self.status_code = status_code
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")


class MockHTTPClient:
    def __init__(self, get_func):
        self.get_func = get_func

    async def get(self, url, **kwargs):
        return await self.get_func(url, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
async def test_http_fetch_success():
    """Verify that a successful HTTP fetch with JSON-LD parses and avoids Playwright fallback."""
    from scraper.product_scraper import ProductScraper
    
    html = """
    <html>
      <body>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Product",
          "name": "HTTP JSONLD Jacket",
          "brand": "Columbia",
          "offers": {
            "@type": "Offer",
            "price": "4999.00",
            "priceCurrency": "INR"
          }
        }
        </script>
      </body>
    </html>
    """
    async def mock_get(url, **kwargs):
        return MockHTTPResponse(200, html)

    client = MockHTTPClient(mock_get)
    scraper = ProductScraper()
    
    class MockBrowserContext:
        async def new_page(self):
            # This should NOT be called since HTTP succeeds
            raise AssertionError("Playwright context was accessed unexpectedly!")

    result = await scraper._scrape_single_product(
        MockBrowserContext(),
        {"url": "https://columbiasportswear.co.in/prod-1"},
        http_client=client
    )
    
    assert result is not None
    assert result["name"] == "HTTP JSONLD Jacket"
    assert result["brand"] == "Columbia"
    assert result["price"] == 4999.0
    assert result["priceCurrency"] == "INR"


@pytest.mark.asyncio
async def test_http_product_group_success():
    """Verify that an HTTP fetch parses ProductGroup structures successfully."""
    from scraper.product_scraper import ProductScraper
    
    html = """
    <html>
      <body>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "ProductGroup",
          "name": "Columbia fleece group",
          "brand": "Columbia",
          "hasVariant": [
            {
              "@type": "Product",
              "name": "Columbia fleece group - S",
              "sku": "123-S",
              "offers": {
                "@type": "Offer",
                "price": "3999.00",
                "priceCurrency": "INR",
                "availability": "https://schema.org/InStock"
              }
            }
          ]
        }
        </script>
      </body>
    </html>
    """
    async def mock_get(url, **kwargs):
        return MockHTTPResponse(200, html)

    client = MockHTTPClient(mock_get)
    scraper = ProductScraper()
    
    result = await scraper._scrape_single_product(
        None, # No context needed since HTTP succeeds
        {"url": "https://columbiasportswear.co.in/prod-group"},
        http_client=client
    )
    
    assert result is not None
    assert result["name"] == "Columbia fleece group"
    assert result["brand"] == "Columbia"
    assert result["price"] == 3999.0
    assert result["priceCurrency"] == "INR"
    assert "S" in result["sizes"]


@pytest.mark.asyncio
async def test_http_html_fallback_bs4():
    """Verify HTML DOM parsing is used when JSON-LD is absent in HTTP response."""
    from scraper.product_scraper import ProductScraper
    
    html = """
    <html>
      <head>
        <meta property="og:title" content="HTML PDP Boots" />
        <meta property="product:brand" content="Decathlon" />
      </head>
      <body>
        <span itemprop="price" content="2499">Rs. 2,499</span>
      </body>
    </html>
    """
    async def mock_get(url, **kwargs):
        return MockHTTPResponse(200, html)

    client = MockHTTPClient(mock_get)
    scraper = ProductScraper()
    
    result = await scraper._scrape_single_product(
        None,
        {"url": "https://decathlon.in/boots"},
        http_client=client
    )
    
    assert result is not None
    assert result["name"] == "HTML PDP PDP Boots" or result["name"] == "HTML PDP Boots"
    assert result["brand"] == "Decathlon"
    assert result["price"] == "2499"


@pytest.mark.asyncio
async def test_http_failure_to_playwright_fallback():
    """Verify that a 403 or 500 error triggers Playwright fallback."""
    from scraper.product_scraper import ProductScraper
    
    async def mock_get(url, **kwargs):
        return MockHTTPResponse(403, "Forbidden")

    client = MockHTTPClient(mock_get)
    scraper = ProductScraper()

    class MockPage:
        async def route(self, pattern, handler):
            pass
        async def goto(self, url, timeout, wait_until):
            return None
        async def wait_for_selector(self, selector, timeout):
            raise Exception("not found")
        async def wait_for_function(self, fn, timeout):
            raise Exception("not found")
        async def content(self):
            return '<html><body><script type="application/ld+json">{"@type": "Product", "name": "Playwright Fallback Item"}</script></body></html>'
        async def close(self):
            pass

    class MockBrowserContext:
        async def new_page(self):
            return MockPage()

    result = await scraper._scrape_single_product(
        MockBrowserContext(),
        {"url": "https://protected-site.com/p"},
        http_client=client
    )

    assert result is not None
    assert result["name"] == "Playwright Fallback Item"


@pytest.mark.asyncio
async def test_http_timeout_to_playwright_fallback():
    """Verify that an HTTP client timeout triggers Playwright fallback."""
    from scraper.product_scraper import ProductScraper
    
    async def mock_get(url, **kwargs):
        raise TimeoutError("Connection timed out")

    client = MockHTTPClient(mock_get)
    scraper = ProductScraper()

    class MockPage:
        async def route(self, pattern, handler):
            pass
        async def goto(self, url, timeout, wait_until):
            return None
        async def wait_for_selector(self, selector, timeout):
            raise Exception("not found")
        async def wait_for_function(self, fn, timeout):
            raise Exception("not found")
        async def content(self):
            return '<html><body><script type="application/ld+json">{"@type": "Product", "name": "Playwright Fallback timeout Item"}</script></body></html>'
        async def close(self):
            pass

    class MockBrowserContext:
        async def new_page(self):
            return MockPage()

    result = await scraper._scrape_single_product(
        MockBrowserContext(),
        {"url": "https://slow-site.com/p"},
        http_client=client
    )

    assert result is not None
    assert result["name"] == "Playwright Fallback timeout Item"


def test_missing_optional_fields_null():
    """Verify that products missing optional fields are parsed successfully and fields set to None."""
    from models.product import Product
    
    raw = {
        "name": "Minimal Product Jacket"
    }
    prod = Product.from_raw(raw, "https://example.com/min")
    assert prod is not None
    assert prod.name == "Minimal Product Jacket"
    assert prod.brand is None
    assert prod.price is None



@pytest.mark.asyncio
async def test_concurrent_http_scraping(monkeypatch):
    """Verify that multiple concurrent product page fetches run concurrently using http client."""
    from scraper.product_scraper import ProductScraper
    import time
    import httpx
    import asyncio
    
    html = '<html><body><script type="application/ld+json">{"@type": "Product", "name": "Fast Item"}</script></body></html>'
    
    async def mock_get(url, **kwargs):
        await asyncio.sleep(0.05) # simulate fetch delay
        return MockHTTPResponse(200, html)

    class DummyBrowserContext:
        async def new_page(self):
            raise AssertionError("Should not be called")
        async def close(self):
            pass

    class DummyBrowser:
        async def new_context(self, **kwargs):
            return DummyBrowserContext()
        async def close(self):
            pass

    client = MockHTTPClient(mock_get)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    
    # Set max_concurrent=3
    scraper = ProductScraper(max_concurrent=3)
    
    start_time = asyncio.get_event_loop().time()
    results = await scraper.scrape_products(
        [
            {"url": "https://fast-site.com/1"},
            {"url": "https://fast-site.com/2"},
            {"url": "https://fast-site.com/3"},
        ],
        browser=DummyBrowser()
    )
    end_time = asyncio.get_event_loop().time()
    
    # If sequential, total time would be >= 0.15s. If concurrent, it will be close to 0.05s.
    assert len(results) == 3
    assert (end_time - start_time) < 0.12

