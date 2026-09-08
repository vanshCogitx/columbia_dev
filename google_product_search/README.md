# Google Product Search — AI Agent Tool

A standalone, modular Python tool that allows AI agents to search Google dynamically for products, crawl individual product pages via headless browser automation, extract and normalize product attributes, and return structured JSON.

## Features

- **Google Search Integration:** Uses the Google Custom Search JSON API to retrieve product-specific URLs.
- **Dynamic Site-Scoping:** Automatically detects brand or retailer site prefixes in queries (e.g. `Columbia men's fleece jackets` translates to `site:columbiasportswear.co.in ...`).
- **Playwright Scraping:** Navigates JS-heavy product pages concurrently, bypassing standard rendering delays.
- **Three-Layer Extraction Pipeline:**
  1. **JSON-LD (schema.org/Product):** Robust parsing of structured microdata.
  2. **Embedded JS State Parser:** Regex & recursive search for inline state variables (e.g. Next.js `__NEXT_DATA__` or Redux `__INITIAL_STATE__`).
  3. **BeautifulSoup Fallback:** Class/attribute heuristic extractor for title, price, description, images, color, and sizes.
- **Resilient Processing:** Isolates failures; individual page failures or parsing issues do not compromise the rest of the search results.

---

## Architecture

```text
google_product_search
        │
        ▼
   Google API Client (Google Custom Search)
        │
        ▼
   Search Result Parser (Filters out categories, blogs, etc.)
        │
        ▼
   Product URL Extractor (Caps to limit, max 5)
        │
        ▼
   Product Page Scraper (Playwright)
        │
        ├── JSON-LD Parser
        ├── Embedded JSON Parser
        └── HTML Fallback Parser
        │
        ▼
   Product Normalizer (Pydantic Mapping)
        │
        ▼
   Structured JSON Response
```

---

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure Environment
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CX=your_search_engine_id_here
SEARCH_COUNTRY=in
SEARCH_LANGUAGE=en
PLAYWRIGHT_TIMEOUT_MS=15000
PLAYWRIGHT_HEADLESS=true
```

---

## Usage

### 1. Python Tool Import
```python
from tool.google_product_search import google_product_search

# Run the search
results = google_product_search("Columbia men's red fleece jackets", limit=3)
print(results)
```

### 2. Command Line Interface (CLI)
To run a query directly from the terminal:
```bash
python src/main.py --query "Columbia men's red fleece jackets" --limit 3
```

### 3. FastAPI HTTP Server
Run the HTTP server:
```bash
python src/main.py --serve
```
Send a request using curl:
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "red fleece jackets for men", "limit": 3}'
```

---

## Schema Reference

### OpenAI Tool Schema
AI agents should call the tool using the following schema:
```json
{
  "name": "google_product_search",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural-language product search query. Include relevant product requirements such as brand, product type, color, gender, activity, material, price range, or other attributes."
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of products to return.",
        "minimum": 1,
        "maximum": 5,
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

### Response Schema
```json
{
  "query": "Columbia men's red fleece jackets",
  "count": 3,
  "products": [
    {
      "name": "Columbia Men's Red Backbowl II Full Zip Fleece",
      "brand": "Columbia",
      "price": 4499.0,
      "currency": "INR",
      "original_price": 4999.0,
      "discount": 10.0,
      "description": "Warm fleece jacket suitable for cold weather.",
      "color": "Red",
      "sizes": ["S", "M", "L", "XL"],
      "rating": 4.5,
      "review_count": 22,
      "availability": "In Stock",
      "image_url": "https://example.com/image.jpg",
      "product_url": "https://example.com/product/1"
    }
  ],
  "error": null
}
```

---

## Running Tests
Run tests using pytest:
```bash
pytest
```
To run the live search tests (requires valid `GOOGLE_API_KEY` and `GOOGLE_CX` set in the environment):
```bash
GOOGLE_API_KEY=your_key GOOGLE_CX=your_cx pytest
```
