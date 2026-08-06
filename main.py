import os
import uuid
import json
import re
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from typing import AsyncGenerator, List, Optional
from fastapi import FastAPI, HTTPException, Body, Security, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import asyncpg
import redis.asyncio as redis_asyncio
import jwt as pyjwt
import httpx
import asyncio
import random

from search_engine import ProductSearchEngine
from search_engine_v2 import CatalogSearchEngineV2
import otp_auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("columbia_backend")

# Global search engine instance (relational — Pinecone + Supabase)
engine = ProductSearchEngine()

# Flat catalog search engine (pure Pinecone, no DB)
catalog_engine_v2 = CatalogSearchEngineV2()

# Global Postgres (Supabase) connection pool, set during lifespan startup
db_pool: Optional[asyncpg.Pool] = None

# Global Redis client (OTP storage + JWT logout blocklist), set during lifespan startup
redis_client: Optional[redis_asyncio.Redis] = None

# JWT expiry in seconds, parsed once at import time from JWT_EXPIRATION (e.g. "24h")
JWT_EXPIRATION_SECONDS = otp_auth.parse_duration_to_seconds(os.getenv("JWT_EXPIRATION", "24h"))

# Auth Header Setup for User Authentication
security_bearer = HTTPBearer()

# API Key Authentication Setup for RAG / Agent Tool keys
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Hardcoded API key for now, with environment variable fallback
HARDCODED_API_KEY = "columbia-secret-api-key"

async def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("AGENT_API_KEY", HARDCODED_API_KEY)
    if api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid or missing API key."
        )
    return api_key

async def init_db(pool: asyncpg.Pool):
    logger.info("Initializing Postgres (Supabase) database tables...")
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT
        )
        """)

        # Migrate an existing username/password-based users table (from before the
        # OTP auth switch) and drop the now-unused opaque session token table.
        await conn.execute("ALTER TABLE users DROP COLUMN IF EXISTS username")
        await conn.execute("ALTER TABLE users DROP COLUMN IF EXISTS password_hash")
        await conn.execute("ALTER TABLE users ALTER COLUMN name DROP NOT NULL")
        await conn.execute("DROP TABLE IF EXISTS user_tokens")

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            session_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            sender TEXT NOT NULL CHECK (sender IN ('user', 'ai')),
            text TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)

        # Product catalog tables (previously populated by ingest_to_pinecone.py into SQLite)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS product_family (
            family_id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            category_level_1 TEXT,
            category_level_2 TEXT,
            product_type TEXT,
            sport TEXT,
            available_colors TEXT,
            available_sizes TEXT,
            available_materials TEXT,
            available_features TEXT,
            available_fits TEXT,
            price_min REAL,
            price_max REAL,
            primary_product_id TEXT,
            thumbnail_image TEXT,
            tags TEXT
        )
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS product_catalog (
            id SERIAL PRIMARY KEY,
            product_id TEXT,
            name TEXT,
            price REAL,
            category_level_1 TEXT,
            category_level_2 TEXT,
            product_type TEXT,
            sport TEXT,
            description TEXT,
            url TEXT,
            image_url TEXT,
            handle TEXT,
            tags TEXT,
            color TEXT,
            material TEXT,
            fit TEXT,
            features TEXT,
            size TEXT,
            stock_quantity INTEGER,
            family_id TEXT REFERENCES product_family(family_id)
        )
        """)

        await conn.execute("CREATE INDEX IF NOT EXISTS idx_catalog_product_id ON product_catalog(product_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_catalog_family ON product_catalog(family_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_catalog_size ON product_catalog(size)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_catalog_color ON product_catalog(color)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_catalog_price ON product_catalog(price)")

        # Cart & wishlist: one row per email, whole-list replace semantics (mirrors Coco-BE's Mongo docs)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS carts (
            email TEXT PRIMARY KEY,
            items JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at TIMESTAMPTZ
        )
        """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS wishlists (
            email TEXT PRIMARY KEY,
            items JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at TIMESTAMPTZ
        )
        """)

        # Seed default user if not exists (preserves user_id=1 continuity for existing chats)
        existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", "vansh@cogitx.ai")
        if not existing:
            await conn.execute(
                "INSERT INTO users (email, name) VALUES ($1, $2)",
                "vansh@cogitx.ai", "Vansh",
            )
    logger.info("Postgres tables ready.")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    token = credentials.credentials
    try:
        payload = otp_auth.decode_jwt(token)
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid session token.")

    if await redis_client.get(f"blocklist:{token}"):
        raise HTTPException(status_code=401, detail="Token has been invalidated.")

    row = await db_pool.fetchrow("SELECT id, email, name FROM users WHERE id = $1", int(payload["sub"]))
    if not row:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to Supabase Postgres + Redis, initialize tables, load search backends
    global db_pool, redis_client
    logger.info("Starting up: connecting to Supabase Postgres, Redis, and search backends...")
    dsn = os.getenv("SUPABASE_DB_URL")
    if not dsn:
        raise RuntimeError("SUPABASE_DB_URL is not set in the environment/.env file.")
    try:
        db_pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=10,
            # Supabase's pooler (pgbouncer, transaction mode) doesn't support
            # session-level prepared statements — disable asyncpg's statement cache.
            statement_cache_size=0,
        )
        app.state.db_pool = db_pool
        await init_db(db_pool)
        engine.pool = db_pool
        engine.load_data()
        catalog_engine_v2.load_data()

        redis_client = redis_asyncio.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD") or None,
            ssl=True,
            decode_responses=True,
        )
        await redis_client.ping()
        app.state.redis_client = redis_client
        logger.info("Redis client connected.")
    except Exception as e:
        logger.error("Error during startup initialization: %s", e)
        raise
    yield
    # Shutdown
    logger.info("Shutting down backend...")
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.aclose()

app = FastAPI(
    title="Columbia Inventory Agent Toolset & RAG API",
    description=(
        "An OpenAPI-compliant backend providing a single unified search and retrieval tool "
        "for AI agents to query the Columbia Sportswear inventory. Integrates semantic vector search "
        "(Pinecone) with relational inventory and variant mapping (Supabase Postgres)."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for agent integration in web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth & Chat Schemas
class SendOtpRequest(BaseModel):
    email: str

class SendOtpResponse(BaseModel):
    success: bool
    message: str
    email: str
    otp: Optional[str] = None  # only populated when ENVIRONMENT=development

class VerifyOtpRequest(BaseModel):
    email: str
    otp: str = Field(..., min_length=6, max_length=6)

class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str] = None

class VerifyOtpResponse(BaseModel):
    success: bool
    message: str
    token: str
    user: UserResponse

class LogoutResponse(BaseModel):
    success: bool
    message: str

class ChatResponseModel(BaseModel):
    id: int
    session_id: str
    title: str
    created_at: str

class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    content_type: Optional[str] = None
    structured_data: Optional[list] = None
    session_id: Optional[str] = None
    image_url: Optional[str] = None
    created_at: str

class ChatHistoryResponse(BaseModel):
    email: str
    messages: List[ChatHistoryMessage]

class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Existing chat's session_id. Omit to start a new chat.")
    text: str

# --- Cart & Wishlist Schemas ---

class CartItem(BaseModel):
    id: str
    name: str = Field(max_length=500)
    price: float
    originalPrice: float
    discount: float
    description: str = Field(max_length=5000)
    image: str = Field(max_length=2000)
    quantity: int

class CartSaveRequest(BaseModel):
    email: str
    items: List[CartItem] = Field(max_length=500)

class CartResponse(BaseModel):
    email: str
    items: List[CartItem]
    updated_at: Optional[datetime] = None

class WishlistItem(BaseModel):
    id: str
    name: str = Field(max_length=500)
    price: float
    originalPrice: float
    discount: float
    description: str = Field(max_length=5000)
    image: str = Field(max_length=2000)
    quantity: int = 0

class WishlistSaveRequest(BaseModel):
    email: str
    items: List[WishlistItem] = Field(max_length=500)

class WishlistResponse(BaseModel):
    email: str
    items: List[WishlistItem]
    updated_at: Optional[datetime] = None

# --- Pydantic Schema Definitions ---

class SearchRequest(BaseModel):
    query: Optional[str] = Field(
        None,
        description=(
            "A natural language search query. E.g., 'blue windbreaker jacket', 'waterproof hiking pants', "
            "'warm fleece'. If provided, results are ranked by semantic relevance to this query. "
            "If omitted, products are returned based on structured filters alone."
        )
    )
    family_id: Optional[str] = Field(
        None,
        description="A specific product model family ID (e.g. 'F000001'). If supplied, performs a direct lookup for that product family and its variants."
    )
    category_level_1: Optional[str] = Field(
        None,
        description="The primary department or category level (e.g., 'Apparel', 'Footwear', 'Equipment')."
    )
    category_level_2: Optional[str] = Field(
        None,
        description="The secondary category, department, or demographic (e.g., 'Men\'s', 'Women\'s', 'Kids', 'Unisex')."
    )
    product_type: Optional[str] = Field(
        None,
        description="The specific product type classification (e.g., 'Jackets', 'Pants', 'Shirts', 'Shoes')."
    )
    sport: Optional[str] = Field(
        None,
        description="The specific sport or activity intended for this item (e.g., 'Hiking', 'Casual', 'Running', 'Training')."
    )
    price_min: Optional[float] = Field(
        None,
        description="Filter products with price greater than or equal to this value."
    )
    price_max: Optional[float] = Field(
        None,
        description="Filter products with price less than or equal to this value."
    )
    color: Optional[str] = Field(
        None,
        description="Exact color filter (e.g., 'Maroon', 'Black', 'Grey')."
    )
    size: Optional[str] = Field(
        None,
        description="Exact size filter (e.g., 'S', 'M', 'L', 'XL', 'XXS')."
    )
    in_stock_only: bool = Field(
        True,
        description="If True, only returns items that are currently in stock and available for purchase."
    )
    top_k: int = Field(
        10,
        description="The maximum number of matched products to return."
    )

class VariantResult(BaseModel):
    product_id: str = Field(..., description="Unique variant product ID.")
    name: str = Field(..., description="The name of this variant.")
    price: float = Field(..., description="The price of this variant.")
    size: str = Field(..., description="The size of this variant.")
    color: str = Field(..., description="The color of this variant.")
    stock_quantity: int = Field(..., description="Stock count remaining.")
    availability: str = Field(..., description="Stock availability ('in_stock' or 'out_of_stock').")
    url: str = Field(..., description="Product detail URL.")
    image_url: str = Field(..., description="Product variant image URL.")
    handle: str = Field(..., description="Product variant URL handle.")
    tags: str = Field(..., description="Tags associated with this variant.")

class ProductFamilyResult(BaseModel):
    family_id: str = Field(..., description="Unique product model family identifier.")
    title: str = Field(..., description="General title of the product model.")
    description: str = Field(..., description="General description of the product model.")
    category_level_1: str = Field(..., description="Primary department/category.")
    category_level_2: str = Field(..., description="Secondary category/demographic.")
    product_type: str = Field(..., description="Product type (e.g. Jackets).")
    sport: str = Field(..., description="Sport category.")
    available_colors: str = Field(..., description="JSON array representation of available colors.")
    available_sizes: str = Field(..., description="JSON array representation of available sizes.")
    available_materials: str = Field(..., description="JSON array representation of materials.")
    available_features: str = Field(..., description="JSON array representation of features.")
    available_fits: str = Field(..., description="JSON array representation of fits.")
    price_min: float = Field(..., description="Minimum price of variants under this family.")
    price_max: float = Field(..., description="Maximum price of variants under this family.")
    primary_product_id: str = Field(..., description="Main product ID used as the family reference.")
    thumbnail_image: str = Field(..., description="URL of the main/primary product image.")
    tags: str = Field(..., description="Tags associated with this product family.")
    variants: List[VariantResult] = Field([], description="List of matching inventory variants (sizes/colors) under this family.")
    score: float = Field(..., description="Relevance score (1.0 = exact or default match, < 1.0 = semantic similarity score).")

class CatalogFacetsResponse(BaseModel):
    category_level_1: List[str] = Field(..., description="List of unique primary categories.")
    category_level_2: List[str] = Field(..., description="List of unique secondary categories.")
    product_type: List[str] = Field(..., description="List of unique product types.")
    sport: List[str] = Field(..., description="List of unique sport categories.")
    brands: List[str] = Field(..., description="List of unique brands in the inventory.")
    price_min: float = Field(..., description="Minimum product price present in the catalog.")
    price_max: float = Field(..., description="Maximum product price present in the catalog.")

class CartesianChatRequest(BaseModel):
    text: str = Field(..., description="User input text")
    wait_seconds: Optional[int] = Field(30, description="Optional override for sync wait duration in seconds (max 30 for inline wait)")
    session_id: Optional[str] = Field(None, description="Optional conversation session identifier. Reuse this across calls to continue context.")
    user_id: Optional[str] = Field(None, description="Identifier for the requesting user, required by the workflow's input schema.")

# --- V2 Flat Catalog Schemas ---

class CatalogSearchRequestV2(BaseModel):
    query: str = Field(
        ...,
        description=(
            "A natural language search query describing what the user is looking for. "
            "E.g., 'waterproof hiking jacket in blue', 'kids fleece jacket', 'trekking poles'. "
            "This field is required — all retrieval is semantic."
        )
    )
    category_level_1: Optional[str] = Field(
        None,
        description="Primary department or category filter (e.g., 'Apparel', 'Accessories', 'Bags & Gear')."
    )
    category_level_2: Optional[str] = Field(
        None,
        description="Secondary category / demographic filter (e.g., \"Men's\", \"Women's\", 'Kids', 'Unisex')."
    )
    product_type: Optional[str] = Field(
        None,
        description="Product type filter (e.g., 'Sportswear', 'Equipment', 'Accessories')."
    )
    sport: Optional[str] = Field(
        None,
        description="Sport or activity filter (e.g., 'Hiking', 'Casual', 'Running')."
    )
    price_min: Optional[float] = Field(
        None,
        description="Minimum price filter (inclusive). Products priced below this value are excluded."
    )
    price_max: Optional[float] = Field(
        None,
        description="Maximum price filter (inclusive). Products priced above this value are excluded."
    )
    color: Optional[str] = Field(
        None,
        description="Exact color filter (e.g., 'Black', 'Blue', 'Maroon')."
    )
    size: Optional[str] = Field(
        None,
        description="Exact size filter (e.g., 'S', 'M', 'L', 'XL', 'XXS', 'OS')."
    )
    top_k: int = Field(
        10,
        description="Maximum number of products to return (default 10)."
    )

class CatalogProductResultV2(BaseModel):
    product_id: str = Field(..., description="Unique SKU / product ID.")
    name: str = Field(..., description="Full product name.")
    price: float = Field(..., description="Product price.")
    category_level_1: str = Field(..., description="Primary department/category.")
    category_level_2: str = Field(..., description="Secondary category/demographic.")
    product_type: str = Field(..., description="Product type classification.")
    sport: str = Field(..., description="Sport or activity.")
    color: str = Field(..., description="Product color.")
    material: str = Field(..., description="Product material.")
    fit: str = Field(..., description="Fit style (e.g., Regular, Slim, Comfort Stretch).")
    features: str = Field(..., description="Key product features.")
    size: str = Field(..., description="Available size for this SKU.")
    stock_quantity: int = Field(..., description="Units currently in stock.")
    availability: str = Field(..., description="'in_stock' or 'out_of_stock'.")
    url: str = Field(..., description="Product detail page URL.")
    image_url: str = Field(..., description="Product image URL.")
    description_snippet: str = Field(..., description="Short description excerpt.")
    score: float = Field(..., description="Semantic similarity score (cosine, 0–1).")

# --- API Tool Endpoints ---

@app.post(
    "/tools/search-products",
    response_model=List[ProductFamilyResult],
    dependencies=[Depends(verify_api_key)],
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
        results = await engine.search(
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


@app.post(
    "/tools/search-products-v2",
    response_model=List[CatalogProductResultV2],
    dependencies=[Depends(verify_api_key)],
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
        results = await catalog_engine_v2.search(
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


@app.get(
    "/tools/catalog-facets",
    response_model=CatalogFacetsResponse,
    dependencies=[Depends(verify_api_key)],
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
        facets = await engine.get_facets()
        logger.info("TOOL catalog-facets: response=%s", facets)
        return facets
    except Exception as e:
        logger.error("TOOL catalog-facets: failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve facets: {str(e)}")

@app.post(
    "/chat",
    tags=["workflow testing"],
    summary="Chat endpoint integrating with Cartesian Columbia workflow",
    description="Send a message to the Cartesian Columbia workflow and get the response."
)
async def chat_endpoint(request: CartesianChatRequest = Body(...)):
    base_url = os.getenv("CARTESIAN_BASE_URL", "https://api.cartesian.ai") # Change this if your base URL is different
    client_id = os.getenv("CARTESIAN_CLIENT_ID")
    client_secret = os.getenv("CARTESIAN_CLIENT_SECRET")

    if not client_id or not client_secret:
         raise HTTPException(status_code=500, detail="Cartesian API credentials (CARTESIAN_CLIENT_ID, CARTESIAN_CLIENT_SECRET) are missing in environment variables.")

    headers = {
        "x-client-id": client_id,
        "x-client-secret": client_secret,
        "Content-Type": "application/json"
    }

    payload = {
        "payload": {
            "user_query": request.text,
            "user_id": request.user_id or "test-user"
        }
    }

    params = {}
    if request.wait_seconds is not None:
        params["waitSeconds"] = request.wait_seconds
    if request.session_id:
        params["sessionId"] = request.session_id

    endpoint_path = "/exports/rest-api/6a59f1b8285cc674dbd79b87/jobs"
    url = f"{base_url.rstrip('/')}{endpoint_path}"

    async with httpx.AsyncClient() as client:
        try:
            # We set a timeout slightly larger than wait_seconds if provided
            timeout = request.wait_seconds + 10 if request.wait_seconds else 40
            response = await client.post(url, json=payload, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Failed to communicate with Cartesian API: {str(e)}")

        # Cartesian API may wrap the response in a "data" object
        resp_data = data.get("data", data)

        if resp_data.get("ok") and resp_data.get("accepted"):
            # Async processing, need to poll
            run_id = resp_data.get("runId")
            if not run_id:
                raise HTTPException(status_code=500, detail="Received accepted response but no runId")

            poll_url = f"{base_url.rstrip('/')}/exports/rest-api/6a59f1b8285cc674dbd79b87/jobs/{run_id}"

            # Poll until complete or timeout (e.g. 30 times with 2s delay = ~60s wait)
            max_retries = 30
            for _ in range(max_retries):
                await asyncio.sleep(2)
                try:
                    poll_resp = await client.get(poll_url, headers=headers, timeout=10)
                    poll_resp.raise_for_status()
                    poll_data_raw = poll_resp.json()
                    poll_data = poll_data_raw.get("data", poll_data_raw)
                except httpx.HTTPError as e:
                    raise HTTPException(status_code=502, detail=f"Failed to poll Cartesian API: {str(e)}")

                if poll_data.get("isCompleted"):
                    return poll_data

            raise HTTPException(status_code=504, detail="Timeout waiting for Cartesian workflow to complete")

        elif resp_data.get("ok") and not resp_data.get("accepted"):
            # Completed within wait window
            return resp_data

        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response from Cartesian API: {data}")


# --- User Authentication Endpoints (email OTP) ---

@app.post(
    "/api/auth/send-otp",
    response_model=SendOtpResponse,
    tags=["auth"],
    summary="Send OTP to email",
    description="Generates a 6-digit OTP, stores it in Redis for 5 minutes, and emails it via Azure Communication Email."
)
async def send_otp(request: SendOtpRequest = Body(...)):
    email = request.email.strip().lower()
    if not otp_auth.is_valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    otp = otp_auth.generate_otp()
    await redis_client.setex(f"otp:{email}", otp_auth.OTP_TTL_SECONDS, otp)

    try:
        await asyncio.to_thread(otp_auth.send_otp_email, otp, email)
    except Exception as e:
        logger.error("send_otp: failed to send email to %s: %s", email, e)
        raise HTTPException(status_code=502, detail="Failed to send OTP email.")

    logger.info("send_otp: OTP sent to %s", email)
    response = {"success": True, "message": "OTP sent successfully", "email": email}
    if os.getenv("ENVIRONMENT", "").lower() == "development":
        response["otp"] = otp
    return response


@app.post(
    "/api/auth/verify-otp",
    response_model=VerifyOtpResponse,
    tags=["auth"],
    summary="Verify OTP and receive JWT token",
    description="Verifies the OTP, creates the user on first login if needed, and returns a signed JWT."
)
async def verify_otp(request: VerifyOtpRequest = Body(...)):
    email = request.email.strip().lower()
    stored_otp = await redis_client.get(f"otp:{email}")
    if not stored_otp or stored_otp != request.otp:
        raise HTTPException(status_code=401, detail="Invalid or expired OTP")
    await redis_client.delete(f"otp:{email}")

    user_row = await db_pool.fetchrow("SELECT id, email, name FROM users WHERE email = $1", email)
    if not user_row:
        default_name = email.split("@")[0]
        user_row = await db_pool.fetchrow(
            "INSERT INTO users (email, name) VALUES ($1, $2) RETURNING id, email, name",
            email, default_name,
        )
        logger.info("verify_otp: created new user for %s (id=%s)", email, user_row["id"])

    token = otp_auth.create_jwt(user_row["id"], user_row["email"], JWT_EXPIRATION_SECONDS)
    logger.info("verify_otp: OTP verified for %s (id=%s)", email, user_row["id"])

    return {
        "success": True,
        "message": "OTP verified successfully",
        "token": token,
        "user": {"id": user_row["id"], "email": user_row["email"], "name": user_row["name"]},
    }


@app.get(
    "/api/auth/me",
    response_model=UserResponse,
    tags=["auth"],
    summary="Get Current User Profile",
    description="Returns profile of the authenticated user."
)
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user


@app.post(
    "/api/auth/logout",
    response_model=LogoutResponse,
    tags=["auth"],
    summary="Logout and invalidate JWT token",
    description="Blocklists the current JWT in Redis until its natural expiry."
)
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    token = credentials.credentials
    try:
        payload = otp_auth.decode_jwt(token)
    except pyjwt.PyJWTError:
        return {"success": True, "message": "Logged out successfully"}

    ttl = payload["exp"] - int(datetime.now(timezone.utc).timestamp())
    if ttl > 0:
        await redis_client.setex(f"blocklist:{token}", ttl, "1")
    return {"success": True, "message": "Logged out successfully"}


# --- Chat & Session Management Endpoints ---


@app.get(
    "/api/chats",
    response_model=List[ChatResponseModel],
    tags=["chat"],
    summary="Get all chats",
    description="Lists all active chat sessions for the logged-in user."
)
async def get_chats(current_user: dict = Depends(get_current_user)):
    try:
        rows = await db_pool.fetch(
            "SELECT id, session_id, title, created_at FROM chats WHERE user_id = $1 ORDER BY created_at DESC",
            current_user["id"],
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return [
        {
            "id": r["id"],
            "session_id": r["session_id"],
            "title": r["title"],
            "created_at": _to_iso(r["created_at"]),
        }
        for r in rows
    ]


@app.delete(
    "/api/chats/{chat_id}",
    tags=["chat"],
    summary="Delete a chat session",
    description="Deletes a chat thread and all its corresponding message history."
)
async def delete_chat(chat_id: int, current_user: dict = Depends(get_current_user)):
    owner_row = await db_pool.fetchrow(
        "SELECT id FROM chats WHERE id = $1 AND user_id = $2", chat_id, current_user["id"]
    )
    if not owner_row:
        raise HTTPException(status_code=404, detail="Chat not found or access denied.")

    try:
        await db_pool.execute("DELETE FROM chats WHERE id = $1", chat_id)
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error deleting chat: {str(e)}")

    return {"detail": "Chat deleted successfully"}


# --- Messaging History & Dispatch Endpoints ---

@app.get(
    "/api/chats/history",
    response_model=ChatHistoryResponse,
    tags=["chat"],
    summary="Get chat history",
    description=(
        "Retrieves message history for a user by email. Pass session_id to scope to one "
        "conversation; omit it to get every message across all of that user's chats."
    )
)
async def get_chat_history(
    email: str,
    session_id: Optional[str] = None,
    limit: int = Query(default=200, le=500),
):
    norm_email = email.strip().lower()
    try:
        user_row = await db_pool.fetchrow("SELECT id FROM users WHERE email = $1", norm_email)
        if not user_row:
            return ChatHistoryResponse(email=norm_email, messages=[])
        user_id = user_row["id"]

        if session_id:
            chat_rows = await db_pool.fetch(
                "SELECT id, session_id FROM chats WHERE user_id = $1 AND session_id = $2",
                user_id, session_id,
            )
        else:
            chat_rows = await db_pool.fetch(
                "SELECT id, session_id FROM chats WHERE user_id = $1", user_id
            )
        if not chat_rows:
            return ChatHistoryResponse(email=norm_email, messages=[])

        session_by_chat = {r["id"]: r["session_id"] for r in chat_rows}
        chat_ids = list(session_by_chat.keys())
        # Cap per chat_id (most recent `limit` messages each), not with one
        # limit shared across every chat — otherwise one long-running chat
        # can starve the others out of the response entirely.
        rows = await db_pool.fetch(
            """
            SELECT chat_id, sender, text, created_at FROM (
                SELECT chat_id, sender, text, created_at,
                       ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY created_at DESC) AS rn
                FROM messages
                WHERE chat_id = ANY($1::int[])
            ) ranked
            WHERE rn <= $2
            ORDER BY created_at ASC
            """,
            chat_ids, limit,
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

    messages = []
    for chat_id, sender, text, created_at in rows:
        sid = session_by_chat.get(chat_id)
        if sender == "user":
            messages.append(ChatHistoryMessage(
                role="user", content=text, session_id=sid, created_at=_to_iso(created_at),
            ))
        else:
            head, groups, tail, obj = _parse_ai_response_parts(text)
            # Only groups that actually have products get a marker + widget —
            # matches what live streaming does (skips empty groups too).
            non_empty_groups = [items for _ptype, items in groups if items]
            has_products = bool(non_empty_groups)
            content_type = None
            structured_data = None
            if has_products:
                content_type = (obj.get("intent") if obj else None) or "product_discovery"
                markers = [_product_widget_marker(i) for i in range(1, len(non_empty_groups) + 1)]
                content = "\n\n".join(head + markers + tail)
                structured_data = non_empty_groups
            else:
                content = "\n\n".join(head + tail) or text
            messages.append(ChatHistoryMessage(
                role="assistant",
                content=content,
                content_type=content_type,
                structured_data=structured_data,
                session_id=sid,
                created_at=_to_iso(created_at),
            ))
    logger.info("chat_history: email=%s session_id=%s returned %d messages", norm_email, session_id, len(messages))
    return ChatHistoryResponse(email=norm_email, messages=messages)


class SessionProductsResponse(BaseModel):
    session_id: str
    products: List[dict]


@app.get(
    "/api/chats/session-products",
    response_model=SessionProductsResponse,
    tags=["chat"],
    summary="Get all products shown in a chat session",
    description=(
        "Returns every product that has been shown (via product_discovery) anywhere in this "
        "chat session, deduped by product_id — powers the frontend's '@product' popup. Backed by "
        "a per-session Redis hash cache; on a cache miss it rebuilds from the chat's saved message "
        "history in Postgres (the real source of truth) and repopulates the cache before returning."
    )
)
async def get_session_products(session_id: str, current_user: dict = Depends(get_current_user)):
    chat_row = await db_pool.fetchrow(
        "SELECT id FROM chats WHERE session_id = $1 AND user_id = $2",
        session_id, current_user["id"],
    )
    if not chat_row:
        raise HTTPException(status_code=404, detail="Chat not found or access denied.")

    key = _session_products_key(session_id)
    cached = await redis_client.hgetall(key)
    if cached:
        products = [json.loads(v) for v in cached.values()]
        logger.info("session-products: session_id=%s cache hit, %d products", session_id, len(products))
        return SessionProductsResponse(session_id=session_id, products=products)

    products = await _rebuild_session_products_from_db(chat_row["id"])
    if products:
        await _cache_session_products(session_id, products)
    logger.info("session-products: session_id=%s cache miss, rebuilt %d products from DB", session_id, len(products))
    return SessionProductsResponse(session_id=session_id, products=products)


CARTESIAN_JOB_PATH = "/exports/rest-api/6a59f1b8285cc674dbd79b87/jobs"
STATUS_MESSAGES = [
    "Thinking...",
    "Searching the catalog...",
    "Fetching product details...",
    "Looking for deals...",
]
def _product_widget_marker(index: int) -> str:
    """Position marker for the Nth product group in a history message's
    text, so the frontend can splice each group's widget into the exact
    spot it belongs (matching structured_data[index - 1]) instead of only
    ever bundling every group into one widget up front."""
    return f":::PRODUCT_WIDGET_{index}:::"


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_NON_NARRATIVE_KEYS = {"products", "intent", "status", "contentType"}


def _extract_ai_text(output_obj: dict) -> Optional[str]:
    """variables.text is the clean, flat JSON string matching the intent's
    schema. workflow_response.content is sometimes a nested dict (e.g.
    {"result": {...fields double-encoded...}}) rather than a string, which
    breaks downstream str-only consumers (DB insert, _parse_ai_response).
    Prefer text; only fall back to content if it's actually a string."""
    text = output_obj.get("variables", {}).get("text")
    if isinstance(text, str):
        return text
    content = output_obj.get("workflow_response", {}).get("content")
    if isinstance(content, str):
        return content
    return None


async def _call_cartesian_workflow(client: httpx.AsyncClient, url: str, poll_url_base: str, params: dict, payload: dict, headers: dict) -> Optional[str]:
    """Runs the full Cartesian call (initial POST, then the poll loop if the
    job was accepted for async processing). Returns the extracted ai_text,
    or None if nothing usable came back. Runs as a background task so the
    caller can tick status messages independently of this function's cadence."""
    response = await client.post(url, json=payload, params=params, headers=headers, timeout=40)
    response.raise_for_status()
    data = response.json()
    logger.info("cartesian: initial response status=%s body=%s", response.status_code, data)
    resp_data = data.get("data", data)

    if resp_data.get("ok") and resp_data.get("accepted"):
        run_id = resp_data.get("runId")
        poll_url = f"{poll_url_base}/{run_id}"
        logger.info("cartesian: job accepted async, runId=%s, polling %s", run_id, poll_url)
        max_retries = 30
        for attempt in range(max_retries):
            await asyncio.sleep(2)
            poll_resp = await client.get(poll_url, headers=headers, timeout=10)
            poll_resp.raise_for_status()
            poll_data_raw = poll_resp.json()
            poll_data = poll_data_raw.get("data", poll_data_raw)
            logger.info(
                "cartesian: poll attempt %d/%d isCompleted=%s body=%s",
                attempt + 1, max_retries, poll_data.get("isCompleted"), poll_data_raw,
            )
            if poll_data.get("isCompleted"):
                return _extract_ai_text(poll_data.get("output", {}))
        logger.error("cartesian: timed out after %d poll attempts", max_retries)
        return None

    if resp_data.get("ok") and not resp_data.get("accepted"):
        logger.info("cartesian: completed synchronously within wait window")
        return _extract_ai_text(resp_data.get("output", {}))

    logger.error("cartesian: no ai_text extractable from response: %s", data)
    return None


def _to_iso(ts: datetime) -> str:
    """Postgres TIMESTAMPTZ comes back from asyncpg as a tz-aware datetime.
    Normalize to UTC and drop the offset for a plain ISO 8601 string."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts.isoformat()


_PRODUCT_ARRAY_STRING_FIELDS = (
    "available_colors", "available_sizes", "available_materials",
    "available_features", "available_fits", "tags",
)


def _coerce_products_list(structured: dict) -> None:
    """Cartesian sometimes double-encodes 'products' as a JSON string instead
    of a real array. Parse it in place so _flatten_products/_normalize_product_arrays
    (which both require an actual list) don't silently no-op on it."""
    products = structured.get("products")
    if isinstance(products, str):
        try:
            parsed = json.loads(products)
        except json.JSONDecodeError:
            return
        if isinstance(parsed, list):
            structured["products"] = parsed


def _flatten_products(structured: dict) -> None:
    """Every intent except 'budget' sends products as a flat list of product
    dicts. 'budget' wraps them one level deeper as [{query, items:[...]}, ...].
    Flatten that here so the frontend's product grid renderer never has to
    branch on intent."""
    products = structured.get("products")
    if not isinstance(products, list) or not products:
        return
    if isinstance(products[0], dict) and "items" in products[0]:
        flat = []
        for group in products:
            if isinstance(group, dict) and isinstance(group.get("items"), list):
                flat.extend(item for item in group["items"] if isinstance(item, dict))
        structured["products"] = flat


def _normalize_product_arrays(structured: dict) -> None:
    """Cartesian sends list-shaped fields (colors, sizes, tags, ...) as a
    stringified JSON array (e.g. '["Black", "Blue"]') instead of a real array.
    Parse those in place so the frontend gets actual arrays."""
    for product in structured.get("products") or []:
        if not isinstance(product, dict):
            continue
        for field in _PRODUCT_ARRAY_STRING_FIELDS:
            value = product.get(field)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list):
                    product[field] = parsed


def _extract_structured_obj(ai_text: str) -> Optional[dict]:
    """Parses ai_text (optionally ```json-fenced) into the structured dict
    Cartesian sends for product replies ({"introduction","orientation",
    "opening","status","products","recovery","closing"}), normalizing the
    products field in place. Returns None for plain prose or anything that
    isn't a JSON object."""
    candidate = ai_text.strip()
    fence_match = _CODE_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    _coerce_products_list(obj)
    _flatten_products(obj)
    _normalize_product_arrays(obj)
    return obj


def _iter_product_groups(products) -> list[tuple[Optional[str], list]]:
    """Yields (product_type, flat_product_list) pairs so the caller can
    stream each group as its own product_discovery event with a plain
    (non-nested) product array. Handles both the grouped shape
    ([{"product_type", "products": [...]}, ...]) and an already-flat list
    of product dicts (e.g. budget intent, post-_flatten_products)."""
    if not isinstance(products, list) or not products:
        return []
    if isinstance(products[0], dict) and isinstance(products[0].get("products"), list):
        groups = []
        for group in products:
            if not isinstance(group, dict):
                continue
            ptype = group.get("product_type")
            items = [p for p in group.get("products", []) if isinstance(p, dict)]
            if ptype:
                for item in items:
                    item.setdefault("product_type", ptype)
            groups.append((ptype, items))
        return groups
    return [(None, [p for p in products if isinstance(p, dict)])]


def _parse_ai_response_parts(ai_text: str) -> tuple[list[str], list[tuple[Optional[str], list]], list[str], Optional[dict]]:
    """Streaming variant of _parse_ai_response: instead of hardcoding which
    field names count as narrative text (introduction/orientation/opening/
    recovery/closing), walks the object's own keys in order and treats any
    non-empty string value as a message — so a new or renamed narrative
    field from Cartesian is picked up automatically instead of silently
    dropped. 'products' is pulled out separately as product groups;
    'status'/'intent'/'contentType' are metadata, not display text.
    Returns (head, groups, tail, obj): head = narrative strings that
    appeared before 'products' in the object, tail = the ones that appeared
    after, obj = the raw parsed object (None for plain-prose responses) so
    callers can still pull metadata like 'intent' off it if needed."""
    obj = _extract_structured_obj(ai_text)
    if obj is None:
        return [ai_text], [], [], None
    head: list[str] = []
    tail: list[str] = []
    seen_products = False
    for key, value in obj.items():
        if key == "products":
            seen_products = True
            continue
        if key in _NON_NARRATIVE_KEYS:
            continue
        if isinstance(value, str) and value:
            (tail if seen_products else head).append(value)
    if not head and not tail:
        status = obj.get("status")
        if isinstance(status, dict) and status.get("message"):
            head = [status["message"]]
        else:
            head = [ai_text]
    groups = _iter_product_groups(obj.get("products"))
    return head, groups, tail, obj


SESSION_PRODUCTS_TTL_SECONDS = 86400  # cache hygiene only — Postgres (messages table) is the real source of truth


def _session_products_key(session_id: str) -> str:
    return f"session_products:{session_id}"


async def _cache_session_products(session_id: str, items: list[dict]) -> None:
    """Pushes newly-shown products into this session's Redis hash, keyed by
    product_id so re-showing the same product just overwrites its field
    instead of creating a duplicate. Refreshes the TTL on every push so an
    actively-used session's cache doesn't go cold mid-conversation."""
    mapping = {p["product_id"]: json.dumps(p) for p in items if p.get("product_id")}
    if not mapping:
        return
    key = _session_products_key(session_id)
    await redis_client.hset(key, mapping=mapping)
    await redis_client.expire(key, SESSION_PRODUCTS_TTL_SECONDS)


async def _rebuild_session_products_from_db(chat_id: int) -> list[dict]:
    """Cache-miss fallback: replays every AI message ever saved for this
    chat, re-parsing each one the same way live streaming does, and
    flattens/dedupes every product group across the whole conversation.
    Postgres already holds this permanently (messages.text) — Redis is only
    ever a disposable speed layer in front of it."""
    rows = await db_pool.fetch(
        "SELECT text FROM messages WHERE chat_id = $1 AND sender = 'ai' ORDER BY created_at ASC",
        chat_id,
    )
    by_id: dict[str, dict] = {}
    for (text,) in rows:
        _head, groups, _tail, _obj = _parse_ai_response_parts(text)
        for _product_type, items in groups:
            for p in items:
                if p.get("product_id"):
                    by_id[p["product_id"]] = p
    return list(by_id.values())


async def _stream_chat_message(session_id: Optional[str], text: str, current_user: dict) -> AsyncGenerator[str, None]:
    logger.info(
        "chat_message: request received user=%s(%s) session_id=%s text=%r",
        current_user["email"], current_user["id"], session_id, text,
    )
    # 1. Resolve chat: create one if session_id omitted, else verify ownership
    if session_id is None:
        session_id = str(uuid.uuid4())
        title = "New Chat"
        row = await db_pool.fetchrow(
            "INSERT INTO chats (user_id, session_id, title) VALUES ($1, $2, $3) RETURNING id",
            current_user["id"], session_id, title,
        )
        chat_id = row["id"]
        logger.info("chat_message: created new chat_id=%s session_id=%s", chat_id, session_id)
    else:
        chat_row = await db_pool.fetchrow(
            "SELECT id, title FROM chats WHERE session_id = $1 AND user_id = $2",
            session_id, current_user["id"],
        )
        if not chat_row:
            logger.warning("chat_message: session_id=%s not found for user_id=%s", session_id, current_user["id"])
            yield f"event: error\ndata: {json.dumps({'detail': 'Chat not found or access denied.'})}\n\n"
            return
        chat_id, title = chat_row["id"], chat_row["title"]
        logger.info("chat_message: continuing chat_id=%s session_id=%s", chat_id, session_id)

    yield f"event: chat\ndata: {json.dumps({'v': {'id': chat_id, 'session_id': session_id, 'title': title}})}\n\n"

    # 2. Save user message
    user_msg_row = await db_pool.fetchrow(
        "INSERT INTO messages (chat_id, sender, text) VALUES ($1, $2, $3) RETURNING id",
        chat_id, 'user', text,
    )
    user_msg_id = user_msg_row["id"]

    # 3. Auto-title from first message
    if title == "New Chat":
        new_title = text[:40] + ("..." if len(text) > 40 else "")
        await db_pool.execute("UPDATE chats SET title = $1 WHERE id = $2", new_title, chat_id)

    # 4. Compile history + profile into the prompt sent to Cartesian
    history_rows = await db_pool.fetch(
        "SELECT sender, text FROM messages WHERE chat_id = $1 AND id < $2 ORDER BY created_at ASC",
        chat_id, user_msg_id,
    )
    profile_header = f"--- User Profile ---\n- Name: {current_user['name']}\n\n"
    if history_rows:
        history_str = "\n".join(
            f"[{'User' if sender == 'user' else 'AI'}]: {msg}" for sender, msg in history_rows
        )
        history_block = f"--- Chat History ---\n{history_str}\n\n"
    else:
        history_block = ""
    compiled_text = (
        "System Instruction: Below is the context of the logged-in user and the chat history of this conversation. "
        "Please use this information to answer the user's latest query at the end. "
        "Respond ONLY to the latest query and do not repeat the context or history.\n\n"
        f"{profile_header}{history_block}"
        f"--- Latest User Query ---\n[User]: {text}"
    )
    logger.info("chat_message: compiled prompt (%d chars): %r", len(compiled_text), compiled_text)

    # 5. Call Cartesian workflow, streaming status ticks while we wait
    base_url = os.getenv("CARTESIAN_BASE_URL", "https://api.cartesian.ai")
    client_id = os.getenv("CARTESIAN_CLIENT_ID")
    client_secret = os.getenv("CARTESIAN_CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.error("chat_message: Cartesian API credentials missing in environment")
        yield f"event: error\ndata: {json.dumps({'detail': 'Cartesian API credentials are missing.'})}\n\n"
        return

    headers = {
        "x-client-id": client_id,
        "x-client-secret": client_secret,
        "Content-Type": "application/json",
    }
    params = {"waitSeconds": 30, "sessionId": session_id}
    payload = {"payload": {"user_query": compiled_text, "user_id": current_user["email"]}}
    url = f"{base_url.rstrip('/')}{CARTESIAN_JOB_PATH}"

    ai_text: Optional[str] = None
    yield f"event: status\ndata: {json.dumps({'v': random.choice(STATUS_MESSAGES)})}\n\n"

    logger.info("cartesian: POST %s params=%s", url, params)

    async with httpx.AsyncClient() as client:
        poll_url_base = f"{base_url.rstrip('/')}{CARTESIAN_JOB_PATH}"
        task = asyncio.create_task(
            _call_cartesian_workflow(client, url, poll_url_base, params, payload, headers)
        )

        # Status ticks run on their own 3-5s cadence, fully decoupled from
        # whatever Cartesian is doing internally (sync wait or async polling)
        # so the frontend always sees continuous progress regardless of path.
        shuffled = random.sample(STATUS_MESSAGES, len(STATUS_MESSAGES))
        tick_idx = 0
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=random.uniform(3, 5))
                if task in done:
                    break
                yield f"event: status\ndata: {json.dumps({'v': shuffled[tick_idx % len(shuffled)]})}\n\n"
                tick_idx += 1
                if tick_idx % len(shuffled) == 0:
                    shuffled = random.sample(STATUS_MESSAGES, len(STATUS_MESSAGES))

            ai_text = await task
        except httpx.HTTPError as e:
            logger.error("cartesian: HTTP error communicating with Cartesian: %s", e)
            yield f"event: error\ndata: {json.dumps({'detail': f'AI service communication error: {str(e)}'})}\n\n"
            return

    if ai_text is None:
        logger.error("cartesian: no ai_text extractable from response")
        yield f"event: error\ndata: {json.dumps({'detail': 'Unexpected response from the AI service.'})}\n\n"
        return

    logger.info("cartesian: extracted ai_text (%d chars): %r", len(ai_text), ai_text)

    # 6. Save AI response, then stream it out
    await db_pool.execute(
        "INSERT INTO messages (chat_id, sender, text) VALUES ($1, $2, $3)",
        chat_id, 'ai', ai_text,
    )

    head, groups, tail, _obj = _parse_ai_response_parts(ai_text)
    product_count = sum(len(items) for _, items in groups)
    logger.info(
        "chat_message: parsed response head=%d groups=%d product_count=%d tail=%d",
        len(head), len(groups), product_count, len(tail),
    )
    for msg in head:
        yield f"event: message\ndata: {json.dumps({'v': msg})}\n\n"
    for _product_type, items in groups:
        if items:
            yield f"event: product_discovery\ndata: {json.dumps({'v': items})}\n\n"
            await _cache_session_products(session_id, items)
    for msg in tail:
        yield f"event: message\ndata: {json.dumps({'v': msg})}\n\n"

    yield f"event: conversation_id\ndata: {json.dumps({'v': session_id})}\n\n"
    yield f"event: end\ndata: {json.dumps({'v': {}})}\n\n"
    logger.info("chat_message: stream complete chat_id=%s session_id=%s", chat_id, session_id)


@app.post(
    "/api/chats/message",
    tags=["chat"],
    summary="Create/continue a chat and send a message (SSE stream)",
    description=(
        "Combines chat creation and message sending into a single streamed endpoint. "
        "Omit session_id to start a new chat; pass an existing session_id to continue it. "
        "Streams event: chat, status, message, product_discovery (when present), conversation_id, end."
    )
)
async def post_chat_message(request: ChatMessageRequest = Body(...), current_user: dict = Depends(get_current_user)):
    return StreamingResponse(
        _stream_chat_message(request.session_id, request.text, current_user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Cart & Wishlist Endpoints (whole-list replace, keyed by email) ---

async def _get_saved_items(table: str, email: str) -> dict:
    row = await db_pool.fetchrow(f"SELECT items, updated_at FROM {table} WHERE email = $1", email)
    if not row:
        return {"email": email, "items": [], "updated_at": None}
    return {"email": email, "items": json.loads(row["items"]), "updated_at": row["updated_at"]}

async def _save_items(table: str, email: str, items: list) -> dict:
    await db_pool.execute(
        f"INSERT INTO {table} (email, items, updated_at) VALUES ($1, $2::jsonb, now()) "
        f"ON CONFLICT (email) DO UPDATE SET items = EXCLUDED.items, updated_at = EXCLUDED.updated_at",
        email, json.dumps(items),
    )
    return await _get_saved_items(table, email)

async def _remove_item(table: str, email: str, item_id: str) -> dict:
    current = await _get_saved_items(table, email)
    filtered = [item for item in current["items"] if item.get("id") != item_id]
    return await _save_items(table, email, filtered)


@app.get(
    "/api/cart",
    response_model=CartResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["cart"],
    summary="Get cart",
)
async def get_cart(email: str):
    return await _get_saved_items("carts", email.strip().lower())


@app.put(
    "/api/cart",
    response_model=CartResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["cart"],
    summary="Save cart",
    description="Replaces the entire cart with the given items list.",
)
async def save_cart(body: CartSaveRequest = Body(...)):
    return await _save_items("carts", body.email.strip().lower(), [item.model_dump() for item in body.items])


@app.delete(
    "/api/cart/items/{item_id}",
    response_model=CartResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["cart"],
    summary="Delete cart item",
)
async def delete_cart_item(item_id: str, email: str):
    return await _remove_item("carts", email.strip().lower(), item_id)


@app.delete(
    "/api/cart",
    response_model=CartResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["cart"],
    summary="Clear cart",
)
async def clear_cart(email: str):
    return await _save_items("carts", email.strip().lower(), [])


@app.get(
    "/api/wishlist",
    response_model=WishlistResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["wishlist"],
    summary="Get wishlist",
)
async def get_wishlist(email: str):
    return await _get_saved_items("wishlists", email.strip().lower())


@app.put(
    "/api/wishlist",
    response_model=WishlistResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["wishlist"],
    summary="Save wishlist",
    description="Replaces the entire wishlist with the given items list.",
)
async def save_wishlist(body: WishlistSaveRequest = Body(...)):
    return await _save_items("wishlists", body.email.strip().lower(), [item.model_dump() for item in body.items])


@app.delete(
    "/api/wishlist/items/{item_id}",
    response_model=WishlistResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["wishlist"],
    summary="Delete wishlist item",
)
async def delete_wishlist_item(item_id: str, email: str):
    return await _remove_item("wishlists", email.strip().lower(), item_id)


@app.delete(
    "/api/wishlist",
    response_model=WishlistResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["wishlist"],
    summary="Clear wishlist",
)
async def clear_wishlist(email: str):
    return await _save_items("wishlists", email.strip().lower(), [])
