import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import asyncpg
import redis.asyncio as redis_asyncio
import jwt as pyjwt

from search_engine.v1 import ProductSearchEngine
from search_engine.v2 import CatalogSearchEngineV2
from otp import otp_auth

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
