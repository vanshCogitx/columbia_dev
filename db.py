import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Security, Depends, Query
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

# V3: same engine class, pointed at the corrected-catalog index — kept
# separate from v2 on purpose so v2's results stay unchanged.
catalog_engine_v3 = CatalogSearchEngineV2(
    index_name=os.getenv("PINECONE_CATALOG_V3_INDEX_NAME", "columbia-catalog-v3")
)

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

        # Evals system, Layer 1: thumbs up/down feedback on an AI message,
        # with an optional free-text reason. This is what triggers the
        # Layer 2 judge pipeline (see judge.py) — the judge only ever runs
        # off real feedback, not on every response.
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS response_feedback (
            id SERIAL PRIMARY KEY,
            chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            reason TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_message ON response_feedback(message_id)")

        # Evals system: static reference copy of the Cartesian workflow's own
        # node definitions (prompts, model config, graph edges), imported from
        # a manually-exported workflow JSON via tools/ingest_workflow.py. Never
        # fetched at request time — this is the prompt library the Layer 2
        # judge reads from for root-cause attribution. A new version row is
        # inserted on every import (never overwritten) so past versions stay
        # available for correlating "did quality drop after this prompt changed".
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS workflow_versions (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            raw_json JSONB NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT true,
            imported_at TIMESTAMPTZ DEFAULT now()
        )
        """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS workflow_nodes (
            id SERIAL PRIMARY KEY,
            workflow_version_id INTEGER NOT NULL REFERENCES workflow_versions(id) ON DELETE CASCADE,
            node_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            alias TEXT,
            agent_instructions TEXT,
            model TEXT,
            provider TEXT,
            temperature REAL,
            raw_config JSONB,
            UNIQUE (workflow_version_id, node_id)
        )
        """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS workflow_edges (
            id SERIAL PRIMARY KEY,
            workflow_version_id INTEGER NOT NULL REFERENCES workflow_versions(id) ON DELETE CASCADE,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            source_handle TEXT
        )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_nodes_version ON workflow_nodes(workflow_version_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_edges_version ON workflow_edges(workflow_version_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_nodes_node_id ON workflow_nodes(workflow_version_id, node_id)")

        # Raw Cartesian response metadata for each AI message, saved at
        # generation time since feedback (and therefore the Layer 2 judge)
        # can arrive long after the run is over — we can't go re-fetch this
        # from Cartesian later. raw_output.executionPath / .agentRawResponses
        # tell judge.py's Stage C exactly which agents ran for this specific
        # response, so it only needs those agents' prompts instead of the
        # whole library (see run_judge_pipeline).
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS message_traces (
            message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
            raw_output JSONB,
            evaluated_branches JSONB,
            execution_summary JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)

        # Evals system, Layer 2 output — see judge.py. One row per feedback
        # event: initial_score/corrected_score are the self-critique pass's
        # before/after (Stage B); root_cause_* are only populated for bad
        # verdicts (Stage C).
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS response_judgments (
            id SERIAL PRIMARY KEY,
            feedback_id INTEGER NOT NULL REFERENCES response_feedback(id) ON DELETE CASCADE,
            initial_score REAL,
            corrected_score REAL,
            judge_reasoning TEXT,
            root_cause_node TEXT,
            root_cause_snippet TEXT,
            root_cause_explanation TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)
        # Raw model "thinking" per stage — surfaced in the evals dashboard so
        # you can see not just the final verdict but how the judge reasoned
        # its way there.
        await conn.execute("ALTER TABLE response_judgments ADD COLUMN IF NOT EXISTS draft_thinking TEXT")
        await conn.execute("ALTER TABLE response_judgments ADD COLUMN IF NOT EXISTS critique_thinking TEXT")
        await conn.execute("ALTER TABLE response_judgments ADD COLUMN IF NOT EXISTS attribution_thinking TEXT")
        # Set when the pipeline raised instead of completing — without this,
        # a failed run never inserts a row at all, so the dashboard's "judged"
        # check (row exists?) can't tell "still running" apart from "finished,
        # but failed", and the feed list spins forever instead of showing the
        # error.
        await conn.execute("ALTER TABLE response_judgments ADD COLUMN IF NOT EXISTS error TEXT")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_judgments_feedback ON response_judgments(feedback_id)")

        # Evals system, "Testing" — proactive scenario simulation, distinct
        # from the reactive real-user feedback above. An LLM plays a
        # simulated shopper against the real Cartesian workflow for several
        # turns; kept fully separate from chats/messages/response_feedback/
        # response_judgments so synthetic test traffic never pollutes real
        # chat history or evals stats. judgments columns are an intentional
        # mirror of response_judgments' shape so judge.py's shared scoring
        # core can write either table with the same result dict.
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS simulation_scenarios (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            user_scenario TEXT NOT NULL,
            success_criteria JSONB NOT NULL,
            max_turns INTEGER NOT NULL DEFAULT 5,
            mock_tools BOOLEAN NOT NULL DEFAULT false,
            created_by_user_id INTEGER REFERENCES users(id),
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id SERIAL PRIMARY KEY,
            scenario_id INTEGER NOT NULL REFERENCES simulation_scenarios(id) ON DELETE CASCADE,
            session_id TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
            turn_count INTEGER NOT NULL DEFAULT 0,
            verdict TEXT CHECK (verdict IN ('pass', 'fail')),
            verdict_reasoning TEXT,
            verdict_thinking TEXT,
            error TEXT,
            started_at TIMESTAMPTZ DEFAULT now(),
            completed_at TIMESTAMPTZ
        )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_runs_scenario ON simulation_runs(scenario_id)")

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS simulation_turns (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
            turn_index INTEGER NOT NULL,
            simulated_user_text TEXT NOT NULL,
            ai_response_text TEXT,
            raw_output JSONB,
            created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE (run_id, turn_index)
        )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_turns_run ON simulation_turns(run_id)")

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS simulation_turn_feedback (
            id SERIAL PRIMARY KEY,
            turn_id INTEGER NOT NULL REFERENCES simulation_turns(id) ON DELETE CASCADE,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            reason TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_turn_feedback_turn ON simulation_turn_feedback(turn_id)")

        await conn.execute("""
        CREATE TABLE IF NOT EXISTS simulation_turn_judgments (
            id SERIAL PRIMARY KEY,
            turn_feedback_id INTEGER NOT NULL REFERENCES simulation_turn_feedback(id) ON DELETE CASCADE,
            initial_score REAL,
            corrected_score REAL,
            judge_reasoning TEXT,
            root_cause_node TEXT,
            root_cause_snippet TEXT,
            root_cause_explanation TEXT,
            draft_thinking TEXT,
            critique_thinking TEXT,
            attribution_thinking TEXT,
            error TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_turn_judgments_feedback ON simulation_turn_judgments(turn_feedback_id)")

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


async def get_current_user_from_token(token: str) -> dict:
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


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    return await get_current_user_from_token(credentials.credentials)


async def get_current_user_query(token: str = Query(...)):
    """Same auth, but reads the JWT from a query param instead of the
    Authorization header — needed for SSE endpoints, since browser
    EventSource can't set custom headers. Used only by the evals dashboard's
    stream endpoints (see routers/evals.py, routers/testing.py)."""
    return await get_current_user_from_token(token)


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
        catalog_engine_v3.load_data()

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
