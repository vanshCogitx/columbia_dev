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


_INIT_DB_LOCK_KEY = 727001  # arbitrary constant, just needs to be stable across deploys

async def init_db(pool: asyncpg.Pool):
    logger.info("Initializing Postgres (Supabase) database tables...")
    async with pool.acquire() as conn:
        # gunicorn boots several worker processes, each running this same
        # lifespan startup independently — without a lock, two workers can
        # both pass a CREATE ... IF NOT EXISTS's existence check before
        # either commits, and one gets a genuine UniqueViolationError from
        # Postgres's own catalog constraint despite the IF NOT EXISTS
        # (observed in production: two workers raced on the same CREATE
        # INDEX and one crashed, taking the whole app down until gunicorn
        # restarted it). This serializes every worker's migration behind a
        # session-level advisory lock — whoever gets there first runs it,
        # the rest just wait, then find everything already there. No
        # explicit unlock needed: if migration fails and the worker exits,
        # its connection (and the lock with it) dies along with it.
        await conn.execute("SELECT pg_advisory_lock($1)", _INIT_DB_LOCK_KEY)
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

        # Multi-project support for the evals dashboard — a "project" is a
        # distinct Cartesian workflow (its own export id + credentials).
        # Deliberately independent of the live chat pipeline: chats/messages
        # (below) NEVER reference projects in any way, so an empty or
        # mid-edit projects table can never block real chat traffic. Only
        # things a human explicitly creates for a specific project (ingested
        # workflow versions, testing scenarios, and is_live below) are
        # allowed to depend on a project row existing.
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            cartesian_export_id TEXT NOT NULL,
            cartesian_client_id TEXT NOT NULL,
            cartesian_client_secret TEXT NOT NULL,
            cartesian_base_url TEXT,
            created_by_user_id INTEGER REFERENCES users(id),
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)
        # Exactly one project may be "the live one" at a time — the project
        # whose workflow real Columbia chat traffic actually runs against.
        # This is a separate concept from workflow_versions.is_active (which
        # is scoped per-project, for that project's own prompt library) —
        # is_live identifies which *project* backs live chat, for attributing
        # real feedback (see response_feedback.project_id below). Nullable by
        # nature of being a plain boolean column — never blocks anything.
        await conn.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS is_live BOOLEAN NOT NULL DEFAULT false")
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_one_live ON projects ((is_live)) WHERE is_live = true"
        )
        # Seed the one project that already exists today, from the same env
        # vars/constant the (untouched) live chat pipeline already uses, and
        # mark it live — so real feedback has somewhere to attribute to out
        # of the box. Purely a convenience seed, not a requirement: chats
        # work fine even if this project (or any project) doesn't exist.
        await conn.execute(
            """
            INSERT INTO projects (name, cartesian_export_id, cartesian_client_id, cartesian_client_secret, cartesian_base_url, is_live)
            SELECT 'Columbia Sportswear', $1, $2, $3, $4, true
            WHERE NOT EXISTS (SELECT 1 FROM projects WHERE name = 'Columbia Sportswear')
            """,
            "6a798ee58953bade21e86591",  # matches cartesian.CARTESIAN_JOB_PATH's export id at the time of this migration
            os.getenv("CARTESIAN_CLIENT_ID"),
            os.getenv("CARTESIAN_CLIENT_SECRET"),
            os.getenv("CARTESIAN_BASE_URL"),
        )
        # If nothing is marked live yet (e.g. the seed above didn't fire
        # because a differently-named project already existed), fall back to
        # the earliest-created project — keeps get_live_project_id() usable
        # without forcing a manual step on every fresh environment.
        await conn.execute(
            """
            UPDATE projects SET is_live = true
            WHERE id = (SELECT id FROM projects ORDER BY id ASC LIMIT 1)
            AND NOT EXISTS (SELECT 1 FROM projects WHERE is_live = true)
            """
        )

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
        # Nullable, unrelated to chats/messages' own schema — set once at
        # INSERT time (see routers/chat.py's post_feedback) to whichever
        # project was is_live=true at that moment. A point-in-time snapshot,
        # not a live lookup: if the live project changes later, past feedback
        # still correctly reflects which project was actually live when it
        # happened. Never NOT NULL — a feedback row with no resolvable live
        # project at insert time just leaves this NULL and still saves fine.
        await conn.execute("ALTER TABLE response_feedback ADD COLUMN IF NOT EXISTS project_id INTEGER REFERENCES projects(id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_project ON response_feedback(project_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_message ON response_feedback(message_id)")

        # HITL (Human-in-the-loop) review — a second, independent opinion on
        # a feedback event, alongside (not overriding) the LLM judge's
        # response_judgments. No row exists until a human actually submits a
        # review — same "absence means not yet done" convention as
        # response_judgments, not a pre-created 'pending' placeholder.
        # One row per feedback_id for now (the UNIQUE constraint below) —
        # only one reviewer's verdict is kept. reviewer_user_id is still a
        # real column today so that when role-based, multi-reviewer access
        # lands later, the only schema change needed is relaxing that UNIQUE
        # to (feedback_id, reviewer_user_id) — this table doesn't need to be
        # redesigned for it.
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS response_human_reviews (
            id SERIAL PRIMARY KEY,
            feedback_id INTEGER NOT NULL UNIQUE REFERENCES response_feedback(id) ON DELETE CASCADE,
            reviewer_user_id INTEGER NOT NULL REFERENCES users(id),
            reason_valid BOOLEAN NOT NULL,
            score REAL,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)

        # Hybrid — a lightweight sign-off, not a computed blended score (per
        # the product decision): a human looks at the LLM judge's score and
        # the HITL reviewer's score side by side and approves the pair. Only
        # ever meaningful for a feedback event that already has both (see
        # routers/hybrid.py's feed query) — this table doesn't duplicate
        # either score, just records that someone signed off. Same
        # no-row-until-done convention as the other two.
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS hybrid_approvals (
            id SERIAL PRIMARY KEY,
            feedback_id INTEGER NOT NULL UNIQUE REFERENCES response_feedback(id) ON DELETE CASCADE,
            approved_by_user_id INTEGER NOT NULL REFERENCES users(id),
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """)

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
        await conn.execute("ALTER TABLE workflow_versions ADD COLUMN IF NOT EXISTS project_id INTEGER REFERENCES projects(id)")
        await conn.execute(
            "UPDATE workflow_versions SET project_id = (SELECT id FROM projects ORDER BY id LIMIT 1) WHERE project_id IS NULL"
        )
        await conn.execute("ALTER TABLE workflow_versions ALTER COLUMN project_id SET NOT NULL")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_versions_project ON workflow_versions(project_id)")
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
        # This one is load-bearing (unlike response_feedback.project_id
        # above, which is just an attribution snapshot) — Testing actually
        # calls Cartesian using this project's own credentials (see
        # simulation.py), not the global env vars. Safe to require: only
        # ever written by an explicit "create scenario for project X" action.
        await conn.execute("ALTER TABLE simulation_scenarios ADD COLUMN IF NOT EXISTS project_id INTEGER REFERENCES projects(id)")
        await conn.execute(
            "UPDATE simulation_scenarios SET project_id = (SELECT id FROM projects ORDER BY id LIMIT 1) WHERE project_id IS NULL"
        )
        await conn.execute("ALTER TABLE simulation_scenarios ALTER COLUMN project_id SET NOT NULL")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_scenarios_project ON simulation_scenarios(project_id)")

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
        await conn.execute("SELECT pg_advisory_unlock($1)", _INIT_DB_LOCK_KEY)
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
