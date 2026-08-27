import logging

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import db
from routers import auth, chat, cart, search, evals, testing, projects, hitl, hybrid, traceability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("columbia_backend")

app = FastAPI(
    title="Columbia Inventory Agent Toolset & RAG API",
    description=(
        "An OpenAPI-compliant backend providing a single unified search and retrieval tool "
        "for AI agents to query the Columbia Sportswear inventory. Integrates semantic vector search "
        "(Pinecone) with relational inventory and variant mapping (Supabase Postgres)."
    ),
    version="1.0.0",
    lifespan=db.lifespan,
)

# Enable CORS for agent integration in web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(cart.router)
app.include_router(search.router)
app.include_router(evals.router)
app.include_router(testing.router)
app.include_router(projects.router)
app.include_router(hitl.router)
app.include_router(hybrid.router)
app.include_router(traceability.router)
