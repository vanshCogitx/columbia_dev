"""Imports an exported Cartesian workflow JSON (nodes + edges + each agent's
own prompt) into workflow_versions / workflow_nodes / workflow_edges — the
static prompt library the Layer 2 judge (judge.py) reads from for root-cause
attribution. Never fetched at request time; re-run this by hand whenever the
workflow changes in Cartesian's builder and you re-export it.

Usage:
    python tools/ingest_workflow.py path/to/exported_workflow.json
"""
import os
import sys
import json

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


def ensure_tables(conn):
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS workflow_versions (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        raw_json JSONB NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT true,
        imported_at TIMESTAMPTZ DEFAULT now()
    )
    """))
    conn.execute(text("""
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
    """))
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS workflow_edges (
        id SERIAL PRIMARY KEY,
        workflow_version_id INTEGER NOT NULL REFERENCES workflow_versions(id) ON DELETE CASCADE,
        source_node_id TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        source_handle TEXT
    )
    """))


def ingest(path: str):
    if not SUPABASE_DB_URL:
        print("ERROR: SUPABASE_DB_URL is not set in the environment or .env file.")
        sys.exit(1)

    with open(path) as f:
        workflow = json.load(f)

    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    name = workflow.get("name", os.path.basename(path))
    description = workflow.get("description")

    print(f"Ingesting workflow '{name}' ({len(nodes)} nodes, {len(edges)} edges) from {path}...")

    engine = create_engine(SUPABASE_DB_URL)
    with engine.begin() as conn:
        ensure_tables(conn)

        # Only one active version at a time — earlier imports stay in the
        # table (never overwritten) so past prompt states remain queryable.
        conn.execute(text("UPDATE workflow_versions SET is_active = false WHERE is_active = true"))

        version_id = conn.execute(
            text("""
                INSERT INTO workflow_versions (name, description, raw_json, is_active)
                VALUES (:name, :description, :raw_json, true)
                RETURNING id
            """),
            {"name": name, "description": description, "raw_json": json.dumps(workflow)},
        ).scalar_one()

        for n in nodes:
            config = n.get("config", {}) or {}
            conn.execute(
                text("""
                    INSERT INTO workflow_nodes
                        (workflow_version_id, node_id, node_type, alias, agent_instructions,
                         model, provider, temperature, raw_config)
                    VALUES
                        (:workflow_version_id, :node_id, :node_type, :alias, :agent_instructions,
                         :model, :provider, :temperature, :raw_config)
                """),
                {
                    "workflow_version_id": version_id,
                    "node_id": n.get("id"),
                    "node_type": n.get("type"),
                    "alias": n.get("alias"),
                    "agent_instructions": config.get("agentInstructions"),
                    "model": config.get("model"),
                    "provider": config.get("provider"),
                    "temperature": config.get("temperature"),
                    "raw_config": json.dumps(config),
                },
            )

        for e in edges:
            conn.execute(
                text("""
                    INSERT INTO workflow_edges (workflow_version_id, source_node_id, target_node_id, source_handle)
                    VALUES (:workflow_version_id, :source_node_id, :target_node_id, :source_handle)
                """),
                {
                    "workflow_version_id": version_id,
                    "source_node_id": e.get("source"),
                    "target_node_id": e.get("target"),
                    "source_handle": e.get("sourceHandle"),
                },
            )

    agent_count = sum(1 for n in nodes if n.get("type") == "agent")
    print(f"Done. workflow_versions.id={version_id}, {agent_count} agent prompts ingested.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/ingest_workflow.py path/to/exported_workflow.json")
        sys.exit(1)
    ingest(sys.argv[1])
