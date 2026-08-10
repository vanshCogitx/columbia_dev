import os
import sys
import time
import requests
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors.exceptions import NotFoundError

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CATALOG_V3_INDEX_NAME = os.getenv("PINECONE_CATALOG_V3_INDEX_NAME", "columbia-catalog-v3")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m")

# Path to the corrected flat catalog CSV — same schema as the V2 catalog,
# minus the unused 'handle' column, with data-quality fixes applied.
CATALOG_CSV = "product_catalog_final(Sheet1).csv"

# Pinecone upsert batch size
PINECONE_BATCH_SIZE = 100
# HuggingFace embedding batch size
HF_BATCH_SIZE = 16


# ------------------------------------------------------------------
# Embedding helper
# ------------------------------------------------------------------

def query_hf_embeddings(texts: list[str], token: str, model_name: str) -> list:
    """Query HF Inference API (Gemma) to get feature-extraction embeddings."""
    api_url = (
        f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
    )
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(5):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=30,
            )
            if response.status_code != 200:
                print(f"  HF API returned {response.status_code}: {response.text}. Retrying...")
                time.sleep(5)
                continue

            res_json = response.json()
            if isinstance(res_json, dict) and "error" in res_json:
                print(f"  HF API error: {res_json['error']}. Retrying...")
                time.sleep(5)
                continue

            return res_json
        except Exception as e:
            print(f"  HTTP request failed: {e}. Retrying...")
            time.sleep(5)

    raise RuntimeError("Failed to get embeddings from HF Inference API after 5 attempts.")


# ------------------------------------------------------------------
# CSV loader + cleaner
# ------------------------------------------------------------------

def load_catalog(csv_path: str) -> pd.DataFrame:
    """Load and clean the corrected flat product catalog CSV."""
    print(f"Loading CSV: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp1252")

    print(f"  Loaded {len(df)} rows.")

    # --- Clean columns ---
    # (No 'handle' column in this sheet — it was unused in build_document/
    # build_metadata anyway, so nothing downstream depended on it.)
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).astype(float)
    df["category_level_1"] = df["category_level_1"].fillna("Unspecified").astype(str).str.strip()
    df["category_level_2"] = df["category_level_2"].fillna("Unspecified").astype(str).str.strip()
    df["product_type"] = df["product_type"].fillna("Unspecified").astype(str).str.strip()
    df["sport"] = df["sport"].fillna("Unspecified").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["image_url"] = df["image_url"].fillna("").astype(str).str.strip()
    df["tags"] = df["tags"].fillna("").astype(str).str.strip()
    df["color"] = df["color"].fillna("Unspecified").astype(str).str.strip()
    df["material"] = df["material"].fillna("Unspecified").astype(str).str.strip()
    df["fit"] = df["fit"].fillna("Unspecified").astype(str).str.strip()
    df["features"] = df["features"].fillna("").astype(str).str.strip()
    df["size"] = df["size"].fillna("Unspecified").astype(str).str.strip()
    df["stock_quantity"] = pd.to_numeric(df["stock_quantity"], errors="coerce").fillna(0).astype(int)

    # Drop duplicate product_ids (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["product_id"], keep="first")
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} duplicate product_ids. Remaining: {after}")

    return df


# ------------------------------------------------------------------
# Document builder — what gets embedded into the vector
# ------------------------------------------------------------------

def build_document(row: pd.Series) -> str:
    """
    Build a rich text string from a product row.
    This text is passed to Gemma to generate the semantic embedding.
    """
    parts = [
        f"Brand: Columbia",
        f"Category: {row['category_level_1']} > {row['category_level_2']}",
    ]
    if row["product_type"] not in ("", "Unspecified"):
        parts.append(f"Type: {row['product_type']}")
    if row["sport"] not in ("", "Unspecified"):
        parts.append(f"Sport: {row['sport']}")
    if row["color"] not in ("", "Unspecified"):
        parts.append(f"Color: {row['color']}")
    if row["material"] not in ("", "Unspecified"):
        parts.append(f"Material: {row['material']}")
    if row["fit"] not in ("", "Unspecified"):
        parts.append(f"Fit: {row['fit']}")
    if row["features"]:
        parts.append(f"Features: {row['features']}")
    if row["size"] not in ("", "Unspecified"):
        parts.append(f"Size: {row['size']}")
    parts.append(f"Price: ${row['price']:.2f}")
    if row["tags"]:
        parts.append(f"Tags: {row['tags']}")
    if row["description"]:
        parts.append(f"Description: {row['description'][:400]}")

    body = "\n".join(parts)
    return f"title: {row['name']} | text: {body}"


# ------------------------------------------------------------------
# Metadata builder — stored alongside the vector in Pinecone
# ------------------------------------------------------------------

def build_metadata(row: pd.Series) -> dict:
    """
    Build the Pinecone metadata dict for a product row.
    Used for pre-filtering queries and returning results without a DB lookup.
    """
    desc = row["description"]
    snippet = (desc[:200] + "...") if len(desc) > 200 else desc

    return {
        "name": row["name"],
        "price": float(row["price"]),
        "category_level_1": row["category_level_1"],
        "category_level_2": row["category_level_2"],
        "product_type": row["product_type"],
        "sport": row["sport"],
        "color": row["color"],
        "material": row["material"],
        "fit": row["fit"],
        "features": row["features"],
        "size": row["size"],
        "stock_quantity": int(row["stock_quantity"]),
        "url": row["url"],
        "image_url": row["image_url"],
        "description_snippet": snippet,
    }


# ------------------------------------------------------------------
# Main ingestion
# ------------------------------------------------------------------

def main():
    # --- Validate env ---
    if not PINECONE_API_KEY or PINECONE_API_KEY.startswith("your_"):
        print("ERROR: PINECONE_API_KEY is not set.")
        sys.exit(1)
    if not HF_TOKEN or HF_TOKEN.startswith("your_"):
        print("ERROR: HF_TOKEN is not set.")
        sys.exit(1)

    print("=== Starting Ingestion: Flat Catalog V3 (corrected) → Pinecone ===")
    print(f"  Index : {PINECONE_CATALOG_V3_INDEX_NAME}")
    print(f"  Model : {MODEL_NAME}")

    # 1. Load CSV
    df = load_catalog(CATALOG_CSV)

    # 2. Build documents to embed
    print("Building embedding documents...")
    documents = [build_document(row) for _, row in df.iterrows()]

    # 3. Discover embedding dimension via a test call
    print(f"Probing embedding dimension with model '{MODEL_NAME}'...")
    try:
        test_res = query_hf_embeddings([documents[0]], HF_TOKEN, MODEL_NAME)
        sample_emb = test_res[0]
        if isinstance(sample_emb[0], list):
            sample_emb = np.mean(np.array(sample_emb), axis=0).tolist()
        embedding_dim = len(sample_emb)
        print(f"  Embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"ERROR: Failed test embedding: {e}")
        sys.exit(1)

    # 4. Connect to Pinecone — create the index if it doesn't exist yet
    print("Connecting to Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        active_indexes = [idx.name for idx in pc.list_indexes()]
    except Exception as e:
        print(f"ERROR: Failed to connect to Pinecone: {e}")
        sys.exit(1)

    if PINECONE_CATALOG_V3_INDEX_NAME in active_indexes:
        existing_dim = pc.describe_index(PINECONE_CATALOG_V3_INDEX_NAME).dimension
        if existing_dim != embedding_dim:
            print(
                f"  Dimension mismatch: index has {existing_dim}D but model outputs {embedding_dim}D. "
                f"Deleting '{PINECONE_CATALOG_V3_INDEX_NAME}' and recreating..."
            )
            pc.delete_index(PINECONE_CATALOG_V3_INDEX_NAME)
            print("  Waiting for deletion to complete...")
            while PINECONE_CATALOG_V3_INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
                time.sleep(2)
            print("  Index deleted.")
            active_indexes = [idx.name for idx in pc.list_indexes()]
        else:
            print(f"  Connected to existing index '{PINECONE_CATALOG_V3_INDEX_NAME}' (dim={existing_dim}).")

    if PINECONE_CATALOG_V3_INDEX_NAME not in active_indexes:
        print(f"Creating serverless index '{PINECONE_CATALOG_V3_INDEX_NAME}' (dim={embedding_dim})...")
        try:
            pc.create_index(
                name=PINECONE_CATALOG_V3_INDEX_NAME,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("Waiting for index to be ready...")
            while not pc.describe_index(PINECONE_CATALOG_V3_INDEX_NAME).status.ready:
                time.sleep(2)
            print("Index ready.")
        except Exception as e:
            print(f"ERROR: Failed to create index: {e}")
            sys.exit(1)

    index = pc.Index(PINECONE_CATALOG_V3_INDEX_NAME)

    # 5. Clear any existing vectors before a fresh ingest
    print("Clearing existing vectors from index...")
    try:
        index.delete(delete_all=True)
        print("  Index cleared ✓")
    except NotFoundError:
        # A brand-new index has no default namespace until something is
        # upserted into it — delete(delete_all=True) 404s on that namespace
        # in that case. Nothing to clear, so just move on.
        print("  Nothing to clear (fresh index, no namespace yet).")

    # 6. Embed + upsert in batches
    print(f"\nIngesting {len(df)} products in HF batches of {HF_BATCH_SIZE}...")
    upsert_buffer = []
    total_upserted = 0

    for i in range(0, len(documents), HF_BATCH_SIZE):
        end_idx = min(i + HF_BATCH_SIZE, len(documents))
        batch_docs = documents[i:end_idx]
        batch_rows = df.iloc[i:end_idx]

        print(f"  Embedding items {i}–{end_idx - 1}...")
        try:
            batch_res = query_hf_embeddings(batch_docs, HF_TOKEN, MODEL_NAME)
        except Exception as e:
            print(f"ERROR: Embedding batch {i}–{end_idx - 1} failed: {e}. Exiting.")
            sys.exit(1)

        for offset, emb in enumerate(batch_res):
            row = batch_rows.iloc[offset]

            # Mean-pool token embeddings if model returns a matrix
            if isinstance(emb[0], list):
                emb_vector = np.mean(np.array(emb), axis=0).tolist()
            else:
                emb_vector = emb

            upsert_buffer.append({
                "id": row["product_id"],
                "values": emb_vector,
                "metadata": build_metadata(row),
            })

            # Flush to Pinecone when buffer is full
            if len(upsert_buffer) >= PINECONE_BATCH_SIZE:
                print(f"  Upserting {len(upsert_buffer)} vectors...")
                index.upsert(vectors=upsert_buffer)
                total_upserted += len(upsert_buffer)
                upsert_buffer = []

        # Respect HF rate limits
        time.sleep(0.5)

    # Flush remaining
    if upsert_buffer:
        print(f"  Upserting final {len(upsert_buffer)} vectors...")
        index.upsert(vectors=upsert_buffer)
        total_upserted += len(upsert_buffer)

    print("\n==========================================")
    print("INGESTION COMPLETE!")
    print(f"  Total products upserted: {total_upserted}")
    print(f"  Pinecone index        : {PINECONE_CATALOG_V3_INDEX_NAME}")
    print("==========================================")


if __name__ == "__main__":
    main()
