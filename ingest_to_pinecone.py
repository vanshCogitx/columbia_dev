import os
import sys
import time
import requests
import json
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from csv_loader import load_csvs

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "columbia-inventory")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m")

def query_hf_embeddings(texts: list[str], token: str, model_name: str) -> list:
    """Queries HF Inference API to get feature extraction embeddings for a list of texts."""
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(5):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=30
            )

            if response.status_code != 200:
                print(f"HF API returned status {response.status_code}: {response.text}. Retrying...")
                time.sleep(5)
                continue

            res_json = response.json()

            # Check for standard HF error responses
            if isinstance(res_json, dict) and "error" in res_json:
                print(f"HF API Error: {res_json['error']}. Retrying...")
                time.sleep(5)
                continue

            return res_json
        except Exception as e:
            print(f"HTTP request failed: {e}. Retrying...")
            time.sleep(5)

    raise RuntimeError("Failed to retrieve embeddings from Hugging Face Inference API after 5 attempts.")

def main():
    if not PINECONE_API_KEY or PINECONE_API_KEY.startswith("your_"):
        print("ERROR: PINECONE_API_KEY is not set in the environment or .env file.")
        sys.exit(1)

    if not HF_TOKEN or HF_TOKEN.startswith("your_"):
        print("ERROR: HF_TOKEN is not set in the environment or .env file.")
        sys.exit(1)

    print("=== Starting Ingestion to Pinecone (Relational Gemma Mode) ===")

    # 1. Read + Clean CSV Files
    df_family, df_catalog = load_csvs()

    # 2. Generate Gemma Search Documents
    print("Generating search documents for unique product families...")
    documents = []
    for _, row in df_family.iterrows():
        parts = [
            f"Brand: Columbia",
            f"Categories: {row['category_level_1']} > {row['category_level_2']}"
        ]
        if row['product_type'] != "Unspecified":
            parts.append(f"Type: {row['product_type']}")
        if row['sport'] != "Unspecified":
            parts.append(f"Sport: {row['sport']}")

        # Parse available colors/sizes lists
        try:
            colors_list = json.loads(row['available_colors'])
            colors = ", ".join(colors_list)
        except:
            colors = str(row['available_colors']).replace('[', '').replace(']', '').replace('"', '')

        try:
            sizes_list = json.loads(row['available_sizes'])
            sizes = ", ".join(sizes_list)
        except:
            sizes = str(row['available_sizes']).replace('[', '').replace(']', '').replace('"', '')

        if colors:
            parts.append(f"Available Colors: {colors}")
        if sizes:
            parts.append(f"Available Sizes: {sizes}")

        if row['description'] and row['description'] != "NOT FOUND":
            parts.append(f"Description: {row['description'][:400]}")

        doc_body = "\n".join(parts)
        # Format for google/embeddinggemma-300m
        documents.append(f"title: {row['title']} | text: {doc_body}")

    # 3. Request single test embedding to discover dimension dynamically
    print(f"Requesting test embedding from Hugging Face model '{MODEL_NAME}'...")
    try:
        test_res = query_hf_embeddings([documents[0]], HF_TOKEN, MODEL_NAME)
        sample_emb = test_res[0]
        if isinstance(sample_emb[0], list):
            sample_emb = np.mean(np.array(sample_emb), axis=0).tolist()
        embedding_dim = len(sample_emb)
        print(f"✓ Discovered embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"ERROR: Failed test embedding request: {e}")
        sys.exit(1)

    # 4. Connect to Pinecone and create/recreate index
    print("Connecting to Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        active_indexes = [idx.name for idx in pc.list_indexes()]
    except Exception as e:
        print(f"ERROR: Failed to connect to Pinecone: {e}")
        sys.exit(1)

    if PINECONE_INDEX_NAME in active_indexes:
        try:
            index_desc = pc.describe_index(PINECONE_INDEX_NAME)
            existing_dim = index_desc.dimension
            if existing_dim != embedding_dim:
                print(f"Dimension mismatch: Index '{PINECONE_INDEX_NAME}' has dimension {existing_dim}, "
                      f"but model '{MODEL_NAME}' outputs dimension {embedding_dim}.")
                print(f"Deleting index '{PINECONE_INDEX_NAME}' to recreate it with correct dimension...")
                pc.delete_index(PINECONE_INDEX_NAME)

                print("Waiting for index deletion to complete...")
                while PINECONE_INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
                    time.sleep(2)
                print("Index deleted successfully.")
                active_indexes.remove(PINECONE_INDEX_NAME)
            else:
                print(f"Index '{PINECONE_INDEX_NAME}' already exists with correct dimension {existing_dim}.")
        except Exception as e:
            print(f"ERROR: Failed to verify index description: {e}")
            sys.exit(1)

    if PINECONE_INDEX_NAME not in active_indexes:
        print(f"Creating serverless index '{PINECONE_INDEX_NAME}' (dim={embedding_dim})...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{PINECONE_INDEX_NAME}' created. Waiting for initialization...")
            while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
                time.sleep(2)
            print("Index is ready.")
        except Exception as e:
            print(f"ERROR: Failed to create index in Pinecone: {e}")
            sys.exit(1)

    index = pc.Index(PINECONE_INDEX_NAME)

    # 5. Batch Ingest Families to Pinecone
    print("Starting batch embedding and ingestion...")
    hf_batch_size = 16
    pinecone_batch_size = 100

    upsert_buffer = []

    for i in range(0, len(documents), hf_batch_size):
        end_idx = min(i + hf_batch_size, len(documents))
        batch_docs = documents[i:end_idx]

        print(f"Embedding batch: items {i} to {end_idx-1}...")
        try:
            batch_res = query_hf_embeddings(batch_docs, HF_TOKEN, MODEL_NAME)
        except Exception as e:
            print(f"ERROR: Failed to embed batch at {i}: {e}. Exiting.")
            sys.exit(1)

        # Process and normalize embeddings
        for idx_offset, emb in enumerate(batch_res):
            global_idx = i + idx_offset
            row = df_family.iloc[global_idx]

            # Pool if sequence representation
            if isinstance(emb[0], list):
                emb_vector = np.mean(np.array(emb), axis=0).tolist()
            else:
                emb_vector = emb

            # Formulate metadata
            desc = row['description']
            snippet = desc[:150] + "..." if len(desc) > 150 else desc

            metadata = {
                "title": row['title'],
                "category_level_1": row['category_level_1'],
                "category_level_2": row['category_level_2'],
                "product_type": row['product_type'],
                "sport": row['sport'],
                "price_min": float(row['price_min']),
                "price_max": float(row['price_max']),
                "thumbnail_image": row['thumbnail_image'],
                "description_snippet": snippet
            }

            upsert_buffer.append({
                "id": row['family_id'],
                "values": emb_vector,
                "metadata": metadata
            })

            # Check if buffer is full and upsert
            if len(upsert_buffer) >= pinecone_batch_size:
                print(f"Upserting {len(upsert_buffer)} vectors to Pinecone...")
                index.upsert(vectors=upsert_buffer)
                upsert_buffer = []

        # Sleep slightly to respect Hugging Face API rate limits
        time.sleep(0.5)

    # Upsert remaining
    if upsert_buffer:
        print(f"Upserting final {len(upsert_buffer)} vectors to Pinecone...")
        index.upsert(vectors=upsert_buffer)

    print("\n==========================================")
    print("PINECONE INGESTION COMPLETED SUCCESSFULLY!")
    print(f"Uploaded {len(df_family)} product families to Pinecone.")
    print("==========================================")

if __name__ == "__main__":
    main()
