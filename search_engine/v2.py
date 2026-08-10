import os
import time
import logging
import requests
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

logger = logging.getLogger("columbia_backend.search_engine_v2")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CATALOG_INDEX_NAME = os.getenv("PINECONE_CATALOG_INDEX_NAME", "columbia-catalog-v2")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m")


class CatalogSearchEngineV2:
    """
    Flat-catalog search engine backed exclusively by Pinecone.
    No Postgres / Supabase joins — all product data is stored in Pinecone metadata.
    Uses the same Gemma embedding model as the existing ProductSearchEngine.
    """

    def __init__(self, index_name: str = None):
        self.pc = None
        self.index = None
        self.index_name = index_name or PINECONE_CATALOG_INDEX_NAME

    def load_data(self):
        """Establish connection to the flat catalog Pinecone index."""
        if not PINECONE_API_KEY or PINECONE_API_KEY.startswith("your_"):
            logger.warning(
                "WARNING: PINECONE_API_KEY is not set. CatalogSearchEngineV2 will be unavailable."
            )
            return

        logger.info(f"Connecting to Pinecone catalog index: '{self.index_name}'...")
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            active_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in active_indexes:
                logger.warning(
                    f"WARNING: Pinecone index '{self.index_name}' does not exist. "
                    "Please run the matching ingestion script first."
                )
            else:
                self.index = self.pc.Index(self.index_name)
                logger.info("CatalogSearchEngineV2: Pinecone index connected successfully!")
        except Exception as e:
            logger.error(f"ERROR: Failed to connect to Pinecone catalog index: {e}")

    # ------------------------------------------------------------------
    # Embedding helper (identical pattern to existing search_engine.py)
    # ------------------------------------------------------------------

    def query_hf_embeddings(self, texts: list[str]) -> list:
        """Query Hugging Face Inference API (Gemma) for feature-extraction embeddings."""
        if not HF_TOKEN or HF_TOKEN.startswith("your_"):
            raise ValueError("HF_TOKEN is not configured in .env file.")

        api_url = (
            f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}/pipeline/feature-extraction"
        )
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        for attempt in range(3):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=15,
                )
                if response.status_code == 200:
                    res_json = response.json()
                    if isinstance(res_json, dict) and "error" in res_json:
                        time.sleep(2)
                        continue
                    return res_json
                else:
                    logger.info(
                        f"HF query attempt {attempt + 1} returned status {response.status_code}: {response.text}"
                    )
                    time.sleep(2)
            except Exception as e:
                logger.info(f"HF query attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        raise RuntimeError("Hugging Face Inference API query failed after 3 attempts.")

    # ------------------------------------------------------------------
    # Pinecone filter builder
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        category_level_1: str = None,
        category_level_2: str = None,
        product_type: str = None,
        sport: str = None,
        price_min: float = None,
        price_max: float = None,
        color: str = None,
        size: str = None,
    ) -> dict:
        """Build a Pinecone metadata filter dict from search parameters."""
        f = {}

        if category_level_1:
            f["category_level_1"] = {"$eq": category_level_1}
        if category_level_2:
            f["category_level_2"] = {"$eq": category_level_2}
        if product_type:
            f["product_type"] = {"$eq": product_type}
        if sport:
            f["sport"] = {"$eq": sport}
        if color:
            f["color"] = {"$eq": color}
        if size:
            f["size"] = {"$eq": size}

        # Price range — filter for products whose price falls within [price_min, price_max]
        if price_min is not None or price_max is not None:
            price_filter = {}
            if price_min is not None:
                price_filter["$gte"] = price_min
            if price_max is not None:
                price_filter["$lte"] = price_max
            f["price"] = price_filter

        return f

    # ------------------------------------------------------------------
    # Internal: query Pinecone and unpack metadata into result dicts
    # ------------------------------------------------------------------

    def _pinecone_query(self, query_emb: list, top_k: int, filter_dict: dict) -> list[dict]:
        """Run a Pinecone query and return a list of result dicts with scores."""
        try:
            response = self.index.query(
                vector=query_emb,
                top_k=top_k,
                filter=filter_dict if filter_dict else None,
                include_metadata=True,
            )
            logger.info(
                "[V2-SEARCH] Pinecone raw matches: %s",
                [{"id": m.id, "score": m.score} for m in response.matches],
            )
        except Exception as e:
            logger.error(f"[V2-SEARCH] Pinecone query failed: {e}")
            return []

        results = []
        for match in response.matches:
            meta = match.metadata or {}
            result = {
                "product_id": match.id,
                "name": meta.get("name", ""),
                "price": float(meta.get("price", 0.0)),
                "category_level_1": meta.get("category_level_1", ""),
                "category_level_2": meta.get("category_level_2", ""),
                "product_type": meta.get("product_type", ""),
                "sport": meta.get("sport", ""),
                "color": meta.get("color", ""),
                "material": meta.get("material", ""),
                "fit": meta.get("fit", ""),
                "features": meta.get("features", ""),
                "size": meta.get("size", ""),
                "stock_quantity": int(meta.get("stock_quantity", 0)),
                "availability": "in_stock" if int(meta.get("stock_quantity", 0)) > 0 else "out_of_stock",
                "url": meta.get("url", ""),
                "image_url": meta.get("image_url", ""),
                "description_snippet": meta.get("description_snippet", ""),
                "score": float(match.score),
            }
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Public search method
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str = None,
        category_level_1: str = None,
        category_level_2: str = None,
        product_type: str = None,
        sport: str = None,
        price_min: float = None,
        price_max: float = None,
        color: str = None,
        size: str = None,
        top_k: int = 10,
        min_score: float = 0.35,
    ) -> list[dict]:
        """
        Unified search for the flat catalog.

        - If no query is given, raises an error (a semantic query is required since
          there's no SQL fallback for this dataset).
        - If Pinecone is not connected, raises an error.
        - Uses 3-stage progressive filter relaxation to maximise results:
            Stage 1: Full filters (category + subcategory + sport + price + color + size)
            Stage 2: Core filters only (category_level_1 + product_type), drop soft filters
            Stage 3: No filters at all — pure semantic similarity
        """
        if self.index is None:
            raise RuntimeError(
                "CatalogSearchEngineV2: Pinecone index is not connected. "
                "Run ingest_catalog_v2.py first and ensure PINECONE_CATALOG_INDEX_NAME is set."
            )

        if not query or not query.strip():
            raise ValueError(
                "A 'query' string is required for the v2 catalog search endpoint."
            )

        # Normalise price guards
        if price_min is not None and price_min <= 0:
            price_min = None
        if price_max is not None and price_max <= 0:
            price_max = None

        # --- Embed query via Gemma ---
        logger.info(f"[V2-SEARCH] Embedding query: '{query}'")
        try:
            gemma_query = f"task: retrieval | query: {query}"
            batch_res = self.query_hf_embeddings([gemma_query])
            emb = batch_res[0]
            query_emb = np.mean(np.array(emb), axis=0).tolist() if isinstance(emb[0], list) else emb
        except Exception as e:
            logger.error(f"[V2-SEARCH] Failed to embed query: {e}")
            raise RuntimeError(f"Embedding failed: {e}")

        results: list[dict] = []
        seen_ids: set = set()
        fetch_k = top_k * 4  # over-fetch to compensate for score threshold drops

        # --------------------------------------------------
        # Stage 1: Full strict filter
        # --------------------------------------------------
        strict_filter = self._build_filter(
            category_level_1, category_level_2, product_type, sport,
            price_min, price_max, color, size
        )
        if strict_filter:
            logger.info(f"[V2-SEARCH] Stage 1 — strict filter: {strict_filter}")
            stage1 = self._pinecone_query(query_emb, fetch_k, strict_filter)
            for r in stage1:
                if r["score"] >= min_score and r["product_id"] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r["product_id"])

        # --------------------------------------------------
        # Stage 2: Relaxed filter — keep category_level_1 + product_type only
        # --------------------------------------------------
        if len(results) < top_k:
            relaxed_filter = self._build_filter(
                category_level_1=category_level_1,
                product_type=product_type,
            )
            has_soft_filters = bool(category_level_2 or sport or price_min or price_max or color or size)
            if relaxed_filter and has_soft_filters:
                logger.info(f"[V2-SEARCH] Stage 2 — relaxed filter: {relaxed_filter}")
                stage2 = self._pinecone_query(query_emb, fetch_k, relaxed_filter)
                for r in stage2:
                    if r["score"] >= min_score and r["product_id"] not in seen_ids:
                        results.append(r)
                        seen_ids.add(r["product_id"])

        # --------------------------------------------------
        # Stage 3: No filter — pure semantic search
        # --------------------------------------------------
        if len(results) < top_k:
            logger.info("[V2-SEARCH] Stage 3 — no filter (pure semantic)")
            stage3 = self._pinecone_query(query_emb, fetch_k, {})
            for r in stage3:
                if r["score"] >= min_score and r["product_id"] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r["product_id"])

        logger.info(f"[V2-SEARCH] Returning {min(len(results), top_k)} results.")
        return results[:top_k]
