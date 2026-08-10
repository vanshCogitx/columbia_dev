import os
import time
import logging
import requests
import json
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

logger = logging.getLogger("columbia_backend.search_engine")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "columbia-inventory")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m")

class ProductSearchEngine:
    def __init__(self, pool=None):
        # asyncpg connection pool - set here or assigned later by the app's lifespan
        # (product_family / product_catalog tables live in Supabase Postgres now)
        self.pool = pool
        self.pc = None
        self.index = None

    def load_data(self):
        """Establish Pinecone connection. Postgres pool is assigned externally by the app lifespan."""
        if self.pool is None:
            logger.warning("WARNING: No Postgres pool assigned to ProductSearchEngine yet.")

        # Connect to Pinecone
        if not PINECONE_API_KEY or PINECONE_API_KEY.startswith("your_"):
            logger.warning("WARNING: PINECONE_API_KEY is placeholder or not set. Semantic search will fallback to SQL.")
        else:
            logger.info(f"Connecting to Pinecone index: '{PINECONE_INDEX_NAME}'...")
            try:
                self.pc = Pinecone(api_key=PINECONE_API_KEY)
                active_indexes = [idx.name for idx in self.pc.list_indexes()]
                if PINECONE_INDEX_NAME not in active_indexes:
                    logger.warning(f"WARNING: Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Please run ingestion script first.")
                else:
                    self.index = self.pc.Index(PINECONE_INDEX_NAME)
                    logger.info("Pinecone vector database connected successfully!")
            except Exception as e:
                logger.error(f"ERROR: Failed to connect to Pinecone: {e}")

    def query_hf_embeddings(self, texts: list[str]) -> list:
        """Queries HF Inference API to get feature extraction embedding for a list of texts."""
        if not HF_TOKEN or HF_TOKEN.startswith("your_"):
            raise ValueError("HF_TOKEN is not configured in .env file.")

        api_url = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}/pipeline/feature-extraction"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        for attempt in range(3):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=15
                )
                if response.status_code == 200:
                    res_json = response.json()
                    if isinstance(res_json, dict) and "error" in res_json:
                        time.sleep(2)
                        continue
                    return res_json
                else:
                    logger.info(f"HF query attempt {attempt+1} returned status {response.status_code}: {response.text}")
                    time.sleep(2)
            except Exception as e:
                logger.info(f"HF query attempt {attempt+1} failed: {e}")
                time.sleep(2)

        raise RuntimeError("Hugging Face Inference API query failed after 3 attempts.")

    async def get_facets(self) -> dict:
        """Returns unique categories, brands, sports, and product types from the product_family table."""
        facets = {}
        try:
            rows = await self.pool.fetch("SELECT DISTINCT category_level_1 FROM product_family ORDER BY category_level_1")
            facets["category_level_1"] = [r["category_level_1"] for r in rows if r["category_level_1"]]

            rows = await self.pool.fetch("SELECT DISTINCT category_level_2 FROM product_family ORDER BY category_level_2")
            facets["category_level_2"] = [r["category_level_2"] for r in rows if r["category_level_2"]]

            rows = await self.pool.fetch("SELECT DISTINCT product_type FROM product_family ORDER BY product_type")
            facets["product_type"] = [r["product_type"] for r in rows if r["product_type"]]

            rows = await self.pool.fetch("SELECT DISTINCT sport FROM product_family ORDER BY sport")
            facets["sport"] = [r["sport"] for r in rows if r["sport"]]

            price_row = await self.pool.fetchrow(
                "SELECT MIN(price_min) AS min_p, MAX(price_max) AS max_p FROM product_family"
            )
            facets["price_min"] = float(price_row["min_p"]) if price_row["min_p"] is not None else 0.0
            facets["price_max"] = float(price_row["max_p"]) if price_row["max_p"] is not None else 0.0

            facets["brands"] = ["Columbia"]
        except Exception as e:
            logger.error(f"ERROR fetching facets: {e}")

        return facets

    async def _get_variants_for_families(self, family_ids: list[str], color: str = None, size: str = None, in_stock_only: bool = True) -> dict:
        """Helper to fetch and filter variants from Postgres for a list of family IDs."""
        if not family_ids:
            return {}

        conditions = ["family_id = ANY($1::text[])"]
        params = [family_ids]

        if color:
            params.append(color)
            conditions.append(f"color = ${len(params)}")
        if size:
            params.append(size)
            conditions.append(f"size = ${len(params)}")
        if in_stock_only:
            conditions.append("stock_quantity > 0")

        query = f"SELECT * FROM product_catalog WHERE {' AND '.join(conditions)}"
        rows = await self.pool.fetch(query, *params)

        # Group variants by family_id
        variants_by_family = {fid: [] for fid in family_ids}
        for row in rows:
            v_dict = dict(row)
            # Remove redundant family_id from variant representation
            fid = v_dict.pop("family_id")
            # Populate availability dynamically
            v_dict["availability"] = "in_stock" if v_dict.get("stock_quantity", 0) > 0 else "out_of_stock"
            variants_by_family[fid].append(v_dict)

        return variants_by_family

    async def _get_families_by_ids(self, family_ids: list[str]) -> list[dict]:
        """Fetch general family details from Postgres for a list of family IDs."""
        if not family_ids:
            return []

        rows = await self.pool.fetch(
            "SELECT * FROM product_family WHERE family_id = ANY($1::text[])", family_ids
        )

        # Create mapping to preserve the input family_ids order (Pinecone's ranking order)
        families_map = {row["family_id"]: dict(row) for row in rows}

        ordered_families = []
        for fid in family_ids:
            if fid in families_map:
                ordered_families.append(families_map[fid])
        return ordered_families

    async def _direct_id_lookup(self, family_id: str, color: str = None, size: str = None, in_stock_only: bool = True) -> list[dict]:
        """Performs a direct relational query using family_id."""
        families = await self._get_families_by_ids([family_id])
        if not families:
            return []

        family = families[0]
        variants_map = await self._get_variants_for_families([family_id], color, size, in_stock_only)
        family["variants"] = variants_map.get(family_id, [])
        family["score"] = 1.0  # Exact match score

        return [family]

    async def _structured_sql_search(
        self,
        category_level_1: str = None,
        category_level_2: str = None,
        product_type: str = None,
        sport: str = None,
        price_min: float = None,
        price_max: float = None,
        color: str = None,
        size: str = None,
        in_stock_only: bool = True,
        top_k: int = 10
    ) -> list[dict]:
        """Performs a pure SQL-based structured search when no semantic query is present."""
        conditions = ["1=1"]
        params = []

        if category_level_1:
            params.append(category_level_1)
            conditions.append(f"category_level_1 = ${len(params)}")
        if category_level_2:
            params.append(category_level_2)
            conditions.append(f"category_level_2 = ${len(params)}")
        if product_type:
            params.append(product_type)
            conditions.append(f"product_type = ${len(params)}")
        if sport:
            params.append(sport)
            conditions.append(f"sport = ${len(params)}")
        if price_min is not None:
            params.append(price_min)
            conditions.append(f"price_max >= ${len(params)}")
        if price_max is not None:
            params.append(price_max)
            conditions.append(f"price_min <= ${len(params)}")

        # Limits candidates
        query = f"SELECT * FROM product_family WHERE {' AND '.join(conditions)} LIMIT {top_k * 2}"

        family_rows = await self.pool.fetch(query, *params)

        if not family_rows:
            return []

        family_ids = [row["family_id"] for row in family_rows]

        # Fetch matching variants
        variants_map = await self._get_variants_for_families(family_ids, color, size, in_stock_only)

        # Group and construct results
        results = []
        for row in family_rows:
            fid = row["family_id"]
            family_dict = dict(row)
            variants = variants_map.get(fid, [])

            # If variant-level filters were passed and no variants match, omit this family
            if (color or size or in_stock_only) and not variants:
                continue

            family_dict["variants"] = variants
            family_dict["score"] = 1.0
            results.append(family_dict)

        return results[:top_k]

    def _build_pinecone_filter(
        self,
        category_level_1: str = None,
        category_level_2: str = None,
        product_type: str = None,
        sport: str = None,
        price_min: float = None,
        price_max: float = None
    ) -> dict:
        """Constructs Pinecone metadata filter matching family-level attributes."""
        p_filter = {}

        if category_level_1:
            p_filter["category_level_1"] = category_level_1
        if category_level_2:
            p_filter["category_level_2"] = category_level_2
        if product_type:
            p_filter["product_type"] = product_type
        if sport:
            p_filter["sport"] = sport

        # Pinecone handles price filter on family price boundaries
        if price_min is not None or price_max is not None:
            # We filter for families whose range overlaps with user's bounds
            # i.e., family_max >= price_min AND family_min <= price_max
            if price_min is not None:
                p_filter["price_max"] = {"$gte": price_min}
            if price_max is not None:
                p_filter["price_min"] = {"$lte": price_max}

        return p_filter

    async def search(
        self,
        query: str = None,
        family_id: str = None,
        category_level_1: str = None,
        category_level_2: str = None,
        product_type: str = None,
        sport: str = None,
        price_min: float = None,
        price_max: float = None,
        color: str = None,
        size: str = None,
        in_stock_only: bool = True,
        top_k: int = 10
    ) -> list[dict]:
        """Unified Search Method. Routes to Direct Lookup, SQL search, or Pinecone RAG search with progressive relaxation."""

        # Normalize price parameters (ignore negative or zero filters)
        if price_min is not None and price_min <= 0:
            price_min = None
        if price_max is not None and price_max <= 0:
            price_max = None

        # Route 1: Direct ID Lookup
        if family_id and family_id.strip():
            logger.info(f"Direct ID Lookup: family_id={family_id}")
            return await self._direct_id_lookup(family_id, color, size, in_stock_only)

        # Route 2: Structured SQL Search Fallback (no semantic query text or Pinecone unavailable)
        if not query or not query.strip() or self.index is None:
            if not query or not query.strip():
                logger.info("No semantic query text. Executing structured SQL search...")
            else:
                logger.info("Pinecone is not connected. Falling back to structured SQL search...")
            return await self._structured_sql_search(
                category_level_1, category_level_2, product_type, sport,
                price_min, price_max, color, size, in_stock_only, top_k
            )

        # Route 3: Semantic RAG Search via Pinecone + Postgres
        logger.info(f"Semantic RAG Search: query='{query}'")

        # 1. Embed query via Hugging Face API using the Gemma instruction prefix
        try:
            gemma_query = f"task: retrieval | query: {query}"
            batch_res = self.query_hf_embeddings([gemma_query])
            emb = batch_res[0]
            # Pool if sequence representation
            if isinstance(emb[0], list):
                query_emb = np.mean(np.array(emb), axis=0).tolist()
            else:
                query_emb = emb
        except Exception as e:
            logger.error(f"ERROR: Failed to embed query using HF API: {e}. Falling back to SQL search.")
            return await self._structured_sql_search(
                category_level_1, category_level_2, product_type, sport,
                price_min, price_max, color, size, in_stock_only, top_k
            )

        results = []
        seen_family_ids = set()

        # Build Pinecone metadata pre-filter
        p_filter = self._build_pinecone_filter(
            category_level_1, category_level_2, product_type, sport, price_min, price_max
        )

        # Build a relaxed pre-filter locking down core department/type
        p_filter_relaxed = {}
        if category_level_1:
            p_filter_relaxed["category_level_1"] = category_level_1
        if product_type:
            p_filter_relaxed["product_type"] = product_type

        # Check if we have soft filters that can be dropped
        has_soft_filters = bool(category_level_2 or sport or price_min is not None or price_max is not None)

        family_ids_filtered = []
        scores_map_filtered = {}

        # ----------------------------------------------------
        # Stage 1: Strict Pinecone + Strict Variant Filter
        # ----------------------------------------------------
        if p_filter:
            logger.info(f"Querying Pinecone index with filters: {p_filter}...")
            try:
                response = self.index.query(
                    vector=query_emb,
                    top_k=top_k * 4,  # Fetch extra to accommodate variant filters
                    filter=p_filter,
                    include_metadata=False
                )
                family_ids_filtered = [match.id for match in response.matches]
                scores_map_filtered = {match.id: match.score for match in response.matches}
                logger.info(f"[SEARCH-ENGINE] [Vector DB Raw Response] Matches: {[{'id': match.id, 'score': match.score} for match in response.matches]}")
            except Exception as e:
                logger.error(f"ERROR: Filtered Pinecone query failed: {e}")

        if family_ids_filtered:
            families = await self._get_families_by_ids(family_ids_filtered)
            variants_map = await self._get_variants_for_families(family_ids_filtered, color, size, in_stock_only)
            logger.info(f"[SEARCH-ENGINE] Variants matching query filters: {list(variants_map.keys())}")
            for fid, v_list in variants_map.items():
                logger.info(f"[SEARCH-ENGINE]   Family {fid} has {len(v_list)} variants (filtered by color={color}, size={size}, in_stock={in_stock_only})")

            for family in families:
                fid = family["family_id"]
                variants = variants_map.get(fid, [])
                if (color or size or in_stock_only) and not variants:
                    continue
                score = float(scores_map_filtered.get(fid, 1.0))
                if score < 0.375:
                    continue
                if fid not in seen_family_ids:
                    family["variants"] = variants
                    family["score"] = score
                    results.append(family)
                    seen_family_ids.add(fid)

        # ----------------------------------------------------
        # Stage 2: Strict Pinecone + Relaxed Variant Filter
        # ----------------------------------------------------
        if len(results) < top_k and family_ids_filtered:
            logger.info("Attempting Stage 2 relaxation: Strict Pinecone filters + Relaxed variant filters (no size/color)...")
            families = await self._get_families_by_ids(family_ids_filtered)
            variants_map = await self._get_variants_for_families(family_ids_filtered, color=None, size=None, in_stock_only=in_stock_only)

            for family in families:
                fid = family["family_id"]
                score = float(scores_map_filtered.get(fid, 1.0))
                if score < 0.375:
                    continue
                if fid not in seen_family_ids:
                    variants = variants_map.get(fid, [])
                    family["variants"] = variants
                    family["score"] = score
                    results.append(family)
                    seen_family_ids.add(fid)

        # ----------------------------------------------------
        # Stage 3: Department-Locked Pinecone + Strict Variant Filter
        # ----------------------------------------------------
        if len(results) < top_k and (has_soft_filters or not p_filter):
            logger.info("Attempting Stage 3 relaxation: Department-locked Pinecone search + Strict variant filters...")
            try:
                response = self.index.query(
                    vector=query_emb,
                    top_k=top_k * 4,
                    filter=p_filter_relaxed if p_filter_relaxed else None,
                    include_metadata=False
                )
                family_ids_relaxed = [match.id for match in response.matches]
                scores_map_relaxed = {match.id: match.score for match in response.matches}

                new_family_ids = [fid for fid in family_ids_relaxed if fid not in seen_family_ids]

                if new_family_ids:
                    families = await self._get_families_by_ids(new_family_ids)
                    variants_map = await self._get_variants_for_families(new_family_ids, color, size, in_stock_only)

                    for family in families:
                        fid = family["family_id"]
                        variants = variants_map.get(fid, [])
                        if (color or size or in_stock_only) and not variants:
                            continue
                        score = float(scores_map_relaxed.get(fid, 1.0))
                        if score < 0.375:
                            continue
                        if fid not in seen_family_ids:
                            family["variants"] = variants
                            family["score"] = score
                            results.append(family)
                            seen_family_ids.add(fid)

                    # ----------------------------------------------------
                    # Stage 4: Department-Locked Pinecone + Relaxed Variant Filter
                    # ----------------------------------------------------
                    if len(results) < top_k:
                        logger.info("Attempting Stage 4 relaxation: Department-locked Pinecone search + Relaxed variant filters...")
                        variants_map = await self._get_variants_for_families(new_family_ids, color=None, size=None, in_stock_only=in_stock_only)
                        for family in families:
                            fid = family["family_id"]
                            score = float(scores_map_relaxed.get(fid, 1.0))
                            if score < 0.375:
                                continue
                            if fid not in seen_family_ids:
                                variants = variants_map.get(fid, [])
                                family["variants"] = variants
                                family["score"] = score
                                results.append(family)
                                seen_family_ids.add(fid)
            except Exception as e:
                logger.error(f"ERROR: Department-locked Pinecone query failed: {e}")

        # Return only the top_k ranked results
        return results[:top_k]
