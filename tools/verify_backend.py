import os
import sys
import sqlite3

# search_engine/ lives at the repo root, one level up from this tools/ directory —
# add it to sys.path so this still works when run directly (python tools/verify_backend.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    print("=== Testing Columbia Inventory Search Engine (Dual-DB SQLite + Pinecone) ===")
    try:
        from search_engine.v1 import ProductSearchEngine
    except ImportError as e:
        print(f"Error importing ProductSearchEngine: {e}")
        sys.exit(1)

    # Initialize Engine
    engine = ProductSearchEngine()
    
    # Load Data
    try:
        engine.load_data()
        print("✓ Database loader initialized.")
    except Exception as e:
        print(f"✗ Failed to load database: {e}")
        sys.exit(1)

    # Check local SQLite database exists and is populated
    if not os.path.exists(engine.db_path):
        print(f"✗ Error: SQLite database not found at {engine.db_path}. Run ingestion first.")
        sys.exit(1)

    conn = sqlite3.connect(engine.db_path)
    cursor = conn.cursor()
    
    # Assert family table is populated
    cursor.execute("SELECT COUNT(*) FROM product_family")
    family_count = cursor.fetchone()[0]
    assert family_count > 0, "product_family table is empty!"
    print(f"✓ Found {family_count} unique product families in SQLite.")

    # Assert catalog table is populated
    cursor.execute("SELECT COUNT(*) FROM product_catalog")
    catalog_count = cursor.fetchone()[0]
    assert catalog_count > 0, "product_catalog table is empty!"
    print(f"✓ Found {catalog_count} catalog variant items in SQLite.")
    
    # Fetch a sample family_id for direct lookup tests
    cursor.execute("SELECT family_id, title FROM product_family LIMIT 1")
    sample_family = cursor.fetchone()
    sample_family_id = sample_family[0]
    sample_family_title = sample_family[1]
    
    conn.close()

    # Test Facets
    print("\n--- Testing Catalog Facets ---")
    facets = engine.get_facets()
    assert "category_level_1" in facets
    assert "brands" in facets
    print(f"Unique Brands: {facets['brands']}")
    print(f"Price Range: ${facets['price_min']} - ${facets['price_max']}")
    print(f"First 5 Product Types: {facets['product_type'][:5]}")
    print("✓ Facets retrieval verified.")

    # Test Direct ID Lookup
    print("\n--- Testing Direct ID Lookup ---")
    results = engine.search(family_id=sample_family_id)
    assert len(results) == 1, f"Expected 1 family result, got {len(results)}"
    res = results[0]
    assert res["family_id"] == sample_family_id
    assert "variants" in res and len(res["variants"]) > 0, "No variants returned for direct lookup!"
    print(f"✓ Direct ID Lookup returned '{res['title']}' with {len(res['variants'])} variants.")
    for v in res["variants"]:
        print(f"   Variant: color={v['color']}, size={v['size']}, price=${v['price']}, stock={v['stock_quantity']}")

    # Test Structured SQLite Fallback Search
    print("\n--- Testing Structured Fallback Search (SQLite Only) ---")
    results = engine.search(query=None, sport="Hiking", size="M", top_k=3)
    assert len(results) > 0, "No items returned for structured fallback!"
    print(f"✓ Structured search returned {len(results)} items matching sport='Hiking' and size='M'.")
    for res in results:
        print(f"   Family: '{res['title']}' ({res['family_id']})")
        for v in res["variants"]:
            assert v["size"] == "M", f"Expected size 'M', got {v['size']}"
            print(f"     - Variant Product: {v['name']} (${v['price']}) - Size: {v['size']}, Stock: {v['stock_quantity']}")

    # Check if Pinecone is connected
    if engine.index is None:
        print("\n[!] WARNING: Pinecone is not connected (or PINECONE_API_KEY is not set).")
        print("    Skipping hybrid semantic search tests.")
    else:
        print("\n=== RUNNING USER-SPECIFIED TEST SCENARIOS ===")
        
        scenarios = {
            "Size and Fit": [
                {"query": "women's omnifreeze jacket", "size": "M"},
                {"query": "outdry wide width hiking boots"},
                {"query": "plus size fleece"},
                {"query": "men's windbreaker jacket", "size": "L"},
                {"query": "omnigrip boots"}
            ],
            "Activity Planning": [
                {"query": "ski jacket for Angel's landing"},
                {"query": "hiking gears for Yosemite National Park"},
                {"query": "rain gears for koko craterarch trail"},
                {"query": "rain jackets for Hoh Rain Forest Loop camping"},
                {"query": "backpacking backpack for diamond head state"}
            ],
            "Budget": [
                {"query": "fleece", "price_max": 100.0},
                {"query": "Women's long sleeve jacket", "price_max": 50.0},
                {"query": "flash forward windbreaker", "price_max": 100.0},
                {"query": "crew neck Tshirt", "price_max": 50.0},
                {"query": "winter omnifreeze jacket", "price_max": 100.0}
            ],
            "Fabric and Tech": [
                {"query": "insulated jacket"},
                {"query": "omnifreeze zero tshirt"},
                {"query": "omni-heat jacket"},
                {"query": "lightweight summer shirt"},
                {"query": "thermal reflective jacket"}
            ],
            "Kids Specific": [
                {"query": "infant jacket"},
                {"query": "toddler snow boots"},
                {"query": "kids beanie"},
                {"query": "youth hiking boots"},
                {"query": "kids rainboots"}
            ]
        }

        for category, queries in scenarios.items():
            print(f"\n==========================================")
            print(f" CATEGORY: {category}")
            print(f"==========================================")
            
            for q_params in queries:
                q_text = q_params.get("query")
                size = q_params.get("size")
                price_max = q_params.get("price_max")
                
                filter_desc = []
                if size: filter_desc.append(f"size={size}")
                if price_max: filter_desc.append(f"price_max<=${price_max}")
                filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
                
                print(f"\n👉 Query: '{q_text}'{filter_str}")
                
                results = engine.search(
                    query=q_text,
                    size=size,
                    price_max=price_max,
                    top_k=2
                )
                
                if not results:
                    print("   ❌ No matching products found.")
                else:
                    for i, res in enumerate(results):
                        avail_variants = [f"{v['color']}/{v['size']}" for v in res["variants"][:3]]
                        variant_suffix = f" (Sample stock options: {', '.join(avail_variants)})" if avail_variants else " (No matching variants)"
                        print(f"   {i+1}. [{res['family_id']}] {res['title']} - Price range: ${res['price_min']:.2f} - ${res['price_max']:.2f}")
                        print(f"      Description: {res['description'][:100]}...")
                        print(f"      Variants found: {len(res['variants'])} items{variant_suffix}")

    print("\n===============================")
    print("VERIFICATION COMPLETED!")
    print("===============================")

if __name__ == "__main__":
    run_tests()
