import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from csv_loader import load_csvs

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


def setup_postgres(df_family, df_catalog):
    """Sets up Supabase Postgres tables and populates them with inventory data."""
    if not SUPABASE_DB_URL:
        print("ERROR: SUPABASE_DB_URL is not set in the environment or .env file.")
        sys.exit(1)

    print("Setting up Postgres (Supabase) product tables...")
    engine = create_engine(SUPABASE_DB_URL)

    with engine.begin() as conn:
        # 1. Drop existing tables
        conn.execute(text("DROP TABLE IF EXISTS product_catalog"))
        conn.execute(text("DROP TABLE IF EXISTS product_family"))

        # 2. Create tables
        conn.execute(text("""
        CREATE TABLE product_family (
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
        """))

        conn.execute(text("""
        CREATE TABLE product_catalog (
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
        """))

    # 3. Insert data using Pandas
    family_cols = [
        'family_id', 'title', 'description', 'category_level_1', 'category_level_2',
        'product_type', 'sport', 'available_colors', 'available_sizes', 'available_materials',
        'available_features', 'available_fits', 'price_min', 'price_max', 'primary_product_id',
        'thumbnail_image', 'tags'
    ]
    df_family[family_cols].to_sql('product_family', engine, if_exists='append', index=False)

    catalog_cols = [
        'product_id', 'name', 'price', 'category_level_1', 'category_level_2',
        'product_type', 'sport', 'description', 'url', 'image_url', 'handle', 'tags',
        'color', 'material', 'fit', 'features', 'size', 'stock_quantity', 'family_id'
    ]
    df_catalog[catalog_cols].to_sql('product_catalog', engine, if_exists='append', index=False)

    # 4. Create indexes for fast joins and lookups
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX idx_catalog_product_id ON product_catalog(product_id)"))
        conn.execute(text("CREATE INDEX idx_catalog_family ON product_catalog(family_id)"))
        conn.execute(text("CREATE INDEX idx_catalog_size ON product_catalog(size)"))
        conn.execute(text("CREATE INDEX idx_catalog_color ON product_catalog(color)"))
        conn.execute(text("CREATE INDEX idx_catalog_price ON product_catalog(price)"))

    engine.dispose()
    print("✓ Postgres (Supabase) database setup completed successfully.")


def main():
    print("=== Loading Product Family & Catalog into Supabase ===")
    df_family, df_catalog = load_csvs()
    setup_postgres(df_family, df_catalog)
    print("\n==========================================")
    print("SUPABASE INGESTION COMPLETED SUCCESSFULLY!")
    print(f"Loaded {len(df_family)} product families and {len(df_catalog)} catalog variants.")
    print("==========================================")


if __name__ == "__main__":
    main()
