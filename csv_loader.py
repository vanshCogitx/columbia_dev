import os
import sys
import pandas as pd

FAMILY_CSV = "data/product_family 3(Product Family) (1).csv"
CATALOG_CSV = "data/product_catalog (2) 3(Product Catalog) (1).csv"


def clean_dataframe(df: pd.DataFrame, is_catalog: bool = False) -> pd.DataFrame:
    """Cleans columns, handling missing values and data type conversions."""
    df = df.copy()
    if is_catalog:
        df['product_id'] = df['product_id'].astype(str)
        df['family_id'] = df['family_id'].astype(str)
        df['name'] = df['name'].fillna("").astype(str)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0).astype(float)
        df['category_level_1'] = df['category_level_1'].fillna("Unspecified").astype(str)
        df['category_level_2'] = df['category_level_2'].fillna("Unspecified").astype(str)
        df['product_type'] = df['product_type'].fillna("Unspecified").astype(str)
        df['sport'] = df['sport'].fillna("Unspecified").astype(str)
        df['description'] = df['description'].fillna("").astype(str)
        df['url'] = df['url'].fillna("").astype(str)
        df['image_url'] = df['image_url'].fillna("").astype(str)
        df['handle'] = df['handle'].fillna("").astype(str)
        df['tags'] = df['tags'].fillna("").astype(str)
        df['color'] = df['color'].fillna("Unspecified").astype(str)
        df['material'] = df['material'].fillna("Unspecified").astype(str)
        df['fit'] = df['fit'].fillna("Unspecified").astype(str)
        df['features'] = df['features'].fillna("").astype(str)
        df['size'] = df['size'].fillna("Unspecified").astype(str)
        df['stock_quantity'] = pd.to_numeric(df['stock_quantity'], errors='coerce').fillna(0).astype(int)
    else:
        df['family_id'] = df['family_id'].astype(str)
        df['title'] = df['title'].fillna("").astype(str)
        df['description'] = df['description'].fillna("").astype(str)
        df['category_level_1'] = df['category_level_1'].fillna("Unspecified").astype(str)
        df['category_level_2'] = df['category_level_2'].fillna("Unspecified").astype(str)
        df['product_type'] = df['product_type'].fillna("Unspecified").astype(str)
        df['sport'] = df['sport'].fillna("Unspecified").astype(str)
        df['available_colors'] = df['available_colors'].fillna("[]").astype(str)
        df['available_sizes'] = df['available_sizes'].fillna("[]").astype(str)
        df['available_materials'] = df['available_materials'].fillna("[]").astype(str)
        df['available_features'] = df['available_features'].fillna("[]").astype(str)
        df['available_fits'] = df['available_fits'].fillna("[]").astype(str)
        df['price_min'] = pd.to_numeric(df['price_min'], errors='coerce').fillna(0.0).astype(float)
        df['price_max'] = pd.to_numeric(df['price_max'], errors='coerce').fillna(0.0).astype(float)
        df['primary_product_id'] = df['primary_product_id'].astype(str)
        df['thumbnail_image'] = df['thumbnail_image'].fillna("").astype(str)
        df['tags'] = df['tags'].fillna("[]").astype(str)
        df['search_text'] = df['search_text'].fillna("").astype(str)
    return df


def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads and cleans the product family/catalog CSVs. Exits the process on failure
    (both ingestion scripts run standalone, so a hard exit here is appropriate)."""
    for csv_file in [FAMILY_CSV, CATALOG_CSV]:
        if not os.path.exists(csv_file):
            print(f"ERROR: File not found: {csv_file}")
            sys.exit(1)

    print("Loading CSV files...")
    try:
        df_family = pd.read_csv(FAMILY_CSV, encoding='utf-8')
    except UnicodeDecodeError:
        df_family = pd.read_csv(FAMILY_CSV, encoding='cp1252')

    try:
        df_catalog = pd.read_csv(CATALOG_CSV, encoding='utf-8')
    except UnicodeDecodeError:
        df_catalog = pd.read_csv(CATALOG_CSV, encoding='cp1252')

    print(f"Loaded {len(df_family)} Product Families and {len(df_catalog)} Catalog Variants.")

    print("Cleaning and validating data...")
    df_family = clean_dataframe(df_family, is_catalog=False)
    df_catalog = clean_dataframe(df_catalog, is_catalog=True)

    return df_family, df_catalog
