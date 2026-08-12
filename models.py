from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

# --- Auth & Chat Schemas ---

class SendOtpRequest(BaseModel):
    email: str

class SendOtpResponse(BaseModel):
    success: bool
    message: str
    email: str
    otp: Optional[str] = None  # only populated when ENVIRONMENT=development

class VerifyOtpRequest(BaseModel):
    email: str
    otp: str = Field(..., min_length=6, max_length=6)

class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str] = None

class VerifyOtpResponse(BaseModel):
    success: bool
    message: str
    token: str
    user: UserResponse

class LogoutResponse(BaseModel):
    success: bool
    message: str

class ChatResponseModel(BaseModel):
    id: int
    session_id: str
    title: str
    created_at: str

class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    content_type: Optional[str] = None
    structured_data: Optional[list] = None
    session_id: Optional[str] = None
    image_url: Optional[str] = None
    created_at: str

class ChatHistoryResponse(BaseModel):
    email: str
    messages: List[ChatHistoryMessage]
    pending: Optional[bool] = None
    pending_status: Optional[str] = None

class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Existing chat's session_id. Omit to start a new chat.")
    text: str

class SessionProductsResponse(BaseModel):
    session_id: str
    products: List[dict]

# --- Cart & Wishlist Schemas ---

class CartItem(BaseModel):
    id: str
    name: str = Field(max_length=500)
    price: float
    originalPrice: float
    discount: float
    description: str = Field(max_length=5000)
    image: str = Field(max_length=2000)
    quantity: int

class CartSaveRequest(BaseModel):
    email: str
    items: List[CartItem] = Field(max_length=500)

class CartResponse(BaseModel):
    email: str
    items: List[CartItem]
    updated_at: Optional[datetime] = None

class WishlistItem(BaseModel):
    id: str
    name: str = Field(max_length=500)
    price: float
    originalPrice: float
    discount: float
    description: str = Field(max_length=5000)
    image: str = Field(max_length=2000)
    quantity: int = 0

class WishlistSaveRequest(BaseModel):
    email: str
    items: List[WishlistItem] = Field(max_length=500)

class WishlistResponse(BaseModel):
    email: str
    items: List[WishlistItem]
    updated_at: Optional[datetime] = None

# --- Pydantic Schema Definitions ---

class SearchRequest(BaseModel):
    query: Optional[str] = Field(
        None,
        description=(
            "A natural language search query. E.g., 'blue windbreaker jacket', 'waterproof hiking pants', "
            "'warm fleece'. If provided, results are ranked by semantic relevance to this query. "
            "If omitted, products are returned based on structured filters alone."
        )
    )
    family_id: Optional[str] = Field(
        None,
        description="A specific product model family ID (e.g. 'F000001'). If supplied, performs a direct lookup for that product family and its variants."
    )
    category_level_1: Optional[str] = Field(
        None,
        description="The primary department or category level (e.g., 'Apparel', 'Footwear', 'Equipment')."
    )
    category_level_2: Optional[str] = Field(
        None,
        description="The secondary category, department, or demographic (e.g., 'Men\'s', 'Women\'s', 'Kids', 'Unisex')."
    )
    product_type: Optional[str] = Field(
        None,
        description="The specific product type classification (e.g., 'Jackets', 'Pants', 'Shirts', 'Shoes')."
    )
    sport: Optional[str] = Field(
        None,
        description="The specific sport or activity intended for this item (e.g., 'Hiking', 'Casual', 'Running', 'Training')."
    )
    price_min: Optional[float] = Field(
        None,
        description="Filter products with price greater than or equal to this value."
    )
    price_max: Optional[float] = Field(
        None,
        description="Filter products with price less than or equal to this value."
    )
    color: Optional[str] = Field(
        None,
        description="Exact color filter (e.g., 'Maroon', 'Black', 'Grey')."
    )
    size: Optional[str] = Field(
        None,
        description="Exact size filter (e.g., 'S', 'M', 'L', 'XL', 'XXS')."
    )
    in_stock_only: bool = Field(
        True,
        description="If True, only returns items that are currently in stock and available for purchase."
    )
    top_k: int = Field(
        10,
        description="The maximum number of matched products to return."
    )

class VariantResult(BaseModel):
    product_id: str = Field(..., description="Unique variant product ID.")
    name: str = Field(..., description="The name of this variant.")
    price: float = Field(..., description="The price of this variant.")
    size: str = Field(..., description="The size of this variant.")
    color: str = Field(..., description="The color of this variant.")
    stock_quantity: int = Field(..., description="Stock count remaining.")
    availability: str = Field(..., description="Stock availability ('in_stock' or 'out_of_stock').")
    url: str = Field(..., description="Product detail URL.")
    image_url: str = Field(..., description="Product variant image URL.")
    handle: str = Field(..., description="Product variant URL handle.")
    tags: str = Field(..., description="Tags associated with this variant.")

class ProductFamilyResult(BaseModel):
    family_id: str = Field(..., description="Unique product model family identifier.")
    title: str = Field(..., description="General title of the product model.")
    description: str = Field(..., description="General description of the product model.")
    category_level_1: str = Field(..., description="Primary department/category.")
    category_level_2: str = Field(..., description="Secondary category/demographic.")
    product_type: str = Field(..., description="Product type (e.g. Jackets).")
    sport: str = Field(..., description="Sport category.")
    available_colors: str = Field(..., description="JSON array representation of available colors.")
    available_sizes: str = Field(..., description="JSON array representation of available sizes.")
    available_materials: str = Field(..., description="JSON array representation of materials.")
    available_features: str = Field(..., description="JSON array representation of features.")
    available_fits: str = Field(..., description="JSON array representation of fits.")
    price_min: float = Field(..., description="Minimum price of variants under this family.")
    price_max: float = Field(..., description="Maximum price of variants under this family.")
    primary_product_id: str = Field(..., description="Main product ID used as the family reference.")
    thumbnail_image: str = Field(..., description="URL of the main/primary product image.")
    tags: str = Field(..., description="Tags associated with this product family.")
    variants: List[VariantResult] = Field([], description="List of matching inventory variants (sizes/colors) under this family.")
    score: float = Field(..., description="Relevance score (1.0 = exact or default match, < 1.0 = semantic similarity score).")

class CatalogFacetsResponse(BaseModel):
    category_level_1: List[str] = Field(..., description="List of unique primary categories.")
    category_level_2: List[str] = Field(..., description="List of unique secondary categories.")
    product_type: List[str] = Field(..., description="List of unique product types.")
    sport: List[str] = Field(..., description="List of unique sport categories.")
    brands: List[str] = Field(..., description="List of unique brands in the inventory.")
    price_min: float = Field(..., description="Minimum product price present in the catalog.")
    price_max: float = Field(..., description="Maximum product price present in the catalog.")

class CartesianChatRequest(BaseModel):
    text: str = Field(..., description="User input text")
    wait_seconds: Optional[int] = Field(30, description="Optional override for sync wait duration in seconds (max 30 for inline wait)")
    session_id: Optional[str] = Field(None, description="Optional conversation session identifier. Reuse this across calls to continue context.")
    user_id: Optional[str] = Field(None, description="Identifier for the requesting user, required by the workflow's input schema.")

# --- V2 Flat Catalog Schemas ---

class CatalogSearchRequestV2(BaseModel):
    query: str = Field(
        ...,
        description=(
            "A natural language search query describing what the user is looking for. "
            "E.g., 'waterproof hiking jacket in blue', 'kids fleece jacket', 'trekking poles'. "
            "This field is required — all retrieval is semantic."
        )
    )
    category_level_1: Optional[str] = Field(
        None,
        description="Primary department or category filter (e.g., 'Apparel', 'Accessories', 'Bags & Gear')."
    )
    category_level_2: Optional[str] = Field(
        None,
        description="Secondary category / demographic filter (e.g., \"Men's\", \"Women's\", 'Kids', 'Unisex')."
    )
    product_type: Optional[str] = Field(
        None,
        description="Product type filter (e.g., 'Sportswear', 'Equipment', 'Accessories')."
    )
    sport: Optional[str] = Field(
        None,
        description="Sport or activity filter (e.g., 'Hiking', 'Casual', 'Running')."
    )
    price_min: Optional[float] = Field(
        None,
        description="Minimum price filter (inclusive). Products priced below this value are excluded."
    )
    price_max: Optional[float] = Field(
        None,
        description="Maximum price filter (inclusive). Products priced above this value are excluded."
    )
    color: Optional[str] = Field(
        None,
        description="Exact color filter (e.g., 'Black', 'Blue', 'Maroon')."
    )
    size: Optional[str] = Field(
        None,
        description="Exact size filter (e.g., 'S', 'M', 'L', 'XL', 'XXS', 'OS')."
    )
    top_k: int = Field(
        10,
        description="Maximum number of products to return (default 10)."
    )

class CatalogSearchRequestV3(BaseModel):
    queries: List[CatalogSearchRequestV2] = Field(
        ...,
        min_length=1,
        description=(
            "A batch of independent search queries, each supporting the same fields as "
            "search-products-v2 (natural language 'query' plus optional structured filters). "
            "Queries run concurrently and independently."
        )
    )

class CatalogProductResultV2(BaseModel):
    product_id: str = Field(..., description="Unique SKU / product ID.")
    name: str = Field(..., description="Full product name.")
    price: float = Field(..., description="Product price.")
    category_level_1: str = Field(..., description="Primary department/category.")
    category_level_2: str = Field(..., description="Secondary category/demographic.")
    product_type: str = Field(..., description="Product type classification.")
    sport: str = Field(..., description="Sport or activity.")
    color: str = Field(..., description="Product color.")
    material: str = Field(..., description="Product material.")
    fit: str = Field(..., description="Fit style (e.g., Regular, Slim, Comfort Stretch).")
    features: str = Field(..., description="Key product features.")
    size: str = Field(..., description="Available size for this SKU.")
    stock_quantity: int = Field(..., description="Units currently in stock.")
    availability: str = Field(..., description="'in_stock' or 'out_of_stock'.")
    url: str = Field(..., description="Product detail page URL.")
    image_url: str = Field(..., description="Product image URL.")
    description_snippet: str = Field(..., description="Short description excerpt.")
    score: float = Field(..., description="Semantic similarity score (cosine, 0–1).")
