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
    id: int
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

class FeedbackRequest(BaseModel):
    # le=2147483647 is Postgres's int4 max — messages.id is a SERIAL (int4)
    # column. Without this bound, a caller sending a JS-style millisecond
    # timestamp instead of a real message id (a real, observed frontend bug)
    # sails through validation and crashes asyncpg with an unhandled
    # OverflowError deep in the DB driver instead of failing cleanly here.
    message_id: int = Field(..., ge=1, le=2147483647, description="ID of the AI message being rated (from ChatHistoryMessage, not exposed today — see note in the endpoint).")
    rating: str = Field(..., description="'up' or 'down'.")
    reason: Optional[str] = Field(None, description="Optional free-text reason, mainly useful for 'down' ratings.")

class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    rating: str

# --- Evals Dashboard Schemas (gated behind the dashboard's own login — see routers/evals.py) ---

class EvalsFeedItem(BaseModel):
    feedback_id: int
    created_at: str
    rating: str
    reason: Optional[str] = None
    query: Optional[str] = None
    response_snippet: Optional[str] = None
    judged: bool
    corrected_score: Optional[float] = None
    root_cause_node: Optional[str] = None
    error: Optional[str] = None

class EvalsFeedResponse(BaseModel):
    items: List[EvalsFeedItem]

class EvalsDetailResponse(BaseModel):
    feedback_id: int
    chat_id: int
    message_id: int
    rating: str
    reason: Optional[str] = None
    created_at: str
    query: Optional[str] = None
    response: Optional[str] = None
    judged: bool
    judged_at: Optional[str] = None
    initial_score: Optional[float] = None
    corrected_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    root_cause_node: Optional[str] = None
    root_cause_snippet: Optional[str] = None
    root_cause_explanation: Optional[str] = None
    draft_thinking: Optional[str] = None
    critique_thinking: Optional[str] = None
    attribution_thinking: Optional[str] = None
    error: Optional[str] = None

class EvalsRootCauseCount(BaseModel):
    root_cause_node: str
    count: int

class EvalsStatsResponse(BaseModel):
    total_feedback: int
    up_count: int
    down_count: int
    avg_score: Optional[float] = None
    top_root_causes: List[EvalsRootCauseCount]

# --- HITL (Human-in-the-loop review) Schemas ---
# A second, independent opinion on a feedback event, alongside (not
# overriding) the LLM judge — see db.py's response_human_reviews comment.

class HitlFeedItem(BaseModel):
    feedback_id: int
    created_at: str
    rating: str
    reason: Optional[str] = None
    query: Optional[str] = None
    response_snippet: Optional[str] = None
    reviewed: bool
    reason_valid: Optional[bool] = None
    score: Optional[float] = None

class HitlFeedResponse(BaseModel):
    items: List[HitlFeedItem]

class HitlDetailResponse(BaseModel):
    feedback_id: int
    chat_id: int
    message_id: int
    rating: str
    reason: Optional[str] = None
    created_at: str
    query: Optional[str] = None
    response: Optional[str] = None
    reviewed: bool
    reviewed_at: Optional[str] = None
    reviewer_name: Optional[str] = None
    reason_valid: Optional[bool] = None
    score: Optional[float] = None
    notes: Optional[str] = None

class HitlReviewRequest(BaseModel):
    reason_valid: bool
    score: Optional[float] = Field(None, ge=0, le=1)
    notes: Optional[str] = None

class HitlReviewResponse(BaseModel):
    feedback_id: int
    reason_valid: bool
    score: Optional[float] = None
    notes: Optional[str] = None
    reviewed_at: str

# --- Hybrid (LLM + human side-by-side sign-off) Schemas ---
# Only ever meaningful for a feedback event that has both an LLM verdict
# (response_judgments) and a HITL review (response_human_reviews) — see
# routers/hybrid.py's feed query. Approving is a lightweight sign-off, not a
# computed blended score — see db.py's hybrid_approvals comment.

class HybridFeedItem(BaseModel):
    feedback_id: int
    created_at: str
    rating: str
    query: Optional[str] = None
    response_snippet: Optional[str] = None
    llm_score: Optional[float] = None
    human_score: Optional[float] = None
    human_reason_valid: Optional[bool] = None
    approved: bool

class HybridFeedResponse(BaseModel):
    items: List[HybridFeedItem]

class HybridDetailResponse(BaseModel):
    feedback_id: int
    chat_id: int
    message_id: int
    rating: str
    reason: Optional[str] = None
    created_at: str
    query: Optional[str] = None
    response: Optional[str] = None
    llm_score: Optional[float] = None
    llm_reasoning: Optional[str] = None
    root_cause_node: Optional[str] = None
    human_score: Optional[float] = None
    human_reason_valid: Optional[bool] = None
    human_notes: Optional[str] = None
    human_reviewer_name: Optional[str] = None
    approved: bool
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

class HybridApproveResponse(BaseModel):
    feedback_id: int
    approved_by: str
    approved_at: str

# --- Traceability Schemas ---
# Execution trace (which workflow nodes ran, and each one's raw output) for
# a real chat query — same underlying data (message_traces, saved at
# generation time) and same TraceTree.jsx component already used by Testing,
# just for real feedback-attached queries instead of simulated turns. Scoped
# to response_feedback (all ratings, not just thumbs-down) because chats/
# messages carry no project_id of their own — response_feedback.project_id
# is the only place a query is ever attributed to a project at all.

class TraceabilityFeedItem(BaseModel):
    feedback_id: int
    created_at: str
    rating: str
    query: Optional[str] = None
    response_snippet: Optional[str] = None
    has_trace: bool
    node_count: Optional[int] = None

class TraceabilityFeedResponse(BaseModel):
    items: List[TraceabilityFeedItem]

class TraceabilityDetailResponse(BaseModel):
    feedback_id: int
    rating: str
    created_at: str
    query: Optional[str] = None
    response: Optional[str] = None
    has_trace: bool
    raw_output: Optional[dict] = None
    execution_summary: Optional[dict] = None

# --- Testing (Simulation) Schemas ---
# Proactive scenario testing: an LLM plays a simulated shopper against the
# real Cartesian workflow for several turns, scored against success criteria.
# Kept fully separate from the reactive Evals schemas above (see
# simulation_scenarios/simulation_runs/simulation_turns/
# simulation_turn_feedback/simulation_turn_judgments in db.py).

class ScenarioCreateRequest(BaseModel):
    name: str = Field(..., max_length=200)
    user_scenario: str = Field(..., description="Free-text description of the simulated shopper's persona/goal.")
    success_criteria: List[str] = Field(..., min_length=1, description="One or more criteria the conversation is judged against.")
    max_turns: int = Field(5, ge=1, le=20)
    mock_tools: bool = Field(False, description="Stored for UI parity with the reference design; not wired to any backend behavior yet.")

class ScenarioResponse(BaseModel):
    id: int
    name: str
    user_scenario: str
    success_criteria: List[str]
    max_turns: int
    mock_tools: bool
    created_at: str
    last_run_status: Optional[str] = None
    last_run_verdict: Optional[str] = None

class ScenarioListResponse(BaseModel):
    items: List[ScenarioResponse]

class RunTriggerResponse(BaseModel):
    run_id: int
    status: str

class SimulationTurnItem(BaseModel):
    id: int
    turn_index: int
    simulated_user_text: str
    ai_response_text: Optional[str] = None
    raw_output: Optional[dict] = None
    feedback_id: Optional[int] = None
    feedback_rating: Optional[str] = None
    judged: bool = False
    corrected_score: Optional[float] = None
    root_cause_node: Optional[str] = None

class RunListItem(BaseModel):
    run_id: int
    scenario_id: int
    scenario_name: str
    status: str
    turn_count: int
    verdict: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None

class RunListResponse(BaseModel):
    items: List[RunListItem]

class RunDetailResponse(BaseModel):
    run_id: int
    scenario_id: int
    scenario_name: str
    user_scenario: str
    success_criteria: List[str]
    max_turns: int
    status: str
    turn_count: int
    verdict: Optional[str] = None
    verdict_reasoning: Optional[str] = None
    verdict_thinking: Optional[str] = None
    error: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
    turns: List[SimulationTurnItem]

class TurnFeedbackRequest(BaseModel):
    rating: str = Field(..., description="'up' or 'down'.")
    reason: Optional[str] = None

class TurnFeedbackResponse(BaseModel):
    id: int
    turn_id: int
    rating: str

class TurnJudgmentDetailResponse(BaseModel):
    """Judge-only fields for one simulated-turn feedback event — mirrors the
    judge-specific subset of EvalsDetailResponse, so the frontend's shared
    judge-verdict panel can render either source with the same component."""
    turn_feedback_id: int
    judged: bool
    initial_score: Optional[float] = None
    corrected_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    root_cause_node: Optional[str] = None
    root_cause_snippet: Optional[str] = None
    root_cause_explanation: Optional[str] = None
    draft_thinking: Optional[str] = None
    critique_thinking: Optional[str] = None
    attribution_thinking: Optional[str] = None
    error: Optional[str] = None

class WorkflowGraphNode(BaseModel):
    node_id: str
    node_type: str
    alias: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None

class WorkflowGraphEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    source_handle: Optional[str] = None

class WorkflowGraphResponse(BaseModel):
    nodes: List[WorkflowGraphNode]
    edges: List[WorkflowGraphEdge]

# --- Projects (evals dashboard multi-project support) ---
# Never includes cartesian_client_id/cartesian_client_secret in any response.

class ProjectResponse(BaseModel):
    id: int
    name: str
    cartesian_export_id: str
    cartesian_base_url: Optional[str] = None
    is_live: bool = False
    created_at: str
    node_count: int = 0
    edge_count: int = 0

class ProjectListResponse(BaseModel):
    items: List[ProjectResponse]

class SessionProductsResponse(BaseModel):
    session_id: str
    products: List[dict]

# --- Cart & Wishlist Schemas ---

class CartItem(BaseModel):
    id: str
    name: str = Field(max_length=500)
    price: float
    originalPrice: Optional[float] = None  # defaults to price (no discount shown) when not sent — AI-recommended products don't carry this
    discount: float = 0
    description: str = Field(max_length=5000)
    image: str = Field(max_length=2000)
    quantity: int = 1

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
