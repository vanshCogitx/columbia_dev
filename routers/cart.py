import json

from fastapi import APIRouter, Body, Depends

import db
from models import (
    CartSaveRequest, CartResponse, WishlistSaveRequest, WishlistResponse,
)

router = APIRouter()


# --- Cart & Wishlist Endpoints (whole-list replace, keyed by email) ---

async def _get_saved_items(table: str, email: str) -> dict:
    row = await db.db_pool.fetchrow(f"SELECT items, updated_at FROM {table} WHERE email = $1", email)
    if not row:
        return {"email": email, "items": [], "updated_at": None}
    return {"email": email, "items": json.loads(row["items"]), "updated_at": row["updated_at"]}

async def _save_items(table: str, email: str, items: list) -> dict:
    await db.db_pool.execute(
        f"INSERT INTO {table} (email, items, updated_at) VALUES ($1, $2::jsonb, now()) "
        f"ON CONFLICT (email) DO UPDATE SET items = EXCLUDED.items, updated_at = EXCLUDED.updated_at",
        email, json.dumps(items),
    )
    return await _get_saved_items(table, email)

async def _remove_item(table: str, email: str, item_id: str) -> dict:
    current = await _get_saved_items(table, email)
    filtered = [item for item in current["items"] if item.get("id") != item_id]
    return await _save_items(table, email, filtered)


@router.get(
    "/api/cart",
    response_model=CartResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["cart"],
    summary="Get cart",
)
async def get_cart(email: str):
    return await _get_saved_items("carts", email.strip().lower())


@router.put(
    "/api/cart",
    response_model=CartResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["cart"],
    summary="Save cart",
    description="Replaces the entire cart with the given items list.",
)
async def save_cart(body: CartSaveRequest = Body(...)):
    return await _save_items("carts", body.email.strip().lower(), [item.model_dump() for item in body.items])


@router.delete(
    "/api/cart/items/{item_id}",
    response_model=CartResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["cart"],
    summary="Delete cart item",
)
async def delete_cart_item(item_id: str, email: str):
    return await _remove_item("carts", email.strip().lower(), item_id)


@router.delete(
    "/api/cart",
    response_model=CartResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["cart"],
    summary="Clear cart",
)
async def clear_cart(email: str):
    return await _save_items("carts", email.strip().lower(), [])


@router.get(
    "/api/wishlist",
    response_model=WishlistResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["wishlist"],
    summary="Get wishlist",
)
async def get_wishlist(email: str):
    return await _get_saved_items("wishlists", email.strip().lower())


@router.put(
    "/api/wishlist",
    response_model=WishlistResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["wishlist"],
    summary="Save wishlist",
    description="Replaces the entire wishlist with the given items list.",
)
async def save_wishlist(body: WishlistSaveRequest = Body(...)):
    return await _save_items("wishlists", body.email.strip().lower(), [item.model_dump() for item in body.items])


@router.delete(
    "/api/wishlist/items/{item_id}",
    response_model=WishlistResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["wishlist"],
    summary="Delete wishlist item",
)
async def delete_wishlist_item(item_id: str, email: str):
    return await _remove_item("wishlists", email.strip().lower(), item_id)


@router.delete(
    "/api/wishlist",
    response_model=WishlistResponse,
    dependencies=[Depends(db.verify_api_key)],
    tags=["wishlist"],
    summary="Clear wishlist",
)
async def clear_wishlist(email: str):
    return await _save_items("wishlists", email.strip().lower(), [])
