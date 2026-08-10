import os
import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import jwt as pyjwt

import db
from otp import otp_auth
from models import (
    SendOtpRequest, SendOtpResponse, VerifyOtpRequest, VerifyOtpResponse,
    UserResponse, LogoutResponse,
)

logger = logging.getLogger("columbia_backend")

router = APIRouter()


# --- User Authentication Endpoints (email OTP) ---

@router.post(
    "/api/auth/send-otp",
    response_model=SendOtpResponse,
    tags=["auth"],
    summary="Send OTP to email",
    description="Generates a 6-digit OTP, stores it in Redis for 5 minutes, and emails it via Azure Communication Email."
)
async def send_otp(request: SendOtpRequest = Body(...)):
    email = request.email.strip().lower()
    if not otp_auth.is_valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    otp = otp_auth.generate_otp()
    await db.redis_client.setex(f"otp:{email}", otp_auth.OTP_TTL_SECONDS, otp)

    try:
        await asyncio.to_thread(otp_auth.send_otp_email, otp, email)
    except Exception as e:
        logger.error("send_otp: failed to send email to %s: %s", email, e)
        raise HTTPException(status_code=502, detail="Failed to send OTP email.")

    logger.info("send_otp: OTP sent to %s", email)
    response = {"success": True, "message": "OTP sent successfully", "email": email}
    if os.getenv("ENVIRONMENT", "").lower() == "development":
        response["otp"] = otp
    return response


@router.post(
    "/api/auth/verify-otp",
    response_model=VerifyOtpResponse,
    tags=["auth"],
    summary="Verify OTP and receive JWT token",
    description="Verifies the OTP, creates the user on first login if needed, and returns a signed JWT."
)
async def verify_otp(request: VerifyOtpRequest = Body(...)):
    email = request.email.strip().lower()
    stored_otp = await db.redis_client.get(f"otp:{email}")
    if not stored_otp or stored_otp != request.otp:
        raise HTTPException(status_code=401, detail="Invalid or expired OTP")
    await db.redis_client.delete(f"otp:{email}")

    user_row = await db.db_pool.fetchrow("SELECT id, email, name FROM users WHERE email = $1", email)
    if not user_row:
        default_name = email.split("@")[0]
        user_row = await db.db_pool.fetchrow(
            "INSERT INTO users (email, name) VALUES ($1, $2) RETURNING id, email, name",
            email, default_name,
        )
        logger.info("verify_otp: created new user for %s (id=%s)", email, user_row["id"])

    token = otp_auth.create_jwt(user_row["id"], user_row["email"], db.JWT_EXPIRATION_SECONDS)
    logger.info("verify_otp: OTP verified for %s (id=%s)", email, user_row["id"])

    return {
        "success": True,
        "message": "OTP verified successfully",
        "token": token,
        "user": {"id": user_row["id"], "email": user_row["email"], "name": user_row["name"]},
    }


@router.get(
    "/api/auth/me",
    response_model=UserResponse,
    tags=["auth"],
    summary="Get Current User Profile",
    description="Returns profile of the authenticated user."
)
async def get_me(current_user: dict = Depends(db.get_current_user)):
    return current_user


@router.post(
    "/api/auth/logout",
    response_model=LogoutResponse,
    tags=["auth"],
    summary="Logout and invalidate JWT token",
    description="Blocklists the current JWT in Redis until its natural expiry."
)
async def logout(credentials: HTTPAuthorizationCredentials = Depends(db.security_bearer)):
    token = credentials.credentials
    try:
        payload = otp_auth.decode_jwt(token)
    except pyjwt.PyJWTError:
        return {"success": True, "message": "Logged out successfully"}

    ttl = payload["exp"] - int(datetime.now(timezone.utc).timestamp())
    if ttl > 0:
        await db.redis_client.setex(f"blocklist:{token}", ttl, "1")
    return {"success": True, "message": "Logged out successfully"}
