import os
import re
import time
import random
import logging

import jwt
from azure.communication.email import EmailClient

logger = logging.getLogger("columbia_backend.otp_auth")

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
OTP_TTL_SECONDS = 300  # 5 minutes, matches the reference NestJS implementation

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

_DURATION_RE = re.compile(r"^(\d+)([smhd])$", re.IGNORECASE)
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))


def parse_duration_to_seconds(value: str, default_seconds: int = 86400) -> int:
    """Parses '24h' / '5m' / '300s' / '1d' (or a plain integer string) into seconds."""
    if not value:
        return default_seconds
    match = _DURATION_RE.match(value.strip())
    if match:
        amount, unit = match.groups()
        return int(amount) * _DURATION_UNITS[unit.lower()]
    try:
        return int(value)
    except ValueError:
        return default_seconds


def generate_otp() -> str:
    return str(random.randint(100000, 999999))


def create_jwt(user_id: int, email: str, expires_in_seconds: int) -> str:
    now = int(time.time())
    payload = {"sub": str(user_id), "email": email, "iat": now, "exp": now + expires_in_seconds}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> dict:
    """Raises jwt.PyJWTError (ExpiredSignatureError, InvalidTokenError, ...) on failure."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def send_otp_email(otp: str, to_address: str) -> None:
    connection_string = os.getenv("AZURE_EMAIL_CONNECTION_STRING")
    sender_address = os.getenv("AZURE_EMAIL_SENDER_ADDRESS")
    if not connection_string or not sender_address:
        raise RuntimeError("Azure email is not configured (AZURE_EMAIL_CONNECTION_STRING / AZURE_EMAIL_SENDER_ADDRESS).")

    client = EmailClient.from_connection_string(connection_string)
    message = {
        "senderAddress": sender_address,
        "content": {
            "subject": "Your Login OTP Code",
            "plainText": f"Your OTP code is: {otp}\nThis code will expire in 5 minutes.\n\nIf you didn't request this, please ignore this email.",
            "html": f"""
                <html>
                  <body>
                    <h2>Your Login OTP Code</h2>
                    <p>Your OTP code is: <strong style="font-size: 20px;">{otp}</strong></p>
                    <p>This code will expire in 5 minutes.</p>
                    <p>If you didn't request this, please ignore this email.</p>
                  </body>
                </html>
            """,
        },
        "recipients": {"to": [{"address": to_address}]},
    }
    poller = client.begin_send(message)
    poller.result()
    logger.info("OTP email sent to %s", to_address)
