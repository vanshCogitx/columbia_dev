import os
import re
import time
import random
import base64
import logging

import jwt
from azure.communication.email import EmailClient

logger = logging.getLogger("columbia_backend.otp_auth")

# Embedded as an inline attachment (see _otp_email_html's cid: reference)
# rather than linked as a remote <img src="https://...">. A remote URL gets
# silently blocked by default in Outlook (and similar clients) for any
# sender not yet in the recipient's Safe Senders list — every brand-new
# user's first OTP email would show a broken image + a "content blocked"
# banner. Inline/CID embedding ships the image bytes as part of the email
# itself, so there's nothing to fetch and nothing to block. Read once at
# import time, not per-send.
_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
with open(_LOGO_PATH, "rb") as _f:
    _LOGO_BASE64 = base64.b64encode(_f.read()).decode("ascii")

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


def _otp_email_html(otp: str) -> str:
    """Matches the CogitX brand template exactly (dark card, purple accent
    bar, spaced-out code digits) — all styling is inline since email clients
    strip <style> blocks and CSS classes unreliably."""
    digits_spaced = " ".join(otp)
    return f"""\
<html>
  <body style="margin:0; padding:32px 16px; background-color:#0d0d12; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
    <div style="max-width:520px; margin:0 auto; background-color:#3a3a45; border:1px solid #55556a; border-radius:16px; overflow:hidden;">
      <div style="background-color:#3a3a45; padding:28px 32px 20px;">
        <img src="cid:cogitx-logo" alt="CogitX" style="height:32px; display:block;" />
      </div>
      <div style="height:4px; background-color:#7c3aed;"></div>
      <div style="background-color:#0d0d12; padding:32px;">
        <div style="color:#a78bfa; font-size:12px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:10px;">
          Login Verification
        </div>
        <div style="color:#ffffff; font-size:28px; font-weight:700; margin-bottom:18px;">
          Your login code
        </div>
        <p style="color:#a3a3b0; font-size:15px; line-height:1.5; margin:0 0 28px;">
          Use this code to log in to your CogitX Platform account. It expires in 5 minutes.
        </p>
        <div style="background-color:#3a3a45; border-radius:12px; padding:24px; text-align:center; margin-bottom:28px;">
          <div style="color:#a3a3b0; font-size:12px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:12px;">
            Login Code
          </div>
          <div style="color:#ffffff; font-size:36px; font-weight:700; letter-spacing:0.15em; background-color:#55556a; display:inline-block; padding:8px 20px; border-radius:6px;">
            {digits_spaced}
          </div>
        </div>
        <p style="color:#a3a3b0; font-size:13px; line-height:1.5; margin:0;">
          If you did not request this, you can ignore this email. Someone else may have typed your address by mistake.
        </p>
      </div>
      <div style="background-color:#3a3a45; border-top:1px solid #55556a; padding:18px 32px;">
        <p style="color:#a3a3b0; font-size:12px; margin:0;">
          &copy; 2026 CogitX AI. All rights reserved.
        </p>
      </div>
    </div>
  </body>
</html>
"""


def send_otp_email(otp: str, to_address: str) -> None:
    connection_string = os.getenv("AZURE_EMAIL_CONNECTION_STRING")
    sender_address = os.getenv("AZURE_EMAIL_SENDER_ADDRESS")
    if not connection_string or not sender_address:
        raise RuntimeError("Azure email is not configured (AZURE_EMAIL_CONNECTION_STRING / AZURE_EMAIL_SENDER_ADDRESS).")

    client = EmailClient.from_connection_string(connection_string)
    message = {
        "senderAddress": sender_address,
        "content": {
            "subject": "Your CogitX Platform login code",
            "plainText": f"Your login code is: {otp}\nThis code will expire in 5 minutes.\n\nIf you didn't request this, please ignore this email.",
            "html": _otp_email_html(otp),
        },
        "recipients": {"to": [{"address": to_address}]},
        # contentId "cogitx-logo" matches the html's <img src="cid:cogitx-logo">
        # above — this is what makes it an inline image instead of a regular
        # (attachment-list) attachment.
        "attachments": [{
            "name": "logo.png",
            "contentType": "image/png",
            "contentInBase64": _LOGO_BASE64,
            "contentId": "cogitx-logo",
        }],
    }
    poller = client.begin_send(message)
    poller.result()
    logger.info("OTP email sent to %s", to_address)
