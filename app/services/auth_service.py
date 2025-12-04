"""Authentication helpers."""
from fastapi import HTTPException, status

from app.models.user_model import Token
from app.services import user_service
from app.utils.security import create_access_token, verify_password


async def signup_user(email: str, password: str, first_name: str | None, last_name: str | None) -> Token:
    existing = await user_service.get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")
    user = await user_service.create_user(email, password, first_name, last_name, None)  # No preferences at signup
    # Use user ID (UUID) as token subject
    token = create_access_token(subject=str(user["id"]))
    return Token(access_token=token)


async def login_user(email: str, password: str) -> Token:
    user = await user_service.get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    # Use user ID (UUID) as token subject
    token = create_access_token(subject=str(user["id"]))
    return Token(access_token=token)
