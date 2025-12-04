"""Authentication endpoints."""
from fastapi import APIRouter, status

from app.models.user_model import Token, UserCreate, UserLogin
from app.services import auth_service

router = APIRouter()


@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserCreate):
    """Create a new user and return an access token.
    
    Note: Preferences can be set later via PUT /users/me/preferences
    """
    return await auth_service.signup_user(
        email=payload.email,
        password=payload.password,
        first_name=payload.first_name,
        last_name=payload.last_name,
    )


@router.post("/login", response_model=Token)
async def login(payload: UserLogin):
    """Login existing user and return an access token."""
    return await auth_service.login_user(email=payload.email, password=payload.password)
