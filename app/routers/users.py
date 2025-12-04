"""User endpoints."""
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.db.connection import get_pool
from app.models.user_model import PreferencesUpdate, User
from app.services import embedding_service, user_service
from app.utils.security import decode_access_token

router = APIRouter()
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and validate JWT token from Authorization header."""
    token = credentials.credentials
    user_id_str = decode_access_token(token)
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # The token subject is the user ID (UUID string)
    from uuid import UUID
    try:
        user_id = UUID(user_id_str)
        user = await user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user
    except ValueError:
        # Fallback: if token was created with email (for backward compatibility)
        user = await user_service.get_user_by_email(user_id_str)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user


@router.get("/me", response_model=User)
async def read_current_user(current_user=Depends(get_current_user)):
    """Get current authenticated user's profile."""
    return User.from_db_record(current_user)


@router.put("/me/preferences", response_model=dict)
async def update_preferences(
    preferences: PreferencesUpdate,
    current_user=Depends(get_current_user)
):
    """Update user preferences/KYC data.
    
    This can be called after signup to set preferences, or later to update them.
    Preferences are used for content-based recommendations.
    """
    # Convert Pydantic model to dict, excluding None values
    prefs_dict = preferences.model_dump(exclude_none=True)
    
    # Map reading_preferences to description for backward compatibility
    if "reading_preferences" in prefs_dict and "description" not in prefs_dict:
        prefs_dict["description"] = prefs_dict.pop("reading_preferences")
    elif "reading_preferences" in prefs_dict:
        # If both exist, prefer description
        prefs_dict.pop("reading_preferences")
    
    # Update preferences in database
    user = await user_service.update_user_preferences(current_user["id"], prefs_dict)
    
    # Generate embedding from preferences if they exist
    if prefs_dict:
        embedding = embedding_service.encode_kyc_preferences(prefs_dict)
        if embedding:
            # Update the embedding in the database
            # Convert list to JSON string for JSONB field
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE users SET kyc_embedding = $1::jsonb WHERE id = $2",
                    json.dumps(embedding),
                    current_user["id"]
                )
    
    return {
        "message": "Preferences updated successfully",
        "preferences": prefs_dict
    }


@router.get("/{user_id}", response_model=User)
async def get_user(user_id: str):
    """Get user by ID (public endpoint)."""
    from uuid import UUID
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user ID format")
    
    user = await user_service.get_user_by_id(user_uuid)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return User.from_db_record(user)
