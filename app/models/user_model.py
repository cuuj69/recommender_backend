"""User models."""
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, computed_field


class UserBase(BaseModel):
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserCreate(UserBase):
    password: str
    # Note: preferences are set separately via /users/me/preferences endpoint


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class User(BaseModel):
    """User response model - only returns full_name, not first_name/last_name separately."""
    id: UUID
    email: EmailStr
    full_name: Optional[str] = None
    is_admin: Optional[bool] = False

    model_config = {"from_attributes": True}
    
    @classmethod
    def from_db_record(cls, record: dict):
        """Create User from database record."""
        first_name = record.get("first_name")
        last_name = record.get("last_name")
        full_name = None
        if first_name and last_name:
            full_name = f"{first_name} {last_name}"
        elif first_name:
            full_name = first_name
        elif last_name:
            full_name = last_name
        
        return cls(
            id=record["id"],
            email=record["email"],
            full_name=full_name,
            is_admin=record.get("is_admin", False)
        )


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[EmailStr]


class PreferencesUpdate(BaseModel):
    """User preferences/KYC data."""
    genres: Optional[list[str]] = None
    authors: Optional[list[str]] = None
    age: Optional[int] = None
    description: Optional[str] = None
    reading_preferences: Optional[str] = None  # Alias for description for frontend compatibility
    # Add any other preference fields as needed
