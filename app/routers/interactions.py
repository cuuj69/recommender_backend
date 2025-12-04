"""Interaction endpoints."""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.models.interaction_model import Interaction
from app.routers.users import get_current_user
from app.services import book_service, interaction_service

router = APIRouter()


class InteractionCreate(BaseModel):
    book_id: int = Field(..., description="ID of the book")
    interaction_type: str = Field(
        ...,
        description="Type of interaction: 'click', 'view', 'like', 'dislike', 'rating'",
    )
    rating: Optional[float] = Field(
        None,
        ge=0.0,
        le=5.0,
        description="Rating (0-5) if interaction_type is 'rating'",
    )


@router.post("/", response_model=Interaction, status_code=status.HTTP_201_CREATED)
async def create_interaction(
    payload: InteractionCreate,
    current_user=Depends(get_current_user),
):
    """Log a user-book interaction (click, view, like, rating, etc.)."""
    # Verify book exists
    book = await book_service.get_book_by_id(payload.book_id)
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID {payload.book_id} not found",
        )

    # Validate interaction type
    valid_types = ["click", "view", "like", "dislike", "rating", "purchase", "share"]
    if payload.interaction_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid interaction_type. Must be one of: {', '.join(valid_types)}",
        )

    # Validate rating is provided for rating type
    if payload.interaction_type == "rating" and payload.rating is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="rating is required when interaction_type is 'rating'",
        )

    interaction = await interaction_service.create_interaction(
        user_id=current_user["id"],
        book_id=payload.book_id,
        interaction_type=payload.interaction_type,
        rating=payload.rating,
    )

    return Interaction(
        id=interaction["id"],
        user_id=str(interaction["user_id"]),  # Convert UUID to string for response
        book_id=interaction["book_id"],
        interaction_type=interaction["interaction_type"],
        rating=float(interaction["rating"]) if interaction["rating"] else None,
        created_at=interaction["created_at"],
    )


@router.get("/me", response_model=List[Interaction])
async def get_my_interactions(
    current_user=Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Get all interactions for the current authenticated user."""
    interactions = await interaction_service.get_user_interactions(
        user_id=current_user["id"],
        limit=limit,
        offset=offset,
    )
    return [
        Interaction(
            id=interaction["id"],
            user_id=str(interaction["user_id"]),
            book_id=interaction["book_id"],
            interaction_type=interaction["interaction_type"],
            rating=float(interaction["rating"]) if interaction["rating"] else None,
            created_at=interaction["created_at"],
        )
        for interaction in interactions
    ]

