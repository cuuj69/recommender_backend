"""Recommendation endpoints."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.models.book_model import Book
from app.routers.users import get_current_user
from app.services import recommender

router = APIRouter()


class RecommendRequest(BaseModel):
    limit: int = 10


class RecommendationMetadata(BaseModel):
    is_personalized: bool
    interaction_count: int
    min_required: int
    needs_more: int


class RecommendResponse(BaseModel):
    recommendations: List[Book]
    metadata: RecommendationMetadata


@router.post("/", response_model=RecommendResponse)
async def get_recommendations(
    payload: RecommendRequest,
    current_user=Depends(get_current_user)
):
    """Get personalized recommendations for the authenticated user (ensures no duplicates).
    
    Returns recommendations along with metadata about personalization status and interaction requirements.
    """
    user_id = current_user["id"]
    books, metadata = await recommender.recommend_for_user(user_id, payload.limit)
    
    if not books:
        # Still return metadata even if no books, so frontend can show helpful message
        return RecommendResponse(
            recommendations=[],
            metadata=RecommendationMetadata(**metadata)
        )
    
    # Final deduplication safety check by book ID
    seen_ids = set()
    unique_books = []
    for book in books:
        book_id = book.id if hasattr(book, 'id') else getattr(book, 'id', None)
        if book_id and book_id not in seen_ids:
            seen_ids.add(book_id)
            unique_books.append(book)
    
    return RecommendResponse(
        recommendations=unique_books[:payload.limit],
        metadata=RecommendationMetadata(**metadata)
    )
