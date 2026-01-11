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


class RecommendResponse(BaseModel):
    recommendations: List[Book]


@router.post("/", response_model=RecommendResponse)
async def get_recommendations(
    payload: RecommendRequest,
    current_user=Depends(get_current_user)
):
    """Get personalized recommendations for the authenticated user (ensures no duplicates)."""
    user_id = current_user["id"]
    books = await recommender.recommend_for_user(user_id, payload.limit)
    if not books:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendations available. Try adding some book interactions first."
        )
    
    # Final deduplication safety check by book ID
    seen_ids = set()
    unique_books = []
    for book in books:
        book_id = book.id if hasattr(book, 'id') else getattr(book, 'id', None)
        if book_id and book_id not in seen_ids:
            seen_ids.add(book_id)
            unique_books.append(book)
    
    # If we have no recommendations, check if user has interactions to provide better error message
    if not unique_books:
        from app.services import interaction_service
        from app.db.connection import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            interaction_count = await conn.fetchval(
                "SELECT COUNT(*) FROM interactions WHERE user_id = $1",
                user_id
            )
        
        if interaction_count and interaction_count > 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No recommendations available. The system is still processing your interactions. Please try again in a moment or interact with more books."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No recommendations available. Try interacting with some books (like, view, rate) to get personalized recommendations."
            )
    
    return RecommendResponse(recommendations=unique_books[:payload.limit])
