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
    """Get personalized recommendations for the authenticated user."""
    user_id = current_user["id"]
    books = await recommender.recommend_for_user(user_id, payload.limit)
    if not books:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendations available. Try adding some book interactions first."
        )
    return RecommendResponse(recommendations=books)
