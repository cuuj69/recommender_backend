"""Interaction models."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class Interaction(BaseModel):
    id: int
    user_id: str  # UUID as string for API response
    book_id: int
    interaction_type: str
    rating: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}
