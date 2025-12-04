"""Book models."""
from typing import List, Optional

from pydantic import BaseModel


class Book(BaseModel):
    id: int
    title: str
    author: Optional[str] = None
    description: Optional[str] = None
    genres: Optional[List[str]] = None
    score: Optional[float] = None
    
    # Additional fields from CSV
    image: Optional[str] = None  # Book cover image URL
    preview_link: Optional[str] = None  # Link to preview the book
    info_link: Optional[str] = None  # Link to book information
    publisher: Optional[str] = None
    published_date: Optional[str] = None
    ratings_count: Optional[float] = None  # Can be float from CSV
    all_authors: Optional[List[str]] = None  # All authors (not just first)

    model_config = {"from_attributes": True}
