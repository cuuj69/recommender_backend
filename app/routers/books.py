"""Book endpoints."""
import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.db.connection import get_pool
from app.models.book_model import Book
from app.services import book_service

router = APIRouter()


def extract_book_data(book_record) -> dict:
    """Extract all book data from database record, including metadata."""
    # Parse metadata if it's a string
    metadata = book_record.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    elif metadata is None:
        metadata = {}
    
    # Extract fields from metadata
    return {
        "id": book_record["id"],
        "title": book_record["title"],
        "author": book_record.get("author"),
        "description": book_record.get("description"),
        "genres": book_record.get("genres"),
        "score": None,  # No score for list/single book endpoints
        "image": metadata.get("image"),
        "preview_link": metadata.get("previewLink"),
        "info_link": metadata.get("infoLink"),
        "publisher": metadata.get("publisher"),
        "published_date": metadata.get("publishedDate"),
        "ratings_count": float(metadata["ratingsCount"]) if metadata.get("ratingsCount") is not None else None,
        "all_authors": metadata.get("allAuthors"),
    }


@router.get("/genres", response_model=List[str])
async def get_genres():
    """Get all unique genres from the database."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get all unique genres from books
        rows = await conn.fetch("""
            SELECT DISTINCT unnest(genres) as genre
            FROM books
            WHERE genres IS NOT NULL AND array_length(genres, 1) > 0
            ORDER BY genre
        """)
        return [row["genre"] for row in rows if row["genre"]]


@router.get("/", response_model=List[Book])
async def list_books(
    limit: int = Query(50, ge=1, le=100, description="Number of books to return"),
    offset: int = Query(0, ge=0, description="Number of books to skip"),
    search: Optional[str] = Query(None, description="Search in title and description"),
    author: Optional[str] = Query(None, description="Filter by author"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
):
    """List books with optional filtering and pagination."""
    books = await book_service.list_books(
        limit=limit,
        offset=offset,
        search=search,
        author=author,
        genre=genre,
    )
    return [Book(**extract_book_data(book)) for book in books]


@router.get("/{book_id}", response_model=Book)
async def get_book(book_id: int):
    """Get book details by ID."""
    book = await book_service.get_book_by_id(book_id)
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID {book_id} not found",
        )
    return Book(**extract_book_data(book))

