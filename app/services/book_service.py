"""Book service helpers."""
import random
from typing import List, Optional

import asyncpg

from app.db.connection import get_pool
from app.models.book_model import Book


async def get_book_by_id(book_id: int) -> Optional[asyncpg.Record]:
    """Get a book by its ID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM books WHERE id=$1", book_id)


async def list_books(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    author: Optional[str] = None,
    genre: Optional[str] = None,
) -> List[asyncpg.Record]:
    """List books with optional filtering (ensures no duplicates)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        query = """SELECT DISTINCT ON (id) * FROM books 
                   WHERE title IS NOT NULL 
                   AND title != '' 
                   AND author IS NOT NULL 
                   AND author != '' 
                   AND description IS NOT NULL 
                   AND description != ''"""
        params = []
        param_count = 0

        if search:
            param_count += 1
            query += f" AND (title ILIKE ${param_count} OR description ILIKE ${param_count})"
            params.append(f"%{search}%")

        if author:
            param_count += 1
            query += f" AND author ILIKE ${param_count}"
            params.append(f"%{author}%")

        if genre:
            param_count += 1
            query += f" AND ${param_count} = ANY(genres)"
            params.append(genre)

        # Check if we have any filters - if not, we'll randomize for variety
        has_filters = bool(search or author or genre)
        
        # DISTINCT ON (id) requires ORDER BY to start with id
        # We'll order by id first (required), then randomize in Python
        if has_filters:
            # When filtering, order by relevance
            query += " ORDER BY id, created_at DESC"
        else:
            # No filters - order by id (required for DISTINCT ON) but we'll randomize in Python
            query += " ORDER BY id DESC"
        
        param_count += 1
        query += f" LIMIT ${param_count}"
        # Fetch more than needed if no filters (for better randomization)
        fetch_limit = int(limit * 2) if not has_filters else limit
        params.append(fetch_limit)
        param_count += 1
        query += f" OFFSET ${param_count}"
        params.append(offset)

        books = await conn.fetch(query, *params)
        
        # Additional deduplication by ID (safety check)
        seen_ids = set()
        unique_books = []
        for book in books:
            book_id = book.get("id")
            if book_id and book_id not in seen_ids:
                seen_ids.add(book_id)
                unique_books.append(book)
        
        # Randomize the order for variety (especially when no filters)
        random.shuffle(unique_books)
        
        # Return only the requested limit
        return unique_books[:limit]


async def create_book(
    title: str,
    author: Optional[str] = None,
    description: Optional[str] = None,
    genres: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> asyncpg.Record:
    """Create a new book."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        book = await conn.fetchrow(
            """
            INSERT INTO books (title, author, description, genres, metadata)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            title,
            author,
            description,
            genres,
            metadata,
        )
        return book

