"""Book service helpers."""
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
        query = "SELECT DISTINCT ON (id) * FROM books WHERE 1=1"
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

        query += " ORDER BY id, created_at DESC"
        param_count += 1
        query += f" LIMIT ${param_count}"
        params.append(limit)
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
        
        return unique_books


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

