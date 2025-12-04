"""Interaction service helpers."""
from typing import List, Optional
from uuid import UUID

import asyncpg

from app.db.connection import get_pool


async def create_interaction(
    user_id: UUID,
    book_id: int,
    interaction_type: str,
    rating: Optional[float] = None,
) -> asyncpg.Record:
    """Create a new user-book interaction."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        interaction = await conn.fetchrow(
            """
            INSERT INTO interactions (user_id, book_id, interaction_type, rating)
            VALUES ($1, $2, $3, $4)
            RETURNING *
            """,
            user_id,
            book_id,
            interaction_type,
            rating,
        )
        return interaction


async def get_user_interactions(
    user_id: UUID,
    limit: int = 50,
    offset: int = 0,
) -> List[asyncpg.Record]:
    """Get all interactions for a user."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT i.*, b.title, b.author
            FROM interactions i
            JOIN books b ON b.id = i.book_id
            WHERE i.user_id = $1
            ORDER BY i.created_at DESC
            LIMIT $2 OFFSET $3
            """,
            user_id,
            limit,
            offset,
        )


async def get_book_interactions(book_id: int) -> List[asyncpg.Record]:
    """Get all interactions for a book."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT * FROM interactions
            WHERE book_id = $1
            ORDER BY created_at DESC
            """,
            book_id,
        )

