"""Collaborative filtering helper functions."""
import json
from typing import List, Optional, Union
from uuid import UUID

from app.db.connection import get_pool
from app.utils.vector_ops import cosine_similarity


async def get_user_cf_vector(user_id: Union[str, UUID]) -> Optional[List[float]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT cf_vector FROM users WHERE id=$1", user_id)
        if not row or not row["cf_vector"]:
            return None
        # Parse JSONB array to Python list
        if isinstance(row["cf_vector"], str):
            return json.loads(row["cf_vector"])
        return row["cf_vector"]


async def get_top_books_by_cf(user_cf_vector: List[float], limit: int = 10):
    """Get top books by CF similarity (calculated in Python)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Fetch all books with CF embeddings
        records = await conn.fetch(
            """
            SELECT id, title, author, description, genres, cf_embedding
            FROM books
            WHERE cf_embedding IS NOT NULL
            """
        )
    
    # Calculate similarity in Python
    scored_books = []
    for record in records:
        if not record["cf_embedding"]:
            continue
        # Parse JSONB array
        book_vector = record["cf_embedding"]
        if isinstance(book_vector, str):
            book_vector = json.loads(book_vector)
        
        # Calculate cosine similarity
        try:
            score = cosine_similarity(user_cf_vector, book_vector)
            scored_books.append({
                "id": record["id"],
                "title": record["title"],
                "author": record.get("author"),
                "description": record.get("description"),
                "genres": record.get("genres"),
                "score": score
            })
        except (ValueError, TypeError):
            continue
    
    # Sort by score (descending) and return top N
    scored_books.sort(key=lambda x: x["score"], reverse=True)
    return scored_books[:limit]
