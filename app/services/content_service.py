"""Content-based filtering service."""
import json
from typing import List, Optional, Union
from uuid import UUID

from app.db.connection import get_pool
from app.services.embedding_service import encode_kyc_preferences
from app.utils.vector_ops import cosine_similarity


async def get_top_books_by_content(user_embedding: List[float], limit: int = 10):
    """Get top books by content similarity (calculated in Python)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(
            """
            SELECT id, title, author, description, genres, content_embedding
            FROM books
            WHERE content_embedding IS NOT NULL
            """
        )
    
    # Calculate similarity in Python
    scored_books = []
    for record in records:
        if not record["content_embedding"]:
            continue
        book_vector = record["content_embedding"]
        if isinstance(book_vector, str):
            book_vector = json.loads(book_vector)
        
        try:
            score = cosine_similarity(user_embedding, book_vector)
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
    
    scored_books.sort(key=lambda x: x["score"], reverse=True)
    return scored_books[:limit]


async def get_top_books_by_kyc(user_id: Union[str, UUID], limit: int = 10):
    """Get top books based on user's KYC preferences."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        user = await conn.fetchrow("SELECT kyc_preferences, kyc_embedding FROM users WHERE id=$1", user_id)
        if not user:
            return []
        
        # Use existing kyc_embedding if available
        if user["kyc_embedding"]:
            kyc_vector = user["kyc_embedding"]
            if isinstance(kyc_vector, str):
                kyc_vector = json.loads(kyc_vector)
            return await get_top_books_by_content(kyc_vector, limit)
        
        # Otherwise encode preferences on the fly
        if user["kyc_preferences"]:
            # Parse JSON string to dict if needed
            prefs = user["kyc_preferences"]
            if isinstance(prefs, str):
                prefs = json.loads(prefs)
            embedding = encode_kyc_preferences(prefs)
            if embedding:
                return await get_top_books_by_content(embedding, limit)
    
    return []

