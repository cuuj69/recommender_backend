"""Content-based filtering service."""
import heapq
import json
from typing import List, Optional, Union
from uuid import UUID

import numpy as np

from app.db.connection import get_pool
from app.services.embedding_service import encode_kyc_preferences
from app.utils.vector_ops import cosine_similarity


async def get_top_books_by_content(user_embedding: List[float], limit: int = 10, sample_size: int = 10000):
    """Get top books by content similarity (optimized with numpy and smart sampling)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get total count first
        total_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM books WHERE content_embedding IS NOT NULL
            """
        )
        
        if total_count == 0:
            return []
        
        # If we have fewer books than sample size, get all
        if total_count <= sample_size:
            records = await conn.fetch(
                """
                SELECT id, title, author, description, genres, content_embedding
                FROM books
                WHERE content_embedding IS NOT NULL
                """
            )
        else:
            # Smart sampling: mix of popular books and random selection
            # 60% from popular books (by interactions), 40% random
            popular_count = int(sample_size * 0.6)
            random_count = sample_size - popular_count
            
            # Get popular books (by interaction count)
            popular_books = await conn.fetch(
                """
                SELECT b.id, b.title, b.author, b.description, b.genres, b.content_embedding
                FROM books b
                LEFT JOIN interactions i ON i.book_id = b.id
                WHERE b.content_embedding IS NOT NULL
                GROUP BY b.id, b.title, b.author, b.description, b.genres, b.content_embedding
                ORDER BY COUNT(i.id) DESC
                LIMIT $1
                """,
                popular_count
            )
            
            # Get random books (excluding popular ones)
            popular_ids = [book["id"] for book in popular_books]
            
            if popular_ids:
                random_books = await conn.fetch(
                    """
                    SELECT id, title, author, description, genres, content_embedding
                    FROM books
                    WHERE content_embedding IS NOT NULL
                    AND id NOT IN (SELECT unnest($1::int[]))
                    ORDER BY RANDOM()
                    LIMIT $2
                    """,
                    popular_ids,
                    random_count
                )
            else:
                random_books = await conn.fetch(
                    """
                    SELECT id, title, author, description, genres, content_embedding
                    FROM books
                    WHERE content_embedding IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT $1
                    """,
                    random_count
                )
            
            records = list(popular_books) + list(random_books)
    
    if not records:
        return []
    
    # Convert user embedding to numpy array once
    user_vec = np.array(user_embedding, dtype=np.float32)
    
    # Batch process with numpy for better performance
    scored_books = []
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        # Parse all vectors in batch
        book_vectors = []
        book_data = []
        for record in batch:
            if not record["content_embedding"]:
                continue
            book_vector = record["content_embedding"]
            if isinstance(book_vector, str):
                book_vector = json.loads(book_vector)
            
            try:
                book_vec = np.array(book_vector, dtype=np.float32)
                # Calculate cosine similarity efficiently
                score = np.dot(user_vec, book_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(book_vec))
                scored_books.append({
                    "id": record["id"],
                    "title": record["title"],
                    "author": record.get("author"),
                    "description": record.get("description"),
                    "genres": record.get("genres"),
                    "score": float(score)
                })
            except (ValueError, TypeError, np.linalg.LinAlgError):
                continue
    
    # Use heap for top N instead of full sort (faster for large lists)
    if len(scored_books) > limit:
        top_books = heapq.nlargest(limit, scored_books, key=lambda x: x["score"])
        return top_books
    
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

