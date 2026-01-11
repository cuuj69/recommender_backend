"""Collaborative filtering helper functions."""
import heapq
import json
from typing import List, Optional, Union
from uuid import UUID

import numpy as np

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


async def get_top_books_by_cf(user_cf_vector: List[float], limit: int = 10, sample_size: int = 2000):
    """Get top books by CF similarity (optimized with numpy and smart sampling)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get total count first
        total_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM books WHERE cf_embedding IS NOT NULL
            """
        )
        
        if total_count == 0:
            return []
        
        # If we have fewer books than sample size, get all
        if total_count <= sample_size:
            records = await conn.fetch(
                """
                SELECT id, title, author, description, genres, cf_embedding
                FROM books
                WHERE cf_embedding IS NOT NULL
                """
            )
        else:
            # Optimized sampling: mostly popular books
            # 80% popular, 20% diverse (using ID pattern instead of RANDOM())
            popular_count = int(sample_size * 0.8)
            random_count = sample_size - popular_count
            
            # Get popular books (optimized query)
            popular_books = await conn.fetch(
                """
                SELECT b.id, b.title, b.author, b.description, b.genres, b.cf_embedding
                FROM books b
                LEFT JOIN interactions i ON i.book_id = b.id
                WHERE b.cf_embedding IS NOT NULL
                GROUP BY b.id, b.title, b.author, b.description, b.genres, b.cf_embedding
                ORDER BY COUNT(i.id) DESC, b.id DESC
                LIMIT $1
                """,
                popular_count
            )
            
            # Fast random sampling using ID pattern (avoid ORDER BY RANDOM())
            if random_count > 0:
                random_books = await conn.fetch(
                    """
                    SELECT id, title, author, description, genres, cf_embedding
                    FROM books
                    WHERE cf_embedding IS NOT NULL
                    AND id % 7 = 0  -- Sample pattern for diversity
                    ORDER BY id DESC
                    LIMIT $1
                    """,
                    random_count
                )
            else:
                random_books = []
            
            records = list(popular_books) + list(random_books)
    
    if not records:
        return []
    
    # Convert user vector to numpy array once
    user_vec = np.array(user_cf_vector, dtype=np.float32)
    
    # Batch process with numpy for better performance
    scored_books = []
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        for record in batch:
            if not record["cf_embedding"]:
                continue
            # Parse JSONB array
            book_vector = record["cf_embedding"]
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
