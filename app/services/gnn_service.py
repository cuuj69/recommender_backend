"""GNN helper functions."""
import json
from typing import List, Union
from uuid import UUID

import numpy as np

from app.db.connection import get_pool
from app.utils.vector_ops import cosine_similarity


async def get_top_books_by_gnn(user_id: Union[str, UUID], limit: int = 10):
    """Get top books by GNN similarity (calculated in Python)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get user's interacted books with GNN vectors
        user_books = await conn.fetch(
            """
            SELECT b.gnn_vector
            FROM interactions i
            JOIN books b ON b.id = i.book_id
            WHERE i.user_id = $1 AND b.gnn_vector IS NOT NULL
            """,
            user_id,
        )
        
        if not user_books:
            return []
        
        # Calculate average GNN vector for user
        vectors = []
        for record in user_books:
            vec = record["gnn_vector"]
            if isinstance(vec, str):
                vec = json.loads(vec)
            vectors.append(vec)
        
        user_gnn = np.mean(vectors, axis=0).tolist()
        
        # Get all books with GNN vectors
        all_books = await conn.fetch(
            """
            SELECT id, title, author, description, genres, gnn_vector
            FROM books
            WHERE gnn_vector IS NOT NULL
            """
        )
    
    # Calculate similarity in Python
    scored_books = []
    for record in all_books:
        if not record["gnn_vector"]:
            continue
        book_vector = record["gnn_vector"]
        if isinstance(book_vector, str):
            book_vector = json.loads(book_vector)
        
        try:
            score = cosine_similarity(user_gnn, book_vector)
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


async def get_popular_books(limit: int = 10):
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(
            """
            SELECT b.id, b.title, b.author, b.description, b.genres,
                   COUNT(i.id) AS score
            FROM books b
            LEFT JOIN interactions i ON i.book_id = b.id
            GROUP BY b.id
            ORDER BY score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
        )
    return records
