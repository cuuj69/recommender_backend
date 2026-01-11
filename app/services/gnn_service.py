"""GNN helper functions."""
import heapq
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
        
        # Get total count for smart sampling
        total_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM books WHERE gnn_vector IS NOT NULL
            """
        )
        
        sample_size = 10000  # Increased for better coverage with large datasets
        
        # Smart sampling: mix of popular books and random selection
        if total_count == 0:
            all_books = []
        elif total_count <= sample_size:
            # If we have fewer books than sample size, get all
            all_books = await conn.fetch(
                """
                SELECT id, title, author, description, genres, gnn_vector
                FROM books
                WHERE gnn_vector IS NOT NULL
                """
            )
        else:
            # 60% from popular books (by interactions), 40% random
            popular_count = int(sample_size * 0.6)
            random_count = sample_size - popular_count
            
            # Get popular books (by interaction count)
            popular_books = await conn.fetch(
                """
                SELECT b.id, b.title, b.author, b.description, b.genres, b.gnn_vector
                FROM books b
                LEFT JOIN interactions i ON i.book_id = b.id
                WHERE b.gnn_vector IS NOT NULL
                GROUP BY b.id, b.title, b.author, b.description, b.genres, b.gnn_vector
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
                    SELECT id, title, author, description, genres, gnn_vector
                    FROM books
                    WHERE gnn_vector IS NOT NULL
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
                    SELECT id, title, author, description, genres, gnn_vector
                    FROM books
                    WHERE gnn_vector IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT $1
                    """,
                    random_count
                )
            
            all_books = list(popular_books) + list(random_books)
    
    if not all_books:
        return []
    
    # Convert user GNN vector to numpy array once
    user_vec = np.array(user_gnn, dtype=np.float32)
    
    # Batch process with numpy for better performance
    scored_books = []
    batch_size = 100
    for i in range(0, len(all_books), batch_size):
        batch = all_books[i:i + batch_size]
        
        for record in batch:
            if not record["gnn_vector"]:
                continue
            book_vector = record["gnn_vector"]
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
