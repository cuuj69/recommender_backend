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
        
        sample_size = 2000  # Optimized for performance while maintaining quality
        
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
                    AND title IS NOT NULL 
                    AND title != ''
                    AND author IS NOT NULL 
                    AND author != ''
                    AND description IS NOT NULL 
                    AND description != ''
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
                SELECT b.id, b.title, b.author, b.description, b.genres, b.gnn_vector
                FROM books b
                LEFT JOIN interactions i ON i.book_id = b.id
                WHERE b.gnn_vector IS NOT NULL
                    AND b.title IS NOT NULL 
                    AND b.title != ''
                    AND b.author IS NOT NULL 
                    AND b.author != ''
                    AND b.description IS NOT NULL 
                    AND b.description != ''
                GROUP BY b.id, b.title, b.author, b.description, b.genres, b.gnn_vector
                ORDER BY COUNT(i.id) DESC, b.id DESC
                LIMIT $1
                """,
                popular_count
            )
            
            # Fast random sampling using ID pattern (avoid ORDER BY RANDOM())
            if random_count > 0:
                random_books = await conn.fetch(
                    """
                    SELECT id, title, author, description, genres, gnn_vector
                    FROM books
                    WHERE gnn_vector IS NOT NULL
                        AND id % 7 = 0  -- Sample pattern for diversity
                        AND title IS NOT NULL 
                        AND title != ''
                        AND author IS NOT NULL 
                        AND author != ''
                        AND description IS NOT NULL 
                        AND description != ''
                    ORDER BY id DESC
                    LIMIT $1
                    """,
                    random_count
                )
            else:
                random_books = []
            
            # Deduplicate by ID when combining popular and random books
            records_dict = {}
            for record in popular_books:
                records_dict[record["id"]] = record
            for record in random_books:
                if record["id"] not in records_dict:
                    records_dict[record["id"]] = record
            all_books = list(records_dict.values())
    
    if not all_books:
        return []
    
    # Convert user GNN vector to numpy array once
    user_vec = np.array(user_gnn, dtype=np.float32)
    
    # Batch process with numpy for better performance
    scored_books = []
    seen_ids = set()  # Track IDs to prevent duplicates
    batch_size = 100
    for i in range(0, len(all_books), batch_size):
        batch = all_books[i:i + batch_size]
        
        for record in batch:
            book_id = record["id"]
            # Skip if we've already processed this book
            if book_id in seen_ids:
                continue
            seen_ids.add(book_id)
            
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
                    "id": book_id,
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
            WHERE b.title IS NOT NULL 
                AND b.title != ''
                AND b.author IS NOT NULL 
                AND b.author != ''
                AND b.description IS NOT NULL 
                AND b.description != ''
            GROUP BY b.id
            ORDER BY score DESC NULLS LAST
            LIMIT $1
            """,
            limit,
        )
    return records
