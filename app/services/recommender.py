"""Hybrid recommender logic (CBF + CF + GNN)."""
from typing import List, Union
from uuid import UUID

from app.db.connection import get_pool
from app.models.book_model import Book
from app.services import cf_service, content_service, gnn_service
from app.utils.vector_ops import normalize


async def _check_user_has_interactions(user_id: Union[str, UUID]) -> bool:
    """Check if user has any interactions in the database."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM interactions WHERE user_id = $1",
            user_id
        )
        return count > 0 if count else False


async def recommend_for_user(user_id: Union[str, UUID], limit: int = 10) -> List[Book]:
    """Hybrid recommendation combining content-based, collaborative, and GNN (optimized).
    
    For new users or users with insufficient data, falls back to popular books.
    """
    import asyncio
    
    # Check if user has interactions
    has_interactions = await _check_user_has_interactions(user_id)
    
    # Use dict to store unique books by ID (prevents duplicates)
    unique_by_id = {}
    
    # Only try personalized methods if user has interactions
    if has_interactions:
        # Run methods in parallel for better performance
        # 1. Content-based filtering (KYC preferences)
        cbf_task = content_service.get_top_books_by_kyc(user_id, limit * 2)
        
        # 2. Get CF vector first (needed for CF recommendations)
        cf_vector_task = cf_service.get_user_cf_vector(user_id)
        
        # Run CBF and CF vector fetch in parallel
        try:
            cbf_books, cf_vector = await asyncio.gather(cbf_task, cf_vector_task)
        except Exception:
            cbf_books = []
            cf_vector = None
        
        # Add CBF books (ensure they're dicts)
        for book in cbf_books:
            if isinstance(book, dict):
                book_id = book.get("id")
                if book_id and book_id not in unique_by_id:
                    unique_by_id[book_id] = book
        
        # 3. Collaborative filtering (if vector exists) - run in parallel with GNN
        tasks = []
        if cf_vector:
            tasks.append(cf_service.get_top_books_by_cf(normalize(cf_vector), limit * 2))
        
        # 4. GNN-based recommendations (only if user has interactions)
        tasks.append(gnn_service.get_top_books_by_gnn(user_id, limit * 2))
        
        # Run CF and GNN in parallel
        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    if result:  # Check if result is not empty
                        for book in result:
                            if isinstance(book, dict):
                                book_id = book.get("id")
                                if book_id and book_id not in unique_by_id:
                                    unique_by_id[book_id] = book
            except Exception:
                pass
    
    # 5. Fallback to popular books if no personalized recommendations or if user has no interactions
    # Always try to get popular books to ensure we return something
    if not unique_by_id:
        try:
            popular_books = await gnn_service.get_popular_books(limit * 2)
            if popular_books:
                for book in popular_books:
                    book_id = book.get("id") if isinstance(book, dict) else getattr(book, "id", None)
                    if book_id and book_id not in unique_by_id:
                        unique_by_id[book_id] = {
                            "id": book_id,
                            "title": book.get("title") if isinstance(book, dict) else getattr(book, "title", ""),
                            "author": book.get("author") if isinstance(book, dict) else getattr(book, "author", None),
                            "description": book.get("description") if isinstance(book, dict) else getattr(book, "description", None),
                            "genres": book.get("genres") if isinstance(book, dict) else getattr(book, "genres", None),
                            "score": book.get("score", 0.0) if isinstance(book, dict) else getattr(book, "score", 0.0)
                        }
        except Exception:
            pass
        
        # Final fallback: get any books from database if popular books didn't work
        if not unique_by_id:
            try:
                pool = await get_pool()
                async with pool.acquire() as conn:
                    any_books = await conn.fetch(
                        """
                        SELECT id, title, author, description, genres
                        FROM books
                        ORDER BY id DESC
                        LIMIT $1
                        """,
                        limit * 2
                    )
                    for book in any_books:
                        book_id = book.get("id")
                        if book_id and book_id not in unique_by_id:
                            unique_by_id[book_id] = {
                                "id": book_id,
                                "title": book.get("title", ""),
                                "author": book.get("author"),
                                "description": book.get("description"),
                                "genres": book.get("genres"),
                                "score": 0.0  # Default score for fallback books
                            }
            except Exception:
                pass
    
    # Sort by score and take top N
    sorted_candidates = sorted(unique_by_id.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    
    # Return top N unique books
    unique_books = []
    for record in sorted_candidates[:limit]:
        unique_books.append(
            Book(
                id=record["id"],
                title=record.get("title", ""),
                author=record.get("author"),
                description=record.get("description"),
                genres=record.get("genres"),
                score=record.get("score"),
            )
        )

    return unique_books
