"""Hybrid recommender logic (CBF + CF + GNN)."""
from typing import List, Union
from uuid import UUID

from app.models.book_model import Book
from app.services import cf_service, content_service, gnn_service
from app.utils.vector_ops import normalize


async def recommend_for_user(user_id: Union[str, UUID], limit: int = 10) -> List[Book]:
    """Hybrid recommendation combining content-based, collaborative, and GNN (optimized)."""
    import asyncio
    
    # Use dict to store unique books by ID (prevents duplicates)
    unique_by_id = {}
    
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
    
    # 4. GNN-based recommendations
    tasks.append(gnn_service.get_top_books_by_gnn(user_id, limit * 2))
    
    # Run CF and GNN in parallel
    if tasks:
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    continue
                for book in result:
                    if isinstance(book, dict):
                        book_id = book.get("id")
                        if book_id and book_id not in unique_by_id:
                            unique_by_id[book_id] = book
        except Exception:
            pass
    
    # 5. Fallback to popular books if no candidates
    if not unique_by_id:
        try:
            popular_books = await gnn_service.get_popular_books(limit * 2)
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
