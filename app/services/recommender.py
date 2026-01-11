"""Hybrid recommender logic (CBF + CF + GNN)."""
from typing import List, Tuple, Union
from uuid import UUID

from app.db.connection import get_pool
from app.models.book_model import Book
from app.services import cf_service, content_service, gnn_service
from app.utils.vector_ops import normalize

# Minimum interactions required for personalized recommendations
MIN_INTERACTIONS_FOR_PERSONALIZATION = 3


async def _get_user_interaction_count(user_id: Union[str, UUID]) -> int:
    """Get the count of interactions for a user."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM interactions WHERE user_id = $1",
            user_id
        )
        return count if count else 0


async def _check_user_has_interactions(user_id: Union[str, UUID]) -> bool:
    """Check if user has any interactions in the database."""
    count = await _get_user_interaction_count(user_id)
    return count > 0


async def recommend_for_user(user_id: Union[str, UUID], limit: int = 10) -> Tuple[List[Book], dict]:
    """Hybrid recommendation combining content-based, collaborative, and GNN (optimized).
    
    For new users or users with insufficient data, falls back to popular books.
    
    Returns:
        Tuple of (books_list, metadata_dict) where metadata contains:
        - is_personalized: bool - whether recommendations are personalized
        - interaction_count: int - current number of user interactions
        - min_required: int - minimum interactions needed for personalization
        - needs_more: int - how many more interactions are needed (0 if enough)
    """
    import asyncio
    
    # Get interaction count
    interaction_count = await _get_user_interaction_count(user_id)
    has_sufficient_interactions = interaction_count >= MIN_INTERACTIONS_FOR_PERSONALIZATION
    
    # If user doesn't have enough interactions, return empty list with metadata
    if not has_sufficient_interactions:
        needs_more = MIN_INTERACTIONS_FOR_PERSONALIZATION - interaction_count
        metadata = {
            "is_personalized": False,
            "interaction_count": interaction_count,
            "min_required": MIN_INTERACTIONS_FOR_PERSONALIZATION,
            "needs_more": needs_more
        }
        return [], metadata
    
    # Use dict to store unique books by ID (prevents duplicates)
    unique_by_id = {}
    is_personalized = False
    
    # User has sufficient interactions - get personalized recommendations
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
                is_personalized = True  # Mark as personalized if we got CBF results
    
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
                                is_personalized = True  # Mark as personalized if we got CF/GNN results
        except Exception:
            pass
    
    # 5. If no personalized recommendations found (even with sufficient interactions), return empty
    # We don't fall back to popular books - only return personalized recommendations
    
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

    # Calculate how many more interactions are needed
    needs_more = max(0, MIN_INTERACTIONS_FOR_PERSONALIZATION - interaction_count)
    
    # Prepare metadata
    metadata = {
        "is_personalized": is_personalized,
        "interaction_count": interaction_count,
        "min_required": MIN_INTERACTIONS_FOR_PERSONALIZATION,
        "needs_more": needs_more
    }

    return unique_books, metadata
