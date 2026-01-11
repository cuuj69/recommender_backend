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
    
    # 5. Fallback: If no personalized recommendations found, use interaction-based recommendations
    # This works even when books don't have embeddings - uses genres/authors from user's interactions
    if not unique_by_id:
        pool = await get_pool()
        async with pool.acquire() as conn:
            try:
                # Get books user has already interacted with (to exclude)
                interacted_books = await conn.fetch(
                    "SELECT DISTINCT book_id FROM interactions WHERE user_id = $1",
                    user_id
                )
                interacted_ids = {row["book_id"] for row in interacted_books}
                
                # Get user's interaction patterns (genres and authors they've interacted with)
                user_patterns = await conn.fetch(
                    """
                    SELECT DISTINCT 
                        unnest(b.genres) as genre,
                        b.author
                    FROM interactions i
                    JOIN books b ON b.id = i.book_id
                    WHERE i.user_id = $1
                        AND (b.genres IS NOT NULL OR b.author IS NOT NULL)
                    """,
                    user_id
                )
                
                # Extract unique genres and authors
                user_genres = set()
                user_authors = set()
                for row in user_patterns:
                    if row["genre"]:
                        user_genres.add(row["genre"])
                    if row["author"]:
                        user_authors.add(row["author"])
                
                # Debug: log what we found
                import logging
                logging.info(f"User {user_id} - Genres: {user_genres}, Authors: {user_authors}, Interacted IDs: {len(interacted_ids)}")
                
                # Get books that match user's preferences but they haven't interacted with
                if user_genres or user_authors:
                    # Build query to find similar books
                    query_parts = []
                    params = []
                    param_count = 0
                    
                    if user_genres:
                        param_count += 1
                        query_parts.append(f"${param_count}::text[] && b.genres")
                        params.append(list(user_genres))
                    
                    if user_authors:
                        param_count += 1
                        query_parts.append(f"b.author = ANY(${param_count}::text[])")
                        params.append(list(user_authors))
                    
                    if query_parts:
                        # Build the query with proper parameter numbering
                        param_count += 1
                        excluded_ids_param = param_count
                        params.append(list(interacted_ids))
                        
                        # Add author array for ordering if we have authors
                        if user_authors:
                            param_count += 1
                            author_array_param = param_count
                            params.append(list(user_authors))
                            order_clause = f"CASE WHEN b.author = ANY(${author_array_param}::text[]) THEN 1 ELSE 2 END,"
                        else:
                            order_clause = ""
                        
                        param_count += 1
                        limit_param = param_count
                        params.append(limit * 2)
                        
                        query = f"""
                            SELECT DISTINCT b.id, b.title, b.author, b.description, b.genres
                            FROM books b
                            WHERE ({' OR '.join(query_parts)})
                                AND b.id != ALL(${excluded_ids_param}::int[])
                            ORDER BY {order_clause} b.id DESC
                            LIMIT ${limit_param}
                        """
                        
                        fallback_books = await conn.fetch(query, *params)
                        logging.info(f"Fallback query returned {len(fallback_books)} books")
                        
                        for book in fallback_books:
                            book_id = book.get("id")
                            if book_id and book_id not in unique_by_id:
                                # Calculate a simple score based on matching genres/authors
                                score = 0.0
                                if book.get("genres"):
                                    matching_genres = len(set(book["genres"]) & user_genres)
                                    score += matching_genres * 0.5
                                if book.get("author") in user_authors:
                                    score += 1.0
                                
                                unique_by_id[book_id] = {
                                    "id": book_id,
                                    "title": book.get("title", ""),
                                    "author": book.get("author"),
                                    "description": book.get("description"),
                                    "genres": book.get("genres"),
                                    "score": score
                                }
                                is_personalized = True  # This is still personalized based on interactions
                
                # If still no recommendations, use a more basic fallback: just recommend any books they haven't seen
                # This will work even if books have no genres/authors
                if not unique_by_id:
                    try:
                        # Get any books the user hasn't interacted with
                        # Use empty array if no interactions (shouldn't happen, but safe)
                        excluded_ids_list = list(interacted_ids) if interacted_ids else []
                        
                        # If we have interacted books, exclude them. Otherwise, just get any books.
                        if excluded_ids_list:
                            any_books = await conn.fetch(
                                """
                                SELECT id, title, author, description, genres
                                FROM books
                                WHERE id != ALL($1::int[])
                                ORDER BY id DESC
                                LIMIT $2
                                """,
                                excluded_ids_list,
                                limit * 2
                            )
                        else:
                            # Fallback if somehow no interacted_ids (shouldn't happen but be safe)
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
                                    "score": 0.5  # Basic score for fallback
                                }
                                is_personalized = True  # Still based on their interactions (excluding what they've seen)
                    except Exception as e2:
                        import logging
                        logging.error(f"Error in basic fallback recommendation: {e2}")
                        import traceback
                        logging.error(traceback.format_exc())
                        # Don't pass - we want to see what's wrong
                        pass
            except Exception as e:
                # Log the error but don't fail completely - try final fallback
                import logging
                logging.error(f"Error in fallback recommendation: {e}")
                import traceback
                logging.error(traceback.format_exc())
                
                # Last resort: try to get ANY books at all with a fresh connection
                try:
                    async with pool.acquire() as conn2:
                        interacted_books = await conn2.fetch(
                            "SELECT DISTINCT book_id FROM interactions WHERE user_id = $1",
                            user_id
                        )
                        interacted_ids = {row["book_id"] for row in interacted_books}
                        
                        any_books = await conn2.fetch(
                            """
                            SELECT id, title, author, description, genres
                            FROM books
                            WHERE id != ALL($1::int[])
                            ORDER BY id DESC
                            LIMIT $2
                            """,
                            list(interacted_ids) if interacted_ids else [],
                            limit * 2
                        )
                        
                        for book in any_books:
                            book_id = book.get("id")
                            if book_id:
                                unique_by_id[book_id] = {
                                    "id": book_id,
                                    "title": book.get("title", ""),
                                    "author": book.get("author"),
                                    "description": book.get("description"),
                                    "genres": book.get("genres"),
                                    "score": 0.3
                                }
                                is_personalized = True
                except Exception as final_error:
                    import logging
                    logging.error(f"Even final fallback failed: {final_error}")
                    import traceback
                    logging.error(traceback.format_exc())
    
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
