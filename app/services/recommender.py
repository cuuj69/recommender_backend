"""Hybrid recommender logic (CBF + CF + GNN)."""
from typing import List, Union
from uuid import UUID

from app.models.book_model import Book
from app.services import cf_service, content_service, gnn_service
from app.utils.vector_ops import normalize


async def recommend_for_user(user_id: Union[str, UUID], limit: int = 10) -> List[Book]:
    """Hybrid recommendation combining content-based, collaborative, and GNN."""
    books = []
    seen_ids = set()
    
    # 1. Content-based filtering (KYC preferences)
    cbf_books = await content_service.get_top_books_by_kyc(user_id, limit)
    for book in cbf_books:
        if book["id"] not in seen_ids:
            books.append(book)
            seen_ids.add(book["id"])
    
    # 2. Collaborative filtering
    cf_vector = await cf_service.get_user_cf_vector(user_id)
    if cf_vector and len(books) < limit:
        cf_books = await cf_service.get_top_books_by_cf(normalize(cf_vector), limit)
        for book in cf_books:
            if book["id"] not in seen_ids:
                books.append(book)
                seen_ids.add(book["id"])
    
    # 3. GNN-based recommendations
    if len(books) < limit:
        gnn_books = await gnn_service.get_top_books_by_gnn(user_id, limit)
        for book in gnn_books:
            if book["id"] not in seen_ids:
                books.append(book)
                seen_ids.add(book["id"])
    
    # 4. Fallback to popular books
    if not books:
        books = await gnn_service.get_popular_books(limit)

    unique_books = []
    seen_ids = set()
    for record in books:
        if record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        unique_books.append(
            Book(
                id=record["id"],
                title=record["title"],
                author=record.get("author"),
                description=record.get("description"),
                genres=record.get("genres"),
                score=record.get("score"),
            )
        )
        if len(unique_books) == limit:
            break

    return unique_books
