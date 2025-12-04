"""Graph service for building recommendation graphs."""
import json
from typing import Dict, List, Optional, Union
from uuid import UUID

from app.db.connection import get_pool
from app.utils.vector_ops import cosine_similarity


async def build_user_graph(user_id: Union[str, UUID], max_books: int = 20) -> Dict:
    """Build ego-graph for a specific user."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get user info
        user = await conn.fetchrow("SELECT id, email, first_name, last_name FROM users WHERE id=$1", user_id)
        if not user:
            return {"nodes": [], "edges": []}
        
        # Get user's interactions
        interactions = await conn.fetch(
            """
            SELECT i.book_id, i.interaction_type, b.title, b.author, b.genres
            FROM interactions i
            JOIN books b ON b.id = i.book_id
            WHERE i.user_id = $1
            ORDER BY i.created_at DESC
            LIMIT $2
            """,
            user_id,
            max_books
        )
        
        # Get recommendations for this user
        rec_books = []
        rec_ids = set()
        try:
            from app.services import recommender
            rec_books = await recommender.recommend_for_user(user_id, limit=10)
            rec_ids = {b.id for b in rec_books}
        except Exception:
            pass
        
        # Get book embeddings for similarity calculation
        book_embeddings = {}
        book_ids = [row["book_id"] for row in interactions] + list(rec_ids)
        if book_ids:
            books_data = await conn.fetch(
                """
                SELECT id, title, author, genres, content_embedding, cf_embedding
                FROM books
                WHERE id = ANY($1::int[])
                """,
                book_ids
            )
            for book in books_data:
                content_emb = book.get("content_embedding")
                cf_emb = book.get("cf_embedding")
                book_embeddings[book["id"]] = {
                    "content": json.loads(content_emb) if content_emb and isinstance(content_emb, str) else (content_emb if content_emb else None),
                    "cf": json.loads(cf_emb) if cf_emb and isinstance(cf_emb, str) else (cf_emb if cf_emb else None),
                }
        
        # Build nodes
        nodes = [
            {
                "id": f"user_{user['id']}",
                "type": "user",
                "label": user.get("email", "User"),
                "name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or user.get("email", "User")
            }
        ]
        
        seen_books = set()
        for row in interactions:
            book_id = row["book_id"]
            if book_id not in seen_books:
                nodes.append({
                    "id": f"book_{book_id}",
                    "type": "book",
                    "label": row["title"],
                    "title": row["title"],
                    "author": row.get("author"),
                    "genres": row.get("genres", [])
                })
                seen_books.add(book_id)
        
        for rec_book in rec_books:
            if rec_book.id not in seen_books:
                nodes.append({
                    "id": f"book_{rec_book.id}",
                    "type": "book",
                    "label": rec_book.title,
                    "title": rec_book.title,
                    "author": rec_book.author,
                    "genres": rec_book.genres or [],
                    "recommended": True
                })
                seen_books.add(rec_book.id)
        
        # Build edges
        edges = []
        for row in interactions:
            edges.append({
                "source": f"user_{user['id']}",
                "target": f"book_{row['book_id']}",
                "type": row["interaction_type"],
                "weight": 1.0
            })
        
        # Add similarity edges between books (if embeddings exist)
        book_list = list(seen_books)
        for i, book_id1 in enumerate(book_list):
            for book_id2 in book_list[i+1:]:
                if book_id1 in book_embeddings and book_id2 in book_embeddings:
                    # Try content embedding first, then CF
                    vec1 = book_embeddings[book_id1].get("content") or book_embeddings[book_id1].get("cf")
                    vec2 = book_embeddings[book_id2].get("content") or book_embeddings[book_id2].get("cf")
                    
                    if vec1 and vec2:
                        try:
                            sim = cosine_similarity(vec1, vec2)
                            if sim > 0.5:  # Only show strong similarities
                                edges.append({
                                    "source": f"book_{book_id1}",
                                    "target": f"book_{book_id2}",
                                    "type": "similar",
                                    "weight": sim
                                })
                        except Exception:
                            pass
        
        return {
            "nodes": nodes,
            "edges": edges
        }


async def build_overview_graph(max_users: int = 50, max_books: int = 100) -> Dict:
    """Build overview graph with top users and books."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get top users by interaction count
        top_users = await conn.fetch(
            """
            SELECT u.id, u.email, u.first_name, u.last_name, COUNT(i.id) as interaction_count
            FROM users u
            LEFT JOIN interactions i ON i.user_id = u.id
            GROUP BY u.id, u.email, u.first_name, u.last_name
            ORDER BY interaction_count DESC
            LIMIT $1
            """,
            max_users
        )
        
        # Get top books by interaction count
        top_books = await conn.fetch(
            """
            SELECT b.id, b.title, b.author, b.genres, COUNT(i.id) as interaction_count
            FROM books b
            LEFT JOIN interactions i ON i.book_id = b.id
            GROUP BY b.id, b.title, b.author, b.genres
            ORDER BY interaction_count DESC
            LIMIT $1
            """,
            max_books
        )
        
        # Get interactions between top users and top books
        user_ids = [row["id"] for row in top_users]
        book_ids = [row["id"] for row in top_books]
        
        interactions = []
        if user_ids and book_ids:
            interactions = await conn.fetch(
                """
                SELECT i.user_id, i.book_id, i.interaction_type
                FROM interactions i
                WHERE i.user_id = ANY($1::uuid[]) AND i.book_id = ANY($2::int[])
                """,
                user_ids,
                book_ids
            )
        
        # Build nodes
        nodes = []
        for user in top_users:
            nodes.append({
                "id": f"user_{user['id']}",
                "type": "user",
                "label": user.get("email", "User"),
                "name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or user.get("email", "User"),
                "interaction_count": user["interaction_count"]
            })
        
        for book in top_books:
            nodes.append({
                "id": f"book_{book['id']}",
                "type": "book",
                "label": book["title"],
                "title": book["title"],
                "author": book.get("author"),
                "genres": book.get("genres", []),
                "interaction_count": book["interaction_count"]
            })
        
        # Build edges
        edges = []
        for inter in interactions:
            edges.append({
                "source": f"user_{inter['user_id']}",
                "target": f"book_{inter['book_id']}",
                "type": inter["interaction_type"],
                "weight": 1.0
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_users": len(top_users),
                "total_books": len(top_books),
                "total_interactions": len(edges)
            }
        }

