"""Evaluation service for the recommender (RMSE + Precision@K)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import asyncpg

from app.db.connection import init_db
from app.services import recommender
from app.utils.vector_ops import cosine_similarity


UserId = Union[str, "UUID"]  # type: ignore[name-defined]


@dataclass
class Interaction:
    user_id: UserId
    book_id: int
    rating: Optional[float]
    created_at: datetime


async def _load_interactions(conn: asyncpg.Connection) -> List[Interaction]:
    """Load all interactions from the database."""
    rows = await conn.fetch(
        """
        SELECT user_id, book_id, rating, created_at
        FROM interactions
        ORDER BY created_at
        """
    )
    interactions: List[Interaction] = []
    for row in rows:
        interactions.append(
            Interaction(
                user_id=row["user_id"],
                book_id=row["book_id"],
                rating=float(row["rating"]) if row["rating"] is not None else None,
                created_at=row["created_at"],
            )
        )
    return interactions


def _train_test_split_by_user(
    interactions: List[Interaction],
    min_interactions: int,
    test_ratio: float = 0.2,
) -> Tuple[Dict[UserId, List[Interaction]], Dict[UserId, List[Interaction]]]:
    """Simple temporal train/test split per user."""
    by_user: Dict[UserId, List[Interaction]] = defaultdict(list)
    for inter in interactions:
        by_user[inter.user_id].append(inter)

    train: Dict[UserId, List[Interaction]] = {}
    test: Dict[UserId, List[Interaction]] = {}

    for user_id, user_inters in by_user.items():
        if len(user_inters) < min_interactions:
            continue

        user_inters.sort(key=lambda x: x.created_at)
        n = len(user_inters)
        test_size = max(1, int(n * test_ratio))

        test[user_id] = user_inters[-test_size:]
        train[user_id] = user_inters[:-test_size]

    return train, test


async def _load_cf_vectors(
    conn: asyncpg.Connection,
) -> Tuple[Dict[UserId, List[float]], Dict[int, List[float]]]:
    """Load CF vectors for users and books, if they exist."""
    import json

    user_vectors: Dict[UserId, List[float]] = {}
    book_vectors: Dict[int, List[float]] = {}

    user_rows = await conn.fetch("SELECT id, cf_vector FROM users WHERE cf_vector IS NOT NULL")
    for row in user_rows:
        vec = row["cf_vector"]
        if isinstance(vec, str):
            vec = json.loads(vec)
        user_vectors[row["id"]] = vec

    book_rows = await conn.fetch("SELECT id, cf_embedding FROM books WHERE cf_embedding IS NOT NULL")
    for row in book_rows:
        vec = row["cf_embedding"]
        if isinstance(vec, str):
            vec = json.loads(vec)
        book_vectors[row["id"]] = vec

    return user_vectors, book_vectors


async def evaluate_recommender(
    k: int = 10,
    min_interactions: int = 5,
) -> Dict[str, object]:
    """Run evaluation and return metrics as a dict."""
    pool = await init_db()
    async with pool.acquire() as conn:
        interactions = await _load_interactions(conn)
        if not interactions:
            return {
                "precision_at_k": None,
                "rmse": None,
                "num_eval_users": 0,
                "num_rating_samples": 0,
                "note": "No interactions in database",
            }

        train, test = _train_test_split_by_user(interactions, min_interactions=min_interactions)
        test_users = list(test.keys())
        if not test_users:
            return {
                "precision_at_k": None,
                "rmse": None,
                "num_eval_users": 0,
                "num_rating_samples": 0,
                "note": f"No users with >= {min_interactions} interactions for evaluation",
            }

        # Load CF vectors once for RMSE
        user_cf, book_cf = await _load_cf_vectors(conn)

    precisions: List[float] = []
    sq_errors: List[float] = []
    rating_count = 0

    for user_id in test_users:
        test_interactions = test[user_id]

        # Precision@K using hybrid recommender
        try:
            rec_books = await recommender.recommend_for_user(user_id, limit=k)
        except Exception:  # noqa: BLE001
            continue

        rec_ids = {b.id for b in rec_books}
        relevant_ids = {inter.book_id for inter in test_interactions}
        hits = len(rec_ids & relevant_ids)
        precisions.append(hits / float(k))

        # RMSE using CF vectors and ratings
        for inter in test_interactions:
            if inter.rating is None:
                continue
            if user_id not in user_cf or inter.book_id not in book_cf:
                continue

            u_vec = user_cf[user_id]
            b_vec = book_cf[inter.book_id]
            try:
                sim = cosine_similarity(u_vec, b_vec)
            except Exception:  # noqa: BLE001
                continue

            sim = max(-1.0, min(1.0, sim))
            pred = 2.0 + 3.0 * sim  # map to ~[ -1,1 ] -> [ -1,5 ]
            err = pred - inter.rating
            sq_errors.append(err * err)
            rating_count += 1

    precision_value: Optional[float]
    if precisions:
        precision_value = sum(precisions) / len(precisions)
    else:
        precision_value = None

    rmse_value: Optional[float]
    if rating_count > 0 and sq_errors:
        rmse_value = math.sqrt(sum(sq_errors) / rating_count)
    else:
        rmse_value = None

    return {
        "precision_at_k": precision_value,
        "rmse": rmse_value,
        "num_eval_users": len(precisions),
        "num_rating_samples": rating_count,
    }


