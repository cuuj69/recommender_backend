"""Evaluation service for the recommender (RMSE, MAE, Precision@K, Recall@K, nDCG@K)."""

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


def _calculate_recall_at_k(recommended_ids: set, relevant_ids: set, k: int) -> float:
    """Calculate Recall@K: proportion of relevant items found in top K recommendations."""
    if not relevant_ids:
        return 0.0
    hits = len(recommended_ids & relevant_ids)
    return hits / float(len(relevant_ids))


def _calculate_ndcg_at_k(recommended_ids: List[int], relevant_ids: set, k: int) -> float:
    """Calculate nDCG@K: normalized discounted cumulative gain at K."""
    if not relevant_ids:
        return 0.0
    
    # Calculate DCG@K
    dcg = 0.0
    for i, book_id in enumerate(recommended_ids[:k], start=1):
        if book_id in relevant_ids:
            # rel = 1 if relevant, 0 otherwise
            # DCG = sum(rel_i / log2(i+1))
            dcg += 1.0 / math.log2(i + 1)
    
    # Calculate IDCG@K (ideal DCG - all relevant items at the top)
    idcg = 0.0
    num_relevant = min(len(relevant_ids), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    # nDCG = DCG / IDCG
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


async def evaluate_recommender(
    k: int = 10,
    min_interactions: int = 5,
    k_values: Optional[List[int]] = None,
) -> Dict[str, object]:
    """Run evaluation and return metrics as a dict.
    
    Args:
        k: Default K value (for backward compatibility)
        min_interactions: Minimum interactions per user for evaluation
        k_values: List of K values to evaluate (default: [5, 10, 20, 50])
    
    Returns:
        Dictionary with all evaluation metrics including:
        - precision_at_k: Dict mapping k values to precision scores
        - recall_at_k: Dict mapping k values to recall scores
        - ndcg_at_k: Dict mapping k values to nDCG scores
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - rmse_table: List of actual vs predicted pairs
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]
    
    pool = await init_db()
    async with pool.acquire() as conn:
        interactions = await _load_interactions(conn)
        if not interactions:
            return {
                "precision_at_k": {},
                "recall_at_k": {},
                "ndcg_at_k": {},
                "rmse": None,
                "mae": None,
                "rmse_table": [],
                "num_eval_users": 0,
                "num_rating_samples": 0,
                "note": "No interactions in database",
            }

        train, test = _train_test_split_by_user(interactions, min_interactions=min_interactions)
        test_users = list(test.keys())
        if not test_users:
            return {
                "precision_at_k": {},
                "recall_at_k": {},
                "ndcg_at_k": {},
                "rmse": None,
                "mae": None,
                "rmse_table": [],
                "num_eval_users": 0,
                "num_rating_samples": 0,
                "note": f"No users with >= {min_interactions} interactions for evaluation",
            }

        # Load CF vectors once for RMSE/MAE
        user_cf, book_cf = await _load_cf_vectors(conn)

    # Initialize accumulators for multiple k values
    precisions_by_k: Dict[int, List[float]] = {k_val: [] for k_val in k_values}
    recalls_by_k: Dict[int, List[float]] = {k_val: [] for k_val in k_values}
    ndcgs_by_k: Dict[int, List[float]] = {k_val: [] for k_val in k_values}
    
    # For RMSE and MAE
    sq_errors: List[float] = []
    abs_errors: List[float] = []
    rmse_table: List[Dict[str, Union[float, str]]] = []  # actual, predicted, user_id, book_id
    rating_count = 0
    
    max_k = max(k_values) if k_values else k

    for user_id in test_users:
        test_interactions = test[user_id]
        relevant_ids = {inter.book_id for inter in test_interactions}

        # Get recommendations for the maximum k needed
        try:
            rec_books, metadata = await recommender.recommend_for_user(user_id, limit=max_k)
        except Exception:  # noqa: BLE001
            continue

        rec_ids_list = [b.id for b in rec_books]
        rec_ids_set = set(rec_ids_list)

        # Calculate metrics for each k value
        for k_val in k_values:
            # Precision@K
            top_k_recs = rec_ids_set if len(rec_ids_list) <= k_val else set(rec_ids_list[:k_val])
            hits = len(top_k_recs & relevant_ids)
            precisions_by_k[k_val].append(hits / float(k_val))
            
            # Recall@K
            recall = _calculate_recall_at_k(top_k_recs, relevant_ids, k_val)
            recalls_by_k[k_val].append(recall)
            
            # nDCG@K
            ndcg = _calculate_ndcg_at_k(rec_ids_list, relevant_ids, k_val)
            ndcgs_by_k[k_val].append(ndcg)

        # RMSE and MAE using CF vectors and ratings
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
            actual = inter.rating
            
            err = pred - actual
            sq_errors.append(err * err)
            abs_errors.append(abs(err))
            rating_count += 1
            
            # Store for RMSE table (limit to first 100 samples for display)
            if len(rmse_table) < 100:
                rmse_table.append({
                    "user_id": str(user_id),
                    "book_id": inter.book_id,
                    "actual": round(actual, 3),
                    "predicted": round(pred, 3),
                    "error": round(err, 3),
                })

    # Calculate average metrics for each k
    precision_at_k: Dict[int, Optional[float]] = {}
    recall_at_k: Dict[int, Optional[float]] = {}
    ndcg_at_k: Dict[int, Optional[float]] = {}
    
    for k_val in k_values:
        if precisions_by_k[k_val]:
            precision_at_k[k_val] = sum(precisions_by_k[k_val]) / len(precisions_by_k[k_val])
        else:
            precision_at_k[k_val] = None
            
        if recalls_by_k[k_val]:
            recall_at_k[k_val] = sum(recalls_by_k[k_val]) / len(recalls_by_k[k_val])
        else:
            recall_at_k[k_val] = None
            
        if ndcgs_by_k[k_val]:
            ndcg_at_k[k_val] = sum(ndcgs_by_k[k_val]) / len(ndcgs_by_k[k_val])
        else:
            ndcg_at_k[k_val] = None

    # Calculate RMSE and MAE
    rmse_value: Optional[float] = None
    if rating_count > 0 and sq_errors:
        rmse_value = math.sqrt(sum(sq_errors) / rating_count)
    
    mae_value: Optional[float] = None
    if rating_count > 0 and abs_errors:
        mae_value = sum(abs_errors) / len(abs_errors)

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "ndcg_at_k": ndcg_at_k,
        "rmse": rmse_value,
        "mae": mae_value,
        "rmse_table": rmse_table,
        "num_eval_users": len(test_users),
        "num_rating_samples": rating_count,
    }


