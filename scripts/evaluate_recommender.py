"""Offline evaluation for the recommender system (RMSE + Precision@K).

This script:
1. Loads user–book interactions from the database
2. Creates a simple train/test split per user
3. Evaluates:
   - Precision@K using the full hybrid recommender
   - RMSE using CF vectors (if ratings and CF vectors exist)

Run from the project root (with venv activated):

    python scripts/evaluate_recommender.py --k 10 --min-interactions 5
"""

import argparse
import asyncio
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import asyncpg

# Make app importable when script is run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

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


async def load_interactions(conn: asyncpg.Connection) -> List[Interaction]:
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


def train_test_split_by_user(
    interactions: List[Interaction],
    min_interactions: int,
    test_ratio: float = 0.2,
) -> Tuple[Dict[UserId, List[Interaction]], Dict[UserId, List[Interaction]]]:
    """Simple temporal train/test split per user.

    We don't actually retrain the model on the train split here – we assume
    the current model is representative. The split is used only to define
    which interactions are used as ground truth for evaluation.
    """
    by_user: Dict[UserId, List[Interaction]] = defaultdict(list)
    for inter in interactions:
        by_user[inter.user_id].append(inter)

    train: Dict[UserId, List[Interaction]] = {}
    test: Dict[UserId, List[Interaction]] = {}

    for user_id, user_inters in by_user.items():
        if len(user_inters) < min_interactions:
            continue

        # Already sorted by created_at globally, but sort again per user for safety
        user_inters.sort(key=lambda x: x.created_at)
        n = len(user_inters)
        test_size = max(1, int(n * test_ratio))

        test[user_id] = user_inters[-test_size:]
        train[user_id] = user_inters[:-test_size]

    return train, test


async def load_cf_vectors(conn: asyncpg.Connection) -> Tuple[Dict[UserId, List[float]], Dict[int, List[float]]]:
    """Load CF vectors for users and books, if they exist."""
    user_vectors: Dict[UserId, List[float]] = {}
    book_vectors: Dict[int, List[float]] = {}

    user_rows = await conn.fetch("SELECT id, cf_vector FROM users WHERE cf_vector IS NOT NULL")
    for row in user_rows:
        vec = row["cf_vector"]
        if isinstance(vec, str):
            import json

            vec = json.loads(vec)
        user_vectors[row["id"]] = vec

    book_rows = await conn.fetch("SELECT id, cf_embedding FROM books WHERE cf_embedding IS NOT NULL")
    for row in book_rows:
        vec = row["cf_embedding"]
        if isinstance(vec, str):
            import json

            vec = json.loads(vec)
        book_vectors[row["id"]] = vec

    return user_vectors, book_vectors


async def evaluate(
    k: int = 10,
    min_interactions: int = 5,
) -> None:
    """Run evaluation and print metrics."""
    pool = await init_db()
    async with pool.acquire() as conn:
        print("📥 Loading interactions...")
        interactions = await load_interactions(conn)
        if not interactions:
            print("❌ No interactions found in database. Ask users to interact with books first.")
            return

        train, test = train_test_split_by_user(interactions, min_interactions=min_interactions)
        test_users = list(test.keys())
        if not test_users:
            print(f"❌ No users with at least {min_interactions} interactions for evaluation.")
            return

        print(f"✅ Using {len(test_users)} users for evaluation (min_interactions={min_interactions})")

        # Load CF vectors once for RMSE
        print("📥 Loading CF vectors for RMSE evaluation (if available)...")
        user_cf, book_cf = await load_cf_vectors(conn)
        if not user_cf or not book_cf:
            print("⚠️  CF vectors not found for many users/books – RMSE might be skipped or based on few samples.")

    # Precision@K and RMSE accumulators
    precisions: List[float] = []
    sq_errors: List[float] = []
    rating_count = 0

    print(f"\n🚀 Evaluating Precision@{k} and RMSE...")

    for idx, user_id in enumerate(test_users, start=1):
        test_interactions = test[user_id]

        # ---- Precision@K (ranking) using hybrid recommender ----
        try:
            rec_books = await recommender.recommend_for_user(user_id, limit=k)
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️  Skipping user {user_id} for Precision@K due to error: {e}")
            continue

        rec_ids = {b.id for b in rec_books}
        relevant_ids = {inter.book_id for inter in test_interactions}
        hits = len(rec_ids & relevant_ids)
        precisions.append(hits / float(k))

        # ---- RMSE using CF vectors (rating prediction) ----
        for inter in test_interactions:
            if inter.rating is None:
                continue
            if user_id not in user_cf or inter.book_id not in book_cf:
                continue

            u_vec = user_cf[user_id]
            b_vec = book_cf[inter.book_id]
            try:
                sim = cosine_similarity(u_vec, b_vec)  # in [-1, 1] or [0, 1] depending on normalization
            except Exception:  # noqa: BLE001
                continue

            # Map similarity to a pseudo-rating scale [1, 5]
            # If sim in [-1,1], shift/scale; if already [0,1], this is still OK.
            pred = 2.0 + 3.0 * max(-1.0, min(1.0, sim))
            err = pred - inter.rating
            sq_errors.append(err * err)
            rating_count += 1

        if idx % 20 == 0:
            print(f"   Processed {idx}/{len(test_users)} users...", end="\r")

    print("\n\n📊 Evaluation Results")

    if precisions:
        avg_precision = sum(precisions) / len(precisions)
        print(f"   Precision@{k}: {avg_precision:.4f} (over {len(precisions)} users)")
    else:
        print(f"   Precision@{k}: N/A (no valid users)")

    if rating_count > 0 and sq_errors:
        rmse = math.sqrt(sum(sq_errors) / rating_count)
        print(f"   RMSE (ratings): {rmse:.4f} (over {rating_count} rated interactions)")
    else:
        print("   RMSE: N/A (no ratings or CF vectors available)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recommender (Precision@K and RMSE)")
    parser.add_argument("--k", type=int, default=10, help="K for Precision@K (default: 10)")
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="Minimum interactions per user to include in evaluation (default: 5)",
    )
    args = parser.parse_args()

    asyncio.run(evaluate(k=args.k, min_interactions=args.min_interactions))


if __name__ == "__main__":
    main()


