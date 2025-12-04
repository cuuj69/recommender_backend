"""Train Collaborative Filtering model using ALS (Alternating Least Squares).

This script:
1. Loads user-book interactions from the database
2. Builds a user-item interaction matrix
3. Trains an ALS model to learn latent factors
4. Generates CF vectors for users (128 dims) and books (128 dims)
5. Updates the database with the new vectors
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import init_db


def als_factorization(
    interactions: List[Tuple[int, int, float]],
    n_factors: int = 128,
    n_iterations: int = 15,
    regularization: float = 0.1,
    learning_rate: float = 0.01,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Alternating Least Squares matrix factorization.
    
    Args:
        interactions: List of (user_idx, item_idx, rating) tuples
        n_factors: Number of latent factors (vector dimension)
        n_iterations: Number of training iterations
        regularization: L2 regularization parameter
        learning_rate: Learning rate
    
    Returns:
        Tuple of (user_vectors, item_vectors) dictionaries
    """
    # Build user-item matrix
    user_items = defaultdict(dict)
    item_users = defaultdict(dict)
    
    for user_idx, item_idx, rating in interactions:
        user_items[user_idx][item_idx] = rating
        item_users[item_idx][user_idx] = rating
    
    # Get unique users and items
    users = sorted(user_items.keys())
    items = sorted(item_users.keys())
    n_users = len(users)
    n_items = len(items)
    
    # Create index mappings
    user_to_idx = {uid: i for i, uid in enumerate(users)}
    item_to_idx = {iid: i for i, iid in enumerate(items)}
    
    # Initialize user and item factor matrices randomly
    user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
    item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
    
    print(f"   Training ALS with {n_users} users, {n_items} items, {len(interactions)} interactions")
    print(f"   Factors: {n_factors}, Iterations: {n_iterations}")
    
    # ALS iterations
    for iteration in range(n_iterations):
        # Update user factors (fix item factors)
        for user_idx, user_id in enumerate(users):
            items_for_user = list(user_items[user_id].keys())
            if not items_for_user:
                continue
            
            item_indices = [item_to_idx[iid] for iid in items_for_user]
            ratings = np.array([user_items[user_id][iid] for iid in items_for_user])
            
            # Solve: user_factor = (item_factors^T * item_factors + lambda*I)^-1 * item_factors^T * ratings
            item_subset = item_factors[item_indices]
            A = item_subset.T @ item_subset + regularization * np.eye(n_factors)
            b = item_subset.T @ ratings
            user_factors[user_idx] = np.linalg.solve(A, b)
        
        # Update item factors (fix user factors)
        for item_idx, item_id in enumerate(items):
            users_for_item = list(item_users[item_id].keys())
            if not users_for_item:
                continue
            
            user_indices = [user_to_idx[uid] for uid in users_for_item]
            ratings = np.array([item_users[item_id][uid] for uid in users_for_item])
            
            # Solve: item_factor = (user_factors^T * user_factors + lambda*I)^-1 * user_factors^T * ratings
            user_subset = user_factors[user_indices]
            A = user_subset.T @ user_subset + regularization * np.eye(n_factors)
            b = user_subset.T @ ratings
            item_factors[item_idx] = np.linalg.solve(A, b)
        
        if (iteration + 1) % 5 == 0:
            print(f"   Completed iteration {iteration + 1}/{n_iterations}")
    
    # Convert back to dictionaries with original IDs
    user_vectors = {users[i]: user_factors[i].tolist() for i in range(n_users)}
    item_vectors = {items[i]: item_factors[i].tolist() for i in range(n_items)}
    
    return user_vectors, item_vectors


async def train_cf_model(
    min_interactions: int = 2,
    n_factors: int = 128,
    n_iterations: int = 15,
):
    """
    Train collaborative filtering model and update database.
    
    Args:
        min_interactions: Minimum interactions per user/item to include
        n_factors: Number of latent factors (vector dimension)
        n_iterations: Number of ALS iterations
    """
    pool = await init_db()
    
    print("🔮 Starting Collaborative Filtering training...")
    print(f"   Min interactions: {min_interactions}")
    print(f"   Vector dimension: {n_factors}")
    
    async with pool.acquire() as conn:
        # Load interactions with ratings
        # Use rating if available, otherwise use interaction type weights
        interactions_data = await conn.fetch(
            """
            SELECT 
                i.user_id,
                i.book_id,
                COALESCE(i.rating, 
                    CASE 
                        WHEN i.interaction_type = 'rating' THEN 3.0
                        WHEN i.interaction_type = 'like' THEN 4.0
                        WHEN i.interaction_type = 'purchase' THEN 5.0
                        WHEN i.interaction_type = 'view' THEN 2.0
                        WHEN i.interaction_type = 'click' THEN 1.0
                        ELSE 2.5
                    END
                ) as score
            FROM interactions i
            """
        )
        
        if not interactions_data:
            print("❌ No interactions found. Need user-book interactions to train CF model.")
            print("   Users need to interact with books (click, view, like, rate) first.")
            return
        
        print(f"   Loaded {len(interactions_data)} interactions")
        
        # Filter users and items with minimum interactions
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        
        for row in interactions_data:
            user_counts[row["user_id"]] += 1
            item_counts[row["book_id"]] += 1
        
        # Get valid users and items
        valid_users = {uid for uid, count in user_counts.items() if count >= min_interactions}
        valid_items = {iid for iid, count in item_counts.items() if count >= min_interactions}
        
        print(f"   Valid users (≥{min_interactions} interactions): {len(valid_users)}")
        print(f"   Valid items (≥{min_interactions} interactions): {len(valid_items)}")
        
        if len(valid_users) < 2 or len(valid_items) < 2:
            print("❌ Not enough users or items with sufficient interactions.")
            print(f"   Need at least 2 users and 2 items with ≥{min_interactions} interactions each.")
            return
        
        # Filter interactions
        filtered_interactions = [
            (row["user_id"], row["book_id"], float(row["score"]))
            for row in interactions_data
            if row["user_id"] in valid_users and row["book_id"] in valid_items
        ]
        
        print(f"   Filtered to {len(filtered_interactions)} interactions")
        
        # Create mappings from UUID/int to sequential indices
        unique_users = sorted(valid_users)
        unique_items = sorted(valid_items)
        user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        item_id_to_idx = {iid: i for i, iid in enumerate(unique_items)}
        
        # Convert to indexed interactions
        indexed_interactions = [
            (user_id_to_idx[uid], item_id_to_idx[iid], score)
            for uid, iid, score in filtered_interactions
        ]
        
        # Train ALS model
        print("\n   Training ALS model...")
        user_vectors, item_vectors = als_factorization(
            indexed_interactions,
            n_factors=n_factors,
            n_iterations=n_iterations,
        )
        
        # Map back to original IDs
        user_vectors_by_id = {unique_users[i]: vec for i, vec in user_vectors.items()}
        item_vectors_by_id = {unique_items[i]: vec for i, vec in item_vectors.items()}
        
        print(f"\n   Generated {len(user_vectors_by_id)} user vectors")
        print(f"   Generated {len(item_vectors_by_id)} item vectors")
        
        # Update user vectors in database
        print("\n   Updating user vectors in database...")
        users_updated = 0
        for user_id, vector in user_vectors_by_id.items():
            await conn.execute(
                "UPDATE users SET cf_vector = $1::jsonb WHERE id = $2",
                json.dumps(vector),
                user_id,
            )
            users_updated += 1
            if users_updated % 100 == 0:
                print(f"   Updated {users_updated} users...", end="\r")
        
        print(f"\n   ✓ Updated {users_updated} user vectors")
        
        # Update book vectors in database
        print("   Updating book vectors in database...")
        books_updated = 0
        for book_id, vector in item_vectors_by_id.items():
            await conn.execute(
                "UPDATE books SET cf_embedding = $1::jsonb WHERE id = $2",
                json.dumps(vector),
                book_id,
            )
            books_updated += 1
            if books_updated % 100 == 0:
                print(f"   Updated {books_updated} books...", end="\r")
        
        print(f"\n   ✓ Updated {books_updated} book vectors")
        
        print("\n✅ CF training complete!")
        print(f"   Users with CF vectors: {users_updated}")
        print(f"   Books with CF embeddings: {books_updated}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Collaborative Filtering model using ALS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python scripts/train_cf.py
  
  # Train with custom parameters
  python scripts/train_cf.py --factors 64 --iterations 20 --min-interactions 3

Note:
  - Requires user-book interactions in the database
  - Run this periodically (daily/weekly) as new interactions accumulate
  - Minimum 2 users and 2 items with interactions needed
        """
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=2,
        help="Minimum interactions per user/item (default: 2)",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=128,
        help="Number of latent factors / vector dimension (default: 128)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of ALS iterations (default: 15)",
    )
    
    args = parser.parse_args()
    
    await train_cf_model(
        min_interactions=args.min_interactions,
        n_factors=args.factors,
        n_iterations=args.iterations,
    )


if __name__ == "__main__":
    asyncio.run(main())

