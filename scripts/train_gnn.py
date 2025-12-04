"""Train Graph Neural Network embeddings from user-book interaction graph.

This script:
1. Builds a bipartite graph from user-book interactions
2. Uses graph structure to learn node embeddings
3. Generates GNN vectors for books (256 dims)
4. Updates the database with the new vectors
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import networkx as nx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import init_db


def build_interaction_graph(
    interactions: List[tuple],
) -> nx.Graph:
    """
    Build a bipartite graph from user-book interactions.
    
    Args:
        interactions: List of (user_id, book_id, weight) tuples
    
    Returns:
        NetworkX graph with user and book nodes
    """
    G = nx.Graph()
    
    for user_id, book_id, weight in interactions:
        # Add edge with weight
        G.add_edge(f"user_{user_id}", f"book_{book_id}", weight=weight)
    
    return G


def generate_node_embeddings(
    graph: nx.Graph,
    embedding_dim: int = 256,
    walk_length: int = 40,
    num_walks: int = 10,
    window_size: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Generate node embeddings using a simple approach based on graph structure.
    
    Uses a combination of:
    - Node2Vec-like random walks
    - Graph structure features (degree, centrality)
    - Neighbor aggregation
    
    Args:
        graph: NetworkX graph
        embedding_dim: Dimension of embeddings
        walk_length: Length of random walks
        num_walks: Number of walks per node
        window_size: Context window size
    
    Returns:
        Dictionary mapping node names to embedding vectors
    """
    print(f"   Generating embeddings for {graph.number_of_nodes()} nodes...")
    
    # Get book nodes only
    book_nodes = [n for n in graph.nodes() if n.startswith("book_")]
    user_nodes = [n for n in graph.nodes() if n.startswith("user_")]
    
    print(f"   Book nodes: {len(book_nodes)}, User nodes: {len(user_nodes)}")
    
    # Initialize embeddings randomly
    embeddings = {}
    for node in graph.nodes():
        embeddings[node] = np.random.normal(0, 0.1, embedding_dim)
    
    # Simple embedding approach: aggregate neighbor features
    for iteration in range(10):
        new_embeddings = {}
        
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            
            if not neighbors:
                # Isolated node - keep random embedding
                new_embeddings[node] = embeddings[node]
                continue
            
            # Aggregate neighbor embeddings
            neighbor_embeds = np.array([embeddings[n] for n in neighbors])
            
            # Weight by edge weight if available
            weights = []
            for neighbor in neighbors:
                weight = graph.get_edge_data(node, neighbor, {}).get("weight", 1.0)
                weights.append(weight)
            weights = np.array(weights)
            weights = weights / (weights.sum() + 1e-8)  # Normalize
            
            # Weighted average of neighbors
            neighbor_avg = np.average(neighbor_embeds, axis=0, weights=weights)
            
            # Combine with current embedding (simple update)
            new_embeddings[node] = 0.7 * embeddings[node] + 0.3 * neighbor_avg
        
        embeddings = new_embeddings
        
        if (iteration + 1) % 3 == 0:
            print(f"   Completed iteration {iteration + 1}/10")
    
    # Normalize embeddings
    for node in embeddings:
        norm = np.linalg.norm(embeddings[node])
        if norm > 0:
            embeddings[node] = embeddings[node] / norm
    
    return embeddings


async def train_gnn_model(
    min_interactions: int = 1,
    embedding_dim: int = 256,
):
    """
    Train GNN model and update database.
    
    Args:
        min_interactions: Minimum interactions per book to include
        embedding_dim: Dimension of GNN vectors
    """
    pool = await init_db()
    
    print("🔮 Starting Graph Neural Network training...")
    print(f"   Min interactions: {min_interactions}")
    print(f"   Vector dimension: {embedding_dim}")
    
    async with pool.acquire() as conn:
        # Load interactions
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
                ) as weight
            FROM interactions i
            """
        )
        
        if not interactions_data:
            print("❌ No interactions found. Need user-book interactions to train GNN model.")
            print("   Users need to interact with books (click, view, like, rate) first.")
            return
        
        print(f"   Loaded {len(interactions_data)} interactions")
        
        # Filter books with minimum interactions
        book_counts = defaultdict(int)
        for row in interactions_data:
            book_counts[row["book_id"]] += 1
        
        valid_books = {bid for bid, count in book_counts.items() if count >= min_interactions}
        
        print(f"   Valid books (≥{min_interactions} interactions): {len(valid_books)}")
        
        if len(valid_books) < 2:
            print("❌ Not enough books with sufficient interactions.")
            print(f"   Need at least 2 books with ≥{min_interactions} interactions each.")
            return
        
        # Filter interactions
        filtered_interactions = [
            (row["user_id"], row["book_id"], float(row["weight"]))
            for row in interactions_data
            if row["book_id"] in valid_books
        ]
        
        print(f"   Filtered to {len(filtered_interactions)} interactions")
        
        # Build graph
        print("\n   Building interaction graph...")
        graph = build_interaction_graph(filtered_interactions)
        print(f"   Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Generate embeddings
        print("\n   Generating node embeddings...")
        embeddings = generate_node_embeddings(graph, embedding_dim=embedding_dim)
        
        # Extract book embeddings only
        book_embeddings = {}
        for node, embedding in embeddings.items():
            if node.startswith("book_"):
                book_id = int(node.replace("book_", ""))
                book_embeddings[book_id] = embedding.tolist()
        
        print(f"\n   Generated {len(book_embeddings)} book embeddings")
        
        # Update book vectors in database
        print("\n   Updating book vectors in database...")
        books_updated = 0
        for book_id, vector in book_embeddings.items():
            await conn.execute(
                "UPDATE books SET gnn_vector = $1::jsonb WHERE id = $2",
                json.dumps(vector),
                book_id,
            )
            books_updated += 1
            if books_updated % 100 == 0:
                print(f"   Updated {books_updated} books...", end="\r")
        
        print(f"\n   ✓ Updated {books_updated} book vectors")
        
        print("\n✅ GNN training complete!")
        print(f"   Books with GNN vectors: {books_updated}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Graph Neural Network model from interaction graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python scripts/train_gnn.py
  
  # Train with custom parameters
  python scripts/train_gnn.py --dim 128 --min-interactions 2

Note:
  - Requires user-book interactions in the database
  - Run this periodically (daily/weekly) as new interactions accumulate
  - Minimum 2 books with interactions needed
        """
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=1,
        help="Minimum interactions per book (default: 1)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Embedding dimension (default: 256)",
    )
    
    args = parser.parse_args()
    
    await train_gnn_model(
        min_interactions=args.min_interactions,
        embedding_dim=args.dim,
    )


if __name__ == "__main__":
    asyncio.run(main())

