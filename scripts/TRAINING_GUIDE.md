# Training Scripts Guide

## Overview

The training scripts generate embeddings for collaborative filtering (CF) and graph neural networks (GNN) based on user-book interactions.

## Prerequisites

Before running training scripts, you need:
1. **User-book interactions** in the database
   - Users must have interacted with books (click, view, like, rate, purchase)
   - Minimum: 2 users and 2 books with interactions

2. **Books in database**
   - Books should be loaded (use `ingest_books.py`)

## Training Scripts

### 1. Collaborative Filtering (CF) Training

**Script:** `scripts/train_cf.py`

**What it does:**
- Learns latent factors from user-book interactions using ALS (Alternating Least Squares)
- Generates 128-dim vectors for users (`users.cf_vector`)
- Generates 128-dim vectors for books (`books.cf_embedding`)
- Enables collaborative filtering recommendations

**Usage:**
```bash
# Train with default settings
venv/bin/python scripts/train_cf.py

# Custom parameters
venv/bin/python scripts/train_cf.py --factors 64 --iterations 20 --min-interactions 3
```

**Parameters:**
- `--min-interactions`: Minimum interactions per user/item (default: 2)
- `--factors`: Vector dimension (default: 128)
- `--iterations`: Number of ALS iterations (default: 15)

**When to run:**
- After accumulating user interactions
- Periodically (daily/weekly) as new interactions come in
- When CF recommendations are needed

**Output:**
- Updates `users.cf_vector` (128-dim JSONB array)
- Updates `books.cf_embedding` (128-dim JSONB array)

---

### 2. Graph Neural Network (GNN) Training

**Script:** `scripts/train_gnn.py`

**What it does:**
- Builds a graph from user-book interactions
- Learns node embeddings using graph structure
- Generates 256-dim vectors for books (`books.gnn_vector`)
- Enables GNN-based recommendations

**Usage:**
```bash
# Train with default settings
venv/bin/python scripts/train_gnn.py

# Custom parameters
venv/bin/python scripts/train_gnn.py --dim 128 --min-interactions 2
```

**Parameters:**
- `--min-interactions`: Minimum interactions per book (default: 1)
- `--dim`: Embedding dimension (default: 256)

**When to run:**
- After accumulating user interactions
- Periodically (daily/weekly) as new interactions come in
- When GNN recommendations are needed

**Output:**
- Updates `books.gnn_vector` (256-dim JSONB array)

---

## Training Workflow

### Initial Setup

1. **Load books** (if not done):
   ```bash
   venv/bin/python scripts/ingest_books.py --skip-embeddings
   ```

2. **Generate content embeddings** (optional, for CBF):
   ```bash
   venv/bin/python scripts/generate_embeddings.py
   ```

3. **Users interact with books** via API:
   - Users sign up and log in
   - Users browse books
   - Users click/view/like/rate books (creates interactions)

### Training Cycle

Once you have interactions:

1. **Train CF model:**
   ```bash
   venv/bin/python scripts/train_cf.py
   ```

2. **Train GNN model:**
   ```bash
   venv/bin/python scripts/train_gnn.py
   ```

3. **Test recommendations:**
   - Use `/recommend` endpoint
   - Should now include CF and GNN recommendations

### Periodic Retraining

Run training scripts periodically to update embeddings with new interactions:

```bash
# Weekly retraining (example cron job)
0 2 * * 0 cd /path/to/reccommender && venv/bin/python scripts/train_cf.py
0 3 * * 0 cd /path/to/reccommender && venv/bin/python scripts/train_gnn.py
```

---

## How It Works

### Collaborative Filtering (CF)

1. **Matrix Factorization:**
   - Builds user-item interaction matrix
   - Factorizes into user factors (128-dim) and item factors (128-dim)
   - Uses ALS (Alternating Least Squares) algorithm

2. **Recommendations:**
   - User vector represents user preferences
   - Book vector represents book characteristics
   - Similar users/books have similar vectors
   - Recommendations based on cosine similarity

### Graph Neural Network (GNN)

1. **Graph Construction:**
   - Creates bipartite graph: users ↔ books
   - Edges weighted by interaction strength (rating, like, etc.)

2. **Embedding Learning:**
   - Aggregates neighbor information
   - Learns structural patterns in the graph
   - Generates 256-dim embeddings for books

3. **Recommendations:**
   - Books with similar graph neighborhoods have similar embeddings
   - User embedding = average of interacted book embeddings
   - Recommendations based on cosine similarity

---

## Troubleshooting

### "No interactions found"
- **Cause:** No user-book interactions in database
- **Solution:** Users need to interact with books first (click, view, like, rate)

### "Not enough users/items"
- **Cause:** Less than 2 users or 2 items with sufficient interactions
- **Solution:** Need more user activity, or lower `--min-interactions`

### Training takes too long
- **Cause:** Too many interactions or iterations
- **Solution:** 
  - Reduce `--iterations` for CF
  - Filter to recent interactions only
  - Run on a schedule (overnight)

### Recommendations still don't work
- **Check:** Do users have CF vectors? (`SELECT COUNT(*) FROM users WHERE cf_vector IS NOT NULL`)
- **Check:** Do books have CF/GNN embeddings? (`SELECT COUNT(*) FROM books WHERE cf_embedding IS NOT NULL`)
- **Solution:** Run training scripts and verify updates

---

## Performance Notes

- **CF Training:** ~1-5 minutes for 10k interactions, ~10-30 minutes for 100k interactions
- **GNN Training:** ~2-10 minutes for 10k interactions, ~20-60 minutes for 100k interactions
- **Memory:** Both scripts load all interactions into memory (should be fine for <1M interactions)

---

## Next Steps

After training:
1. ✅ CF recommendations will work (`/recommend` endpoint)
2. ✅ GNN recommendations will work
3. ✅ Hybrid recommendations combine CBF + CF + GNN
4. 🔄 Retrain periodically as new interactions accumulate

