# Book Recommender API

A hybrid book recommendation platform that combines Content-Based Filtering (CBF), Collaborative Filtering (CF), and Graph Neural Network (GNN) approaches to provide personalized book recommendations.

## Features

- **Hybrid Recommendation Engine**: Combines CBF, CF, and GNN for accurate recommendations
- **User Authentication**: JWT-based auth with signup/login
- **KYC Preferences**: Users can specify preferences (genres, authors, etc.) for better recommendations
- **Vector Embeddings**: Uses SentenceTransformers for content-based recommendations
- **PostgreSQL + JSONB**: Stores vectors as JSONB arrays (pgvector optional)
- **FastAPI**: Modern async API with automatic OpenAPI documentation

## Architecture

```
┌────────────┐
│  FastAPI   │
│ Recommender│
└──────┬─────┘
       │ asyncpg queries
       ▼
┌────────────┐
│ PostgreSQL │
│ + JSONB    │
└────────────┘
       ▲
       │ batch updates
       ▼
┌───────────────────────┐
│ Training Jobs (Python)│
│ - Embeddings (SBERT)  │
│ - CF (ALS)            │
│ - GNN (PyG)           │
└───────────────────────┘
```

## Tech Stack

- **FastAPI** - Web framework
- **PostgreSQL** - Database (Azure PostgreSQL)
- **asyncpg** - Async PostgreSQL driver
- **SentenceTransformers** - Text embeddings
- **scikit-learn** - Collaborative filtering
- **NetworkX** - Graph operations
- **Pydantic v2** - Data validation

## Prerequisites

- Python 3.12+
- PostgreSQL database (Azure PostgreSQL or local)
- pip

## Setup

### 1. Clone and Install Dependencies

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
APP_ENV=development
PGHOST=your-postgres-host
PGPORT=5432
PGDATABASE=your_database
PGUSER=your_username
PGPASSWORD=your_password
PGVECTOR_SCHEMA=public
MODEL_DIR=/app/models
SENTENCE_MODEL_PATH=/app/models/sentence_model
JWT_SECRET=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### 3. Database Setup

The database schema will be created automatically. To manually set it up:

```bash
python -c "
import asyncio
from app.db.connection import init_db
from pathlib import Path

async def setup():
    pool = await init_db()
    async with pool.acquire() as conn:
        schema = Path('app/db/schema_basic.sql').read_text()
        await conn.execute(schema)
    print('✓ Database schema created')

asyncio.run(setup())
"
```

### 4. Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Checks
- `GET /health` - Basic health check
- `GET /health/db` - Database connectivity check

### Authentication
- `POST /auth/signup` - Create new user account
- `POST /auth/login` - Login and get JWT token

### Users
- `GET /users/{user_id}` - Get user profile

### Recommendations
- `POST /recommend` - Get personalized book recommendations

### Books (to be implemented)
- `GET /books/{id}` - Get book details
- `POST /interactions` - Log user interactions

## Project Structure

```
book-recommender/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py            # Configuration & env vars
│   ├── db/
│   │   ├── connection.py    # Database connection pool
│   │   └── schema_basic.sql # Database schema (JSONB)
│   ├── models/              # Pydantic models
│   │   ├── book_model.py
│   │   ├── user_model.py
│   │   └── interaction_model.py
│   ├── routers/             # API endpoints
│   │   ├── auth.py
│   │   ├── recommend.py
│   │   └── users.py
│   ├── services/            # Business logic
│   │   ├── auth_service.py
│   │   ├── cf_service.py
│   │   ├── content_service.py
│   │   ├── embedding_service.py
│   │   ├── gnn_service.py
│   │   ├── recommender.py
│   │   └── user_service.py
│   └── utils/               # Utilities
│       ├── logger.py
│       ├── preprocessing.py
│       ├── security.py
│       └── vector_ops.py
├── scripts/                 # Data ingestion & training
│   ├── ingest_books.py
│   ├── train_cf.py
│   └── train_gnn.py
├── data/                    # Data files
├── models/                  # Saved models
├── requirements.txt
└── README.md
```

## Database Schema

### Users Table
- `id` - Primary key
- `email` - Unique email
- `password_hash` - Bcrypt hashed password
- `name` - User name
- `kyc_preferences` - JSONB preferences (genres, authors, etc.)
- `kyc_embedding` - JSONB array (768 dims) - content embedding
- `cf_vector` - JSONB array (128 dims) - collaborative filtering vector

### Books Table
- `id` - Primary key
- `title`, `author`, `description` - Book metadata
- `genres` - Text array
- `content_embedding` - JSONB array (768 dims)
- `cf_embedding` - JSONB array (128 dims)
- `gnn_vector` - JSONB array (256 dims)

### Interactions Table
- `id` - Primary key
- `user_id`, `book_id` - Foreign keys
- `interaction_type` - click, like, view, etc.
- `rating` - Optional numeric rating

## Recommendation Flow

1. **Content-Based**: Uses SentenceTransformers to encode user KYC preferences and book descriptions, finds similar books
2. **Collaborative Filtering**: Uses user-item interaction matrix to find similar users/books
3. **GNN**: Uses graph structure of user-book interactions for recommendations
4. **Hybrid**: Combines all three approaches with weighted scoring

## Development

### Running Tests
```bash
# (Tests to be added)
pytest
```

### Code Style
```bash
# (Linting to be configured)
black app/
flake8 app/
```

