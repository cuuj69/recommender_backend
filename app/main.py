"""FastAPI entrypoint for the recommender service."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import analytics, auth, books, interactions, recommend, users

app = FastAPI(
    title="Book Recommender API",
    version="0.1.0",
    description="Hybrid recommendation engine combining content, CF, and GNN embeddings.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def healthcheck():
    """Basic health check."""
    return {"status": "ok", "env": settings.app_env}


@app.get("/health/db", tags=["health"])
async def db_healthcheck():
    """Database connectivity health check."""
    try:
        from app.db.connection import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Test query
            version = await conn.fetchval("SELECT version()")
            # Check tables
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
            )
            # Count records
            user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
            book_count = await conn.fetchval("SELECT COUNT(*) FROM books")
            interaction_count = await conn.fetchval("SELECT COUNT(*) FROM interactions")
            
            return {
                "status": "connected",
                "database": {
                    "version": version.split(",")[0] if version else "unknown",
                    "tables": [t["tablename"] for t in tables],
                    "counts": {
                        "users": user_count,
                        "books": book_count,
                        "interactions": interaction_count,
                    },
                },
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__,
        }


app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(books.router, prefix="/books", tags=["books"])
app.include_router(interactions.router, prefix="/interactions", tags=["interactions"])
app.include_router(recommend.router, prefix="/recommend", tags=["recommend"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
