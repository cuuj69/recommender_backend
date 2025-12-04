"""Asyncpg connection utilities."""
import asyncio
from pathlib import Path
from typing import Optional

import asyncpg

from app.config import settings

_pool: Optional[asyncpg.pool.Pool] = None


async def ensure_database_exists() -> None:
    """Create the database if it doesn't exist."""
    # Azure PostgreSQL requires SSL
    ssl_required = "azure" in settings.pg_host.lower() or "postgres.database.azure.com" in settings.pg_host.lower()
    # Handle empty password for local development (use None for trust auth, empty string for no password)
    password = settings.pg_password.strip() if settings.pg_password and settings.pg_password.strip() else None
    
    # Connect to default 'postgres' database to check/create target database
    try:
        conn = await asyncpg.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            user=settings.pg_user,
            password=password,
            database="postgres",  # Connect to default database
            ssl="require" if ssl_required else None,
        )
        
        # Check if database exists
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            settings.pg_database
        )
        
        if not db_exists:
            # Create database (terminate connections first if any)
            await conn.execute(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{settings.pg_database}' AND pid <> pg_backend_pid()"
            )
            await conn.execute(f'CREATE DATABASE "{settings.pg_database}"')
            print(f"✓ Created database: {settings.pg_database}")
        else:
            print(f"✓ Database exists: {settings.pg_database}")
        
        await conn.close()
    except Exception as e:
        # If we can't connect to 'postgres', try connecting directly to target database
        # This handles cases where user doesn't have access to 'postgres' database
        print(f"Note: Could not check/create database via 'postgres' database: {e}")
        print(f"Attempting to connect directly to {settings.pg_database}...")


async def ensure_schema_exists(pool: asyncpg.pool.Pool) -> None:
    """Create tables and indexes if they don't exist."""
    async with pool.acquire() as conn:
        # Load schema SQL
        schema_path = Path(__file__).parent / "schema_basic.sql"
        if not schema_path.exists():
            print(f"⚠ Warning: Schema file not found at {schema_path}")
            return
        
        schema_sql = schema_path.read_text()
        
        # Check if tables already exist
        table_count = await conn.fetchval(
            """
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('users', 'books', 'interactions')
            """
        )
        
        if table_count < 3:
            # Create extension if needed (for UUID generation)
            try:
                await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            except Exception:
                # Extension might already exist or not be available, that's okay
                pass
            
            # Execute schema
            await conn.execute(schema_sql)
            print("✓ Database schema created")
        else:
            # Check if is_admin column exists, add it if not (migration)
            column_exists = await conn.fetchval(
                """
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'users' 
                AND column_name = 'is_admin'
                """
            )
            if not column_exists:
                await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE")
                print("✓ Added is_admin column to users table")
            print("✓ Database schema already exists")


async def init_db() -> asyncpg.pool.Pool:
    """Initialize database connection pool and ensure database/schema exist."""
    global _pool
    if _pool is None:
        # First, ensure the database exists
        await ensure_database_exists()
        
        # Azure PostgreSQL requires SSL
        ssl_required = "azure" in settings.pg_host.lower() or "postgres.database.azure.com" in settings.pg_host.lower()
        # Handle empty password for local development (use None for trust auth, empty string for no password)
        password = settings.pg_password.strip() if settings.pg_password and settings.pg_password.strip() else None
        
        # Create connection pool
        _pool = await asyncpg.create_pool(
            host=settings.pg_host,
            port=settings.pg_port,
            user=settings.pg_user,
            password=password,
            database=settings.pg_database,
            min_size=1,
            max_size=10,
            ssl="require" if ssl_required else None,
        )
        
        # Ensure schema exists (pass the pool we just created)
        await ensure_schema_exists(_pool)
    
    return _pool


async def get_pool() -> asyncpg.pool.Pool:
    if _pool is None:
        return await init_db()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def run_migrations(schema_sql: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)


def init_db_sync() -> None:
    asyncio.get_event_loop().run_until_complete(init_db())
