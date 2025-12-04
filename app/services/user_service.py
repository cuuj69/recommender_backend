"""User service helpers."""
import json
from typing import Optional, Union
from uuid import UUID

import asyncpg

from app.db.connection import get_pool
from app.utils.security import hash_password


async def create_user(
    email: str, 
    password: str, 
    first_name: Optional[str], 
    last_name: Optional[str],
    preferences: Optional[dict]
) -> asyncpg.Record:
    pool = await get_pool()
    password_hash = hash_password(password)
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            """
            INSERT INTO users (email, password_hash, first_name, last_name, kyc_preferences)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            email,
            password_hash,
            first_name,
            last_name,
            preferences,
        )
    return user


async def get_user_by_email(email: str) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM users WHERE email=$1", email)


async def get_user_by_id(user_id: Union[str, UUID]) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM users WHERE id=$1", user_id)


async def update_user_preferences(user_id: Union[str, UUID], preferences: dict) -> asyncpg.Record:
    """Update user preferences/KYC data."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # asyncpg handles dicts automatically for JSONB fields
        # Pass the dict directly, asyncpg will serialize it
        user = await conn.fetchrow(
            """
            UPDATE users
            SET kyc_preferences = $1::jsonb
            WHERE id = $2
            RETURNING *
            """,
            json.dumps(preferences) if preferences else None,
            user_id,
        )
    return user
