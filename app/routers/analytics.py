"""Analytics endpoints for admin dashboard."""
import json
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db.connection import get_pool
from app.routers.users import get_current_user
from app.services import eval_service, graph_service

router = APIRouter()
security = HTTPBearer()


async def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Check if current user is admin."""
    user = await get_current_user(credentials)
    
    # Check is_admin field in database
    is_admin = user.get("is_admin", False)
    
    # Fallback: also check email for backward compatibility
    if not is_admin:
        email_lower = user.get("email", "").lower()
        if "admin" in email_lower:
            is_admin = True
    
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return user


@router.get("/metrics")
async def get_metrics(
    k: int = Query(10, ge=1, le=50, description="Default K for backward compatibility"),
    min_interactions: int = Query(5, ge=1, description="Minimum interactions for evaluation"),
    admin_user=Depends(get_admin_user)
):
    """Get global metrics (RMSE, MAE, Precision@K, Recall@K, nDCG@K, counts)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get counts
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        total_books = await conn.fetchval("SELECT COUNT(*) FROM books")
        total_interactions = await conn.fetchval("SELECT COUNT(*) FROM interactions")
        
        users_with_cf = await conn.fetchval("SELECT COUNT(*) FROM users WHERE cf_vector IS NOT NULL")
        books_with_content = await conn.fetchval("SELECT COUNT(*) FROM books WHERE content_embedding IS NOT NULL")
        books_with_cf = await conn.fetchval("SELECT COUNT(*) FROM books WHERE cf_embedding IS NOT NULL")
        books_with_gnn = await conn.fetchval("SELECT COUNT(*) FROM books WHERE gnn_vector IS NOT NULL")
        
        # Get evaluation metrics with multiple k values
        eval_results = await eval_service.evaluate_recommender(k=k, min_interactions=min_interactions)
    
    return {
        "metrics": {
            "rmse": eval_results.get("rmse"),
            "mae": eval_results.get("mae"),
            "precision_at_k": eval_results.get("precision_at_k", {}),
            "recall_at_k": eval_results.get("recall_at_k", {}),
            "ndcg_at_k": eval_results.get("ndcg_at_k", {}),
            "rmse_table": eval_results.get("rmse_table", []),
            "num_eval_users": eval_results.get("num_eval_users", 0),
            "num_rating_samples": eval_results.get("num_rating_samples", 0),
        },
        "counts": {
            "users": total_users,
            "books": total_books,
            "interactions": total_interactions,
            "users_with_cf_vectors": users_with_cf,
            "books_with_content_embeddings": books_with_content,
            "books_with_cf_embeddings": books_with_cf,
            "books_with_gnn_vectors": books_with_gnn,
        },
        "coverage": {
            "content_embeddings": round((books_with_content / total_books * 100) if total_books > 0 else 0, 2),
            "cf_embeddings": round((books_with_cf / total_books * 100) if total_books > 0 else 0, 2),
            "gnn_vectors": round((books_with_gnn / total_books * 100) if total_books > 0 else 0, 2),
            "user_cf_vectors": round((users_with_cf / total_users * 100) if total_users > 0 else 0, 2),
        }
    }


@router.get("/graph/overview")
async def get_overview_graph(
    max_users: int = Query(50, ge=1, le=200, description="Max users in graph"),
    max_books: int = Query(100, ge=1, le=500, description="Max books in graph"),
    admin_user=Depends(get_admin_user)
):
    """Get overview graph of users and books."""
    graph = await graph_service.build_overview_graph(max_users=max_users, max_books=max_books)
    return graph


@router.get("/graph/user/{user_id}")
async def get_user_graph(
    user_id: UUID,
    max_books: int = Query(20, ge=1, le=100, description="Max books in user graph"),
    admin_user=Depends(get_admin_user)
):
    """Get ego-graph for a specific user."""
    graph = await graph_service.build_user_graph(user_id, max_books=max_books)
    return graph


@router.get("/users")
async def list_users(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    admin_user=Depends(get_admin_user)
):
    """List all users (admin only)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        users = await conn.fetch(
            """
            SELECT 
                u.id, u.email, u.first_name, u.last_name, u.created_at, u.is_admin,
                COUNT(i.id) as interaction_count,
                CASE WHEN u.cf_vector IS NOT NULL THEN true ELSE false END as has_cf_vector,
                CASE WHEN u.kyc_embedding IS NOT NULL THEN true ELSE false END as has_kyc_embedding
            FROM users u
            LEFT JOIN interactions i ON i.user_id = u.id
            GROUP BY u.id, u.email, u.first_name, u.last_name, u.created_at, u.is_admin, u.cf_vector, u.kyc_embedding
            ORDER BY interaction_count DESC, u.created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset
        )
        
        total = await conn.fetchval("SELECT COUNT(*) FROM users")
    
    return {
        "users": [
            {
                "id": str(user["id"]),
                "email": user["email"],
                "first_name": user.get("first_name"),
                "last_name": user.get("last_name"),
                "created_at": user["created_at"].isoformat() if user["created_at"] else None,
                "is_admin": user.get("is_admin", False),
                "interaction_count": user["interaction_count"],
                "has_cf_vector": user["has_cf_vector"],
                "has_kyc_embedding": user["has_kyc_embedding"]
            }
            for user in users
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.put("/users/{user_id}/admin")
async def set_admin_status(
    user_id: UUID,
    is_admin: bool = Query(..., description="Set admin status"),
    admin_user=Depends(get_admin_user)
):
    """Set admin status for a user (admin only)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "UPDATE users SET is_admin = $1 WHERE id = $2 RETURNING id, email, is_admin",
            is_admin,
            user_id
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    
    return {
        "message": f"Admin status {'granted' if is_admin else 'revoked'}",
        "user": {
            "id": str(user["id"]),
            "email": user["email"],
            "is_admin": user["is_admin"]
        }
    }

