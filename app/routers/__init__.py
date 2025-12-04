"""API routers package."""
from fastapi import APIRouter

from . import auth, books, interactions, recommend, users

router = APIRouter()
router.include_router(auth.router)
router.include_router(users.router)
router.include_router(books.router)
router.include_router(interactions.router)
router.include_router(recommend.router)
