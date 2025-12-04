"""Embedding service using SentenceTransformers."""
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_path = str(settings.sentence_model_path)
        logger.info(f"Loading SentenceTransformer model from {model_path}")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        if settings.sentence_model_path.exists():
            try:
                _model = SentenceTransformer(str(settings.sentence_model_path))
                logger.info(f"Loaded custom model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom model, using default: {e}")
    return _model


def encode_text(text: str) -> List[float]:
    """Encode a single text string into an embedding vector."""
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


def encode_texts(texts: List[str]) -> List[List[float]]:
    """Encode multiple text strings into embedding vectors."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


def encode_kyc_preferences(preferences: dict) -> List[float]:
    """Encode user KYC preferences into an embedding vector."""
    if not preferences:
        return None
    
    # Build text representation from preferences
    parts = []
    if "genres" in preferences:
        genres = preferences.get('genres', [])
        if isinstance(genres, list):
            parts.append(f"Genres: {', '.join(genres)}")
        else:
            parts.append(f"Genres: {genres}")
    if "authors" in preferences:
        authors = preferences.get('authors', [])
        if isinstance(authors, list):
            parts.append(f"Authors: {', '.join(authors)}")
        else:
            parts.append(f"Authors: {authors}")
    if "age" in preferences:
        parts.append(f"Age: {preferences.get('age')}")
    # Handle both description and reading_preferences
    if "description" in preferences:
        parts.append(preferences.get("description", ""))
    elif "reading_preferences" in preferences:
        parts.append(preferences.get("reading_preferences", ""))
    
    text = " ".join(parts) if parts else ""
    if not text:
        return None
    
    return encode_text(text)

