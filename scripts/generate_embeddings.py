"""Script to generate embeddings for books that don't have them yet."""
import asyncio
import json
from typing import List, Optional

import asyncpg

from app.db.connection import init_db
from app.services.embedding_service import encode_text


def prepare_book_text(title: str, description: Optional[str], authors: List[str], categories: List[str]) -> str:
    """Prepare text for embedding by combining book metadata."""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if authors:
        parts.append(f"Authors: {', '.join(authors)}")
    if categories:
        parts.append(f"Categories: {', '.join(categories)}")
    if description:
        parts.append(description)
    return " ".join(parts)


async def generate_embeddings_for_books(
    batch_size: int = 100,
    limit: Optional[int] = None,
):
    """
    Generate content embeddings for books that don't have them.
    
    Args:
        batch_size: Number of books to process in each batch
        limit: Maximum number of books to process (None for all)
    """
    pool = await init_db()
    
    print("🔮 Starting embedding generation for books...")
    print(f"   Batch size: {batch_size}")
    if limit:
        print(f"   Limit: {limit} books")
    
    books_processed = 0
    books_updated = 0
    errors = 0
    
    async with pool.acquire() as conn:
        # Get books without embeddings
        query = """
            SELECT id, title, author, description, genres, metadata
            FROM books
            WHERE content_embedding IS NULL
            ORDER BY id
        """
        if limit:
            query += f" LIMIT {limit}"
        
        books = await conn.fetch(query)
        total_books = len(books)
        
        if total_books == 0:
            print("✅ All books already have embeddings!")
            return
        
        print(f"   Found {total_books} books without embeddings")
        
        for book in books:
            if limit and books_processed >= limit:
                break
            
            try:
                # Extract data
                title = book["title"] or ""
                description = book.get("description")
                author = book.get("author")
                authors = [author] if author else []
                
                # Parse genres from array or metadata
                genres = book.get("genres") or []
                if isinstance(genres, str):
                    try:
                        genres = json.loads(genres) if genres.startswith("[") else [genres]
                    except:
                        genres = [genres] if genres else []
                
                # Extract categories from metadata if available
                categories = list(genres) if genres else []
                if book.get("metadata"):
                    metadata = json.loads(book["metadata"]) if isinstance(book["metadata"], str) else book["metadata"]
                    if isinstance(metadata, dict) and "allAuthors" in metadata:
                        authors = metadata.get("allAuthors", authors)
                
                # Generate embedding
                book_text = prepare_book_text(title, description, authors, categories)
                if book_text:
                    embedding = encode_text(book_text)
                    
                    # Update book
                    await conn.execute(
                        """
                        UPDATE books
                        SET content_embedding = $1::jsonb
                        WHERE id = $2
                        """,
                        json.dumps(embedding),
                        book["id"],
                    )
                    books_updated += 1
                else:
                    print(f"⚠️  Skipping book {book['id']}: no text to embed")
                
                books_processed += 1
                if books_processed % batch_size == 0:
                    print(f"   ✓ Processed {books_processed}/{total_books} books ({books_updated} updated)...", end="\r")
            
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"\n⚠️  Error processing book {book['id']}: {e}")
                continue
        
        print(f"\n✅ Embedding generation complete!")
        print(f"   Books processed: {books_processed}")
        print(f"   Books updated: {books_updated}")
        print(f"   Errors: {errors}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate embeddings for books without them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for first 1000 books
  python scripts/generate_embeddings.py --limit 1000
  
  # Generate embeddings for all books (takes hours)
  python scripts/generate_embeddings.py
        """
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of books to process per batch (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of books to process (default: all)",
    )
    
    args = parser.parse_args()
    
    if not args.limit:
        print("⚠️  WARNING: This will process ALL books without embeddings (may take hours)!")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return
    
    await generate_embeddings_for_books(
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    asyncio.run(main())

