"""Ingest books from CSV file into the database."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import init_db
from app.services.embedding_service import encode_text


def parse_list_field(value: str) -> List[str]:
    """Parse a string that looks like a Python list into an actual list."""
    if not value or pd.isna(value):
        return []
    
    # If it's already a list string like "['item1', 'item2']"
    if isinstance(value, str) and value.startswith('['):
        try:
            # Use eval for simple list strings (safe in this context)
            result = eval(value)
            if isinstance(result, list):
                return [str(item).strip() for item in result if item]
            return []
        except:
            # If eval fails, try to parse manually
            value = value.strip('[]').replace("'", "").replace('"', '')
            return [item.strip() for item in value.split(',') if item.strip()]
    
    # If it's a comma-separated string
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    
    return []


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


async def ingest_books(
    csv_path: Path,
    batch_size: int = 100,
    limit: Optional[int] = None,
    skip_embeddings: bool = False,
):
    """
    Ingest books from CSV file into the database.
    
    Args:
        csv_path: Path to the CSV file
        batch_size: Number of books to insert in each batch
        limit: Maximum number of books to ingest (None for all)
        skip_embeddings: If True, skip generating embeddings (faster)
    """
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"📚 Starting book ingestion from {csv_path}")
    print(f"   Batch size: {batch_size}")
    if limit:
        print(f"   Limit: {limit} books")
    print(f"   Skip embeddings: {skip_embeddings}")
    
    # Initialize database connection
    pool = await init_db()
    
    # Read CSV file
    print("\n📖 Reading CSV file...")
    try:
        df = pd.read_csv(csv_path, nrows=limit)
        print(f"   Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Process books in batches
    total_inserted = 0
    total_errors = 0
    
    async with pool.acquire() as conn:
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch = df.iloc[batch_start:batch_end]
            
            print(f"\n   Processing batch {batch_start // batch_size + 1} (rows {batch_start+1}-{batch_end})...")
            
            batch_books = []
            for idx, row in batch.iterrows():
                try:
                    # Extract and parse fields
                    title = str(row.get('Title', '')).strip()
                    if not title:
                        continue
                    
                    description = str(row.get('description', '')).strip() if pd.notna(row.get('description')) else None
                    if description == '':
                        description = None
                    
                    # Parse authors (can be list or string)
                    authors_raw = row.get('authors', '')
                    authors = parse_list_field(authors_raw)
                    author = authors[0] if authors else None
                    
                    # Parse categories/genres
                    categories_raw = row.get('categories', '')
                    genres = parse_list_field(categories_raw)
                    
                    # Build metadata JSONB object
                    metadata = {}
                    if pd.notna(row.get('image')):
                        metadata['image'] = str(row.get('image'))
                    if pd.notna(row.get('previewLink')):
                        metadata['previewLink'] = str(row.get('previewLink'))
                    if pd.notna(row.get('infoLink')):
                        metadata['infoLink'] = str(row.get('infoLink'))
                    if pd.notna(row.get('publisher')):
                        metadata['publisher'] = str(row.get('publisher'))
                    if pd.notna(row.get('publishedDate')):
                        metadata['publishedDate'] = str(row.get('publishedDate'))
                    if pd.notna(row.get('ratingsCount')):
                        try:
                            metadata['ratingsCount'] = float(row.get('ratingsCount'))
                        except:
                            pass
                    if len(authors) > 1:
                        metadata['allAuthors'] = authors
                    
                    # Generate embedding if not skipping
                    content_embedding = None
                    if not skip_embeddings:
                        try:
                            book_text = prepare_book_text(title, description, authors, genres)
                            if book_text:
                                content_embedding = encode_text(book_text)
                        except Exception as e:
                            print(f"      ⚠️  Failed to generate embedding for '{title}': {e}")
                    
                    batch_books.append({
                        'title': title,
                        'author': author,
                        'description': description,
                        'genres': genres if genres else None,
                        'content_embedding': json.dumps(content_embedding) if content_embedding else None,
                        'metadata': json.dumps(metadata) if metadata else None,
                    })
                    
                except Exception as e:
                    print(f"      ⚠️  Error processing row {idx}: {e}")
                    total_errors += 1
                    continue
            
            # Insert batch into database
            if batch_books:
                try:
                    # Use a transaction for the batch
                    async with conn.transaction():
                        for book in batch_books:
                            await conn.execute(
                                """
                                INSERT INTO books (title, author, description, genres, content_embedding, metadata)
                                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
                                ON CONFLICT DO NOTHING
                                """,
                                book['title'],
                                book['author'],
                                book['description'],
                                book['genres'],
                                book['content_embedding'],
                                book['metadata'],
                            )
                    
                    total_inserted += len(batch_books)
                    print(f"      ✅ Inserted {len(batch_books)} books (total: {total_inserted})")
                    
                except Exception as e:
                    print(f"      ❌ Error inserting batch: {e}")
                    total_errors += len(batch_books)
    
    print(f"\n✅ Book ingestion complete!")
    print(f"   Total books inserted: {total_inserted}")
    if total_errors > 0:
        print(f"   Errors: {total_errors}")
    
    if skip_embeddings:
        print(f"\n💡 Tip: Generate embeddings later with:")
        print(f"   python scripts/generate_embeddings.py")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest books from CSV file into the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest 1,000 books without embeddings (fast)
  python scripts/ingest_books.py --limit 1000 --skip-embeddings
  
  # Ingest all books with embeddings (slow)
  python scripts/ingest_books.py
  
  # Ingest 10,000 books with embeddings
  python scripts/ingest_books.py --limit 10000
        """
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="books_data.csv",
        help="Path to CSV file (default: books_data.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of books to insert per batch (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of books to ingest (default: all)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip generating embeddings (much faster)",
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent.parent / csv_path
    
    await ingest_books(
        csv_path=csv_path,
        batch_size=args.batch_size,
        limit=args.limit,
        skip_embeddings=args.skip_embeddings,
    )


if __name__ == "__main__":
    asyncio.run(main())

