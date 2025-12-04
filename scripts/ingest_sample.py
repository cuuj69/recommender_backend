"""Quick script to ingest a small sample of books for development/testing."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_books import ingest_books


async def main():
    """Ingest a small sample for development."""
    csv_path = Path("books_data.csv")
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print("🚀 Ingesting sample books for development...")
    print("   This will load 1,000 books (no embeddings) for quick testing")
    
    await ingest_books(
        csv_path=csv_path,
        batch_size=100,
        limit=1000,  # Small sample for dev
        skip_embeddings=True,  # Fast, no CPU-intensive work
    )
    
    print("\n✅ Sample books loaded! You can now test the API.")
    print("   To load more books, run: python scripts/ingest_books.py --limit 10000")


if __name__ == "__main__":
    asyncio.run(main())

