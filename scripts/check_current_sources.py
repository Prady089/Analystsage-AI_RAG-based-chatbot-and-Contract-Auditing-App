import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.vectorstore import VectorStore
from src.config import VECTOR_STORE_DIR

def check_sources():
    print(f"Checking vector store at: {VECTOR_STORE_DIR}")
    vs = VectorStore()
    stats = vs.get_stats()
    
    print(f"\nTotal Chunks: {stats['total_chunks']}")
    print(f"Unique Sources: {len(stats['sources'])}")
    print("\nSources List:")
    for source in stats['sources']:
        print(f" - {source}")
        
    # Check for anything related to "Techniques"
    techniques_sources = [s for s in stats['sources'] if "technique" in s.lower()]
    if techniques_sources:
        print("\nFound sources related to 'Techniques':")
        for s in techniques_sources:
            print(f" * {s}")
    else:
        print("\nNo sources found with 'Techniques' in the name.")

if __name__ == "__main__":
    check_sources()
