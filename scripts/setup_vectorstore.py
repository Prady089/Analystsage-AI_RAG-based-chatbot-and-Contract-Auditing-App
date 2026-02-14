"""
Vector Store Setup Script

PURPOSE:
One-time script to index your documents into the vector store.

WHAT IT DOES:
1. Loads all documents from data/raw/
2. Chunks them into smaller pieces
3. Generates embeddings
4. Stores in ChromaDB vector store

RUN THIS:
- After adding new documents
- When starting fresh
- To rebuild the index

USAGE:
    python scripts/setup_vectorstore.py
    
    # Or with custom settings
    python scripts/setup_vectorstore.py --chunk-size 1000 --strategy recursive
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import load_document, detect_file_type
from src.chunker import chunk_documents
from src.vectorstore import VectorStore
from src.config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
import argparse


def find_documents(directory: Path) -> list:
    """
    Find all supported documents in a directory.
    
    Args:
        directory: Directory to search
    
    RETURNS:
        List of file paths
    """
    supported_extensions = ['*.pdf', '*.docx', '*.doc', '*.txt', '*.md']
    files = []
    
    for pattern in supported_extensions:
        files.extend(directory.glob(pattern))
    
    return files


def setup_vectorstore(
    data_dir: Path = RAW_DATA_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunking_strategy: str = "sentence_aware",
    clear_existing: bool = False
):
    """
    Set up the vector store with documents.
    
    Args:
        data_dir: Directory containing documents
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        chunking_strategy: Strategy to use (fixed, sentence_aware, paragraph, recursive)
        clear_existing: Whether to clear existing data first
    
    PROCESS:
    1. Find all documents
    2. Load each document
    3. Chunk the documents
    4. Add to vector store
    """
    print("\n" + "="*60)
    print("VECTOR STORE SETUP")
    print("="*60 + "\n")
    
    # Initialize vector store
    print("📚 Initializing vector store...")
    vector_store = VectorStore()
    
    # Clear existing data if requested
    if clear_existing:
        print("\n🗑️  Clearing existing data...")
        vector_store.clear()
    
    # Find documents
    print(f"\n📂 Searching for documents in: {data_dir}")
    document_files = find_documents(data_dir)
    
    if not document_files:
        print(f"\n❌ No documents found in {data_dir}")
        print(f"\nSupported formats: PDF, DOCX, TXT, MD")
        print(f"\nPlease add documents to {data_dir} and try again.")
        return
    
    print(f"   Found {len(document_files)} document(s):")
    for file in document_files:
        file_type = detect_file_type(str(file))
        print(f"   - {file.name} ({file_type})")
    
    # Process each document
    all_chunks = []
    
    for file_path in document_files:
        print(f"\n📄 Processing: {file_path.name}")
        
        try:
            # Load document
            print(f"   Loading...")
            documents = load_document(str(file_path))
            print(f"   ✓ Loaded {len(documents)} document(s)")
            
            # Chunk documents
            print(f"   Chunking (strategy: {chunking_strategy})...")
            chunks = chunk_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=chunking_strategy
            )
            print(f"   ✓ Created {len(chunks)} chunks")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {e}")
            continue
    
    # Add all chunks to vector store
    if all_chunks:
        print(f"\n💾 Adding {len(all_chunks)} chunks to vector store...")
        vector_store.add_chunks(all_chunks)
        
        # Show statistics
        print(f"\n📊 VECTOR STORE STATISTICS:")
        stats = vector_store.get_stats()
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Documents indexed: {len(stats['sources'])}")
        print(f"   Sources:")
        for source in stats['sources']:
            print(f"      - {source}")
        print(f"   Storage location: {stats['persist_directory']}")
        
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("\nYour documents are now indexed and ready for querying!")
        print("You can now run the RAG application to ask questions.")
        
    else:
        print("\n⚠️  No chunks were created. Check your documents.")


def main():
    """
    Main entry point with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Set up vector store with documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (use defaults)
  python scripts/setup_vectorstore.py
  
  # Custom chunk size
  python scripts/setup_vectorstore.py --chunk-size 1000
  
  # Different chunking strategy
  python scripts/setup_vectorstore.py --strategy recursive
  
  # Clear existing data first
  python scripts/setup_vectorstore.py --clear
  
  # Combine options
  python scripts/setup_vectorstore.py --chunk-size 1200 --overlap 300 --strategy paragraph --clear
        """
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=CHUNK_SIZE,
        help=f'Size of text chunks (default: {CHUNK_SIZE})'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=CHUNK_OVERLAP,
        help=f'Overlap between chunks (default: {CHUNK_OVERLAP})'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['fixed', 'sentence_aware', 'paragraph', 'recursive'],
        default='sentence_aware',
        help='Chunking strategy (default: sentence_aware)'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing vector store data before adding'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=RAW_DATA_DIR,
        help=f'Directory containing documents (default: {RAW_DATA_DIR})'
    )
    
    args = parser.parse_args()
    
    # Run setup
    try:
        setup_vectorstore(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            chunking_strategy=args.strategy,
            clear_existing=args.clear
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
