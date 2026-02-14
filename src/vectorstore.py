"""
Vector Store Module

PURPOSE:
Stores and retrieves document chunks using semantic similarity search.

WHAT IS A VECTOR STORE:
A specialized database that stores:
1. Text chunks (the actual content)
2. Embeddings (numerical vectors)
3. Metadata (source, page, etc.)

And enables:
- Similarity search: Find chunks similar to a query
- Filtering: Search within specific documents
- Persistence: Save and load the database

WHY CHROMADB:
- Local storage (no cloud required)
- Fast similarity search
- Persistent (survives restarts)
- Simple API
- Free and open source

ANALOGY:
Traditional DB: "Find documents WHERE title = 'Policy'"
Vector Store: "Find documents SIMILAR TO 'remote work guidelines'"
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings

from .embeddings import get_embeddings
from .chunker import Chunk
from .config import VECTOR_STORE_PATH, COLLECTION_NAME


class VectorStore:
    """
    Manages document storage and retrieval using ChromaDB.
    
    WHAT IT DOES:
    - Stores document chunks with embeddings
    - Searches for similar chunks
    - Filters by metadata
    - Persists to disk
    
    ARCHITECTURE:
    ┌─────────────────────────────────────┐
    │ User Query: "remote work policy"    │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │ Embeddings: Convert to vector       │
    │ [0.23, -0.45, 0.67, ...]           │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │ Vector Store: Find similar vectors  │
    │ Compare with stored embeddings      │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │ Results: Top-k most similar chunks  │
    │ 1. "Remote work policy allows..."   │
    │ 2. "Employees may work from home..."│
    │ 3. "Work from home guidelines..."   │
    └─────────────────────────────────────┘
    
    USAGE:
        # Initialize
        vector_store = VectorStore()
        
        # Add documents
        vector_store.add_chunks(chunks)
        
        # Search
        results = vector_store.search("remote work policy", top_k=5)
    """
    
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = VECTOR_STORE_PATH,
        embeddings: Optional[OllamaEmbeddings] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection (like a table in SQL)
            persist_directory: Where to save the database
            embeddings: Embeddings instance (creates one if not provided)
        
        COLLECTIONS:
        Think of collections like tables in SQL:
        - "documents" collection: All your documents
        - "legal" collection: Just legal contracts
        - "standard" collection: Reference standards
        
        You can have multiple collections in one database.
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = embeddings or get_embeddings()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,  # Don't send usage data
                allow_reset=True
            )
        )
        
        # Get or create collection
        # hnsw:space can be 'l2', 'ip', or 'cosine'
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Document chunks for RAG system",
                "hnsw:space": "cosine"
            }
        )
        
        print(f"📚 Vector Store initialized:")
        print(f"   Collection: {collection_name}")
        print(f"   Location: {self.persist_directory}")
        print(f"   Existing chunks: {self.collection.count()}")
    
    
    def add_chunks(self, chunks: List[Chunk], show_progress: bool = True):
        """
        Add document chunks to the vector store.
        
        PROCESS:
        1. Extract text from chunks
        2. Generate embeddings for all texts
        3. Store in ChromaDB with metadata
        
        Args:
            chunks: List of Chunk objects
            show_progress: Show progress messages
        
        WHAT GETS STORED:
        - IDs: Unique chunk identifiers
        - Documents: The actual text content
        - Embeddings: Numerical vectors
        - Metadata: Source, page, chunk_index, etc.
        
        EXAMPLE:
            chunks = chunk_documents(documents)
            vector_store.add_chunks(chunks)
        """
        if not chunks:
            print("⚠️  No chunks to add")
            return
        
        if show_progress:
            print(f"\n📥 Adding {len(chunks)} chunks to vector store...")
        
        # Extract data from chunks
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        if show_progress:
            print("   Generating embeddings...")
        embeddings = self.embeddings.embed_documents(documents)
        
        # Add to ChromaDB
        if show_progress:
            print("   Storing in database...")
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        if show_progress:
            print(f"✓ Added {len(chunks)} chunks")
            print(f"   Total chunks in store: {self.collection.count()}")
    
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for chunks similar to the query.
        
        HOW IT WORKS:
        1. Convert query to embedding
        2. Find chunks with similar embeddings
        3. Return top-k most similar
        
        Args:
            query: Search query (natural language)
            top_k: Number of results to return
            filter_dict: Filter by metadata (e.g., {"source": "policy.pdf"})
        
        RETURNS:
            List of dictionaries with:
            - id: Chunk ID
            - content: Chunk text
            - metadata: Chunk metadata
            - distance: Similarity score (lower = more similar)
        
        EXAMPLE:
            # Basic search
            results = vector_store.search("remote work policy", top_k=5)
            
            # Search within specific document
            results = vector_store.search(
                "remote work",
                top_k=3,
                filter_dict={"source": "employee_handbook.pdf"}
            )
            
            # Use results
            for result in results:
                print(f"Source: {result['metadata']['source']}")
                print(f"Content: {result['content'][:200]}...")
                print(f"Similarity: {result['distance']}")
        """
        # Check if collection is empty
        if self.collection.count() == 0:
            print("⚠️  Vector store is empty. Add documents first.")
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict  # Optional metadata filter
        )
        
        # Format results
        formatted_results = []
        
        # ChromaDB returns results in a specific format
        ids = results['ids'][0] if results['ids'] else []
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        for i in range(len(ids)):
            formatted_results.append({
                'id': ids[i],
                'content': documents[i],
                'metadata': metadatas[i],
                'distance': distances[i]  # Lower = more similar
            })
        
        return formatted_results
    
    
    def delete_by_source(self, source: str):
        """
        Delete all chunks from a specific source document.
        
        USE CASE:
        User wants to remove a document from the system.
        
        Args:
            source: Source filename (e.g., "old_policy.pdf")
        
        EXAMPLE:
            vector_store.delete_by_source("old_policy.pdf")
        """
        # Get all IDs for this source
        results = self.collection.get(
            where={"source": source}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"✓ Deleted {len(results['ids'])} chunks from '{source}'")
        else:
            print(f"⚠️  No chunks found for source '{source}'")
    
    
    def list_sources(self) -> List[str]:
        """
        List all unique source documents in the vector store.
        
        RETURNS:
            List of source filenames
        
        EXAMPLE:
            sources = vector_store.list_sources()
            print(f"Documents in store: {sources}")
            # ['policy.pdf', 'handbook.docx', 'guide.txt']
        """
        # Get all metadata
        all_data = self.collection.get()
        
        if not all_data['metadatas']:
            return []
        
        # Extract unique sources
        sources = set()
        for metadata in all_data['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
        
        return sorted(list(sources))
    
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        RETURNS:
            Dictionary with stats:
            - total_chunks: Total number of chunks
            - sources: List of source documents
            - collection_name: Name of the collection
        
        EXAMPLE:
            stats = vector_store.get_stats()
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Documents: {stats['sources']}")
        """
        return {
            'total_chunks': self.collection.count(),
            'sources': self.list_sources(),
            'collection_name': self.collection_name,
            'persist_directory': str(self.persist_directory)
        }
    
    
    def clear(self):
        """
        Clear all data from the collection.
        
        ⚠️  WARNING: This deletes everything!
        
        USE CASE:
        Starting fresh, removing all documents.
        
        EXAMPLE:
            vector_store.clear()
        """
        # Delete the collection
        self.client.delete_collection(self.collection_name)
        
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document chunks for RAG system"}
        )
        
        print(f"✓ Cleared collection '{self.collection_name}'")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_vector_store(
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = VECTOR_STORE_PATH
) -> VectorStore:
    """
    Create a vector store instance (convenience function).
    
    Args:
        collection_name: Collection name
        persist_directory: Storage location
    
    RETURNS:
        VectorStore instance
    
    EXAMPLE:
        vector_store = create_vector_store()
    """
    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test the vector store module.
    
    Run: python src/vectorstore.py
    """
    print("\n" + "="*60)
    print("TESTING VECTOR STORE")
    print("="*60 + "\n")
    
    try:
        # Initialize vector store
        print("1. Initializing vector store...")
        vector_store = VectorStore(collection_name="test_collection")
        
        # Create test chunks
        print("\n2. Creating test chunks...")
        from src.chunker import Chunk
        
        test_chunks = [
            Chunk(
                content="AnalystSage AI is a premium document intelligence suite.",
                metadata={"source": "test.txt", "page": 1},
                chunk_id="test_0_0"
            ),
            Chunk(
                content="Retrieval Augmented Generation combines LLMs with external data.",
                metadata={"source": "test.txt", "page": 1},
                chunk_id="test_0_1"
            ),
            Chunk(
                content="Semantic search allows finding content by meaning instead of keywords.",
                metadata={"source": "test.txt", "page": 2},
                chunk_id="test_0_2"
            )
        ]
        
        # Add chunks
        print("\n3. Adding chunks to vector store...")
        vector_store.add_chunks(test_chunks)
        
        # Get stats
        print("\n4. Vector store statistics:")
        stats = vector_store.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test search
        print("\n5. Testing search...")
        queries = [
            "What is business analysis?",
            "How to gather requirements?",
            "stakeholder engagement"
        ]
        
        for query in queries:
            print(f"\n   Query: '{query}'")
            results = vector_store.search(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"   Result {i}:")
                print(f"      Content: {result['content'][:80]}...")
                print(f"      Distance: {result['distance']:.4f}")
        
        # Clean up
        print("\n6. Cleaning up test collection...")
        vector_store.clear()
        
        print("\n" + "="*60)
        print("✓ Vector store test complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
