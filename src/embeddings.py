"""
Embeddings Module

PURPOSE:
Converts text into numerical vectors (embeddings) that capture semantic meaning.

WHY EMBEDDINGS:
Text: "How do I gather requirements?"
Embedding: [0.23, -0.45, 0.67, 0.12, ...] (768 numbers)

Similar text = Similar numbers:
"gather requirements" → [0.23, -0.45, 0.67, ...]
"collect requirements" → [0.24, -0.44, 0.68, ...]  ← Very similar!
"database design" → [-0.67, 0.89, -0.23, ...]      ← Very different!

This enables semantic search: Find text by MEANING, not just keywords.

EMBEDDING MODEL:
We use Ollama's "nomic-embed-text" model:
- Runs locally (100% private)
- 768-dimensional vectors
- Optimized for retrieval tasks
- Free and fast
"""

from typing import List, Optional
import requests
import json

from .config import (
    OLLAMA_BASE_URL, 
    OLLAMA_EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL
)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OllamaEmbeddings:
    """
    Generate embeddings using Ollama's embedding models.
    
    WHAT IT DOES:
    - Connects to local Ollama server
    - Sends text to embedding model
    - Returns numerical vectors
    
    WHY OLLAMA:
    - 100% local (no data sent to cloud)
    - Free (no API costs)
    - Fast (runs on your hardware)
    - Private (your documents stay on your machine)
    
    USAGE:
        embedder = OllamaEmbeddings()
        
        # Embed single text
        vector = embedder.embed_query("What is semantic intelligence?")
        
        # Embed multiple texts
        vectors = embedder.embed_documents([
            "Document 1 content",
            "Document 2 content",
            "Document 3 content"
        ])
    """
    
    def __init__(
        self,
        model: str = OLLAMA_EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL
    ):
        """
        Initialize Ollama embeddings.
        
        Args:
            model: Embedding model name (default: nomic-embed-text)
            base_url: Ollama server URL (default: http://localhost:11434)
        
        MODELS AVAILABLE:
        - nomic-embed-text (recommended): 768 dimensions, optimized for retrieval
        - all-minilm: 384 dimensions, smaller/faster
        - mxbai-embed-large: 1024 dimensions, highest quality
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.embed_endpoint = f"{self.base_url}/api/embeddings"
        
        # Verify Ollama is running
        self._verify_connection()
        
        print(f"🤖 Embeddings initialized:")
        print(f"   Model: {self.model}")
        print(f"   Server: {self.base_url}")
    
    
    def _verify_connection(self):
        """
        Verify Ollama server is running and model is available.
        
        WHY THIS CHECK:
        - Fail fast if Ollama isn't running
        - Clear error messages
        - Prevents cryptic errors later
        
        RAISES:
            ConnectionError: If Ollama server is not reachable
            ValueError: If model is not available
        """
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if embedding model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            # Check if our model exists (handle version tags)
            model_available = any(
                self.model in name or name.startswith(self.model.split(':')[0])
                for name in model_names
            )
            
            if not model_available:
                print(f"\n⚠️  Warning: Model '{self.model}' not found in Ollama")
                print(f"   Available models: {', '.join(model_names)}")
                print(f"\n   To install the model, run:")
                print(f"   ollama pull {self.model}")
                
                # Don't fail here, let it fail on first embed call
                # This allows the class to be instantiated for testing
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}\n"
                f"Please ensure Ollama is running:\n"
                f"  1. Check if Ollama is installed\n"
                f"  2. Start Ollama service\n"
                f"  3. Verify with: ollama list"
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not verify Ollama connection: {e}")
    
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        USE CASE:
        When user asks a question, convert it to a vector to search
        for similar document chunks.
        
        Args:
            text: Query text to embed
        
        RETURNS:
            List of floats (embedding vector)
            Example: [0.23, -0.45, 0.67, ...] (768 numbers)
        
        EXAMPLE:
            embedder = OllamaEmbeddings()
            query_vector = embedder.embed_query("What is the remote work policy?")
            # Returns: [0.23, -0.45, 0.67, ...] (768 dimensions)
        """
        return self._embed_single(text)
    
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents/chunks.
        
        USE CASE:
        When indexing document chunks, convert all of them to vectors
        for storage in the vector database.
        
        Args:
            texts: List of text strings to embed
        
        RETURNS:
            List of embedding vectors (one per text)
        
        EXAMPLE:
            embedder = OllamaEmbeddings()
            chunks = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
            vectors = embedder.embed_documents(chunks)
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        """
        embeddings = []
        total = len(texts)
        
        print(f"\n🔢 Generating embeddings for {total} text(s)...")
        
        for idx, text in enumerate(texts):
            embedding = self._embed_single(text)
            embeddings.append(embedding)
            
            # Progress indicator
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"   Progress: {idx + 1}/{total} embeddings generated")
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings
    
    
    def _embed_single(self, text: str) -> List[float]:
        """
        Internal method to embed a single text.
        
        HOW IT WORKS:
        1. Send text to Ollama API
        2. Ollama runs embedding model
        3. Returns numerical vector
        
        Args:
            text: Text to embed
        
        RETURNS:
            Embedding vector (list of floats)
        
        RAISES:
            RuntimeError: If embedding fails
        """
        try:
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            # Send to Ollama
            response = requests.post(
                self.embed_endpoint,
                json=payload,
                timeout=300  # 300 second timeout (5 minutes)
            )
            response.raise_for_status()
            
            # Extract embedding from response
            result = response.json()
            embedding = result.get('embedding')
            
            if not embedding:
                raise ValueError(f"No embedding in response: {result}")
            
            return embedding
        
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Embedding request timed out for model '{self.model}'\n"
                f"The text might be too long or Ollama is slow."
            )
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to generate embedding: {e}\n"
                f"Model: {self.model}\n"
                f"Server: {self.base_url}\n"
                f"Make sure Ollama is running and the model is installed."
            )
    
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        """
        # Generate a test embedding
        test_embedding = self.embed_query("test")
        return len(test_embedding)


class OpenAIEmbeddings:
    """
    Generate embeddings using OpenAI's embedding models.
    """
    
    def __init__(
        self,
        model: str = OPENAI_EMBEDDING_MODEL,
        api_key: str = OPENAI_API_KEY
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAIEmbeddings")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        print(f"✨ OpenAI Embeddings initialized:")
        print(f"   Model: {self.model}")

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenAI supports batching
        texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in response.data]
    
    def get_embedding_dimension(self) -> int:
        return len(self.embed_query("test"))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embeddings():
    """
    Get an embeddings instance based on configuration.
    """
    if EMBEDDING_PROVIDER == "openai" and OPENAI_AVAILABLE:
        try:
            return OpenAIEmbeddings()
        except Exception as e:
            print(f"⚠️  Error initializing OpenAI embeddings: {e}. Falling back to Ollama.")
            return OllamaEmbeddings()
    
    return OllamaEmbeddings()


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test the embeddings module.
    
    Run: python src/embeddings.py
    """
    print("\n" + "="*60)
    print("TESTING OLLAMA EMBEDDINGS")
    print("="*60 + "\n")
    
    try:
        # Initialize embedder
        print("1. Initializing embedder...")
        embedder = OllamaEmbeddings()
        
        # Test single embedding
        print("\n2. Testing single embedding...")
        test_text = "What is semantic document analysis?"
        embedding = embedder.embed_query(test_text)
        
        print(f"   Text: '{test_text}'")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 10 values: {embedding[:10]}")
        
        # Test multiple embeddings
        print("\n3. Testing multiple embeddings...")
        test_texts = [
            "Artificial intelligence is transforming business",
            "Retrieval augmented generation explained",
            "Vector databases and semantic search"
        ]
        embeddings = embedder.embed_documents(test_texts)
        
        print(f"   Generated {len(embeddings)} embeddings")
        
        # Test similarity
        print("\n4. Testing semantic similarity...")
        text1 = "artificial intelligence"
        text2 = "machine learning"
        text3 = "vintage car restoration"
        
        emb1 = embedder.embed_query(text1)
        emb2 = embedder.embed_query(text2)
        emb3 = embedder.embed_query(text3)
        
        # Calculate cosine similarity
        import math
        
        def cosine_similarity(v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            magnitude1 = math.sqrt(sum(a * a for a in v1))
            magnitude2 = math.sqrt(sum(b * b for b in v2))
            return dot_product / (magnitude1 * magnitude2)
        
        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)
        
        print(f"\n   Similarity Results:")
        print(f"   '{text1}' vs '{text2}': {sim_1_2:.4f} (should be high)")
        print(f"   '{text1}' vs '{text3}': {sim_1_3:.4f} (should be low)")
        
        if sim_1_2 > sim_1_3:
            print(f"\n   ✅ Semantic similarity working correctly!")
        else:
            print(f"\n   ⚠️  Unexpected similarity scores")
        
        print("\n" + "="*60)
        print("✓ Embeddings test complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Ollama is running: ollama list")
        print("   2. Install embedding model: ollama pull nomic-embed-text")
        print("   3. Check Ollama is on port 11434")
