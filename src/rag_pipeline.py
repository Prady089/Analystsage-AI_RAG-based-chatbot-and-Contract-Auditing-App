"""
RAG Pipeline Module

PURPOSE:
Combines Retrieval and Generation to answer questions about documents.

RAG = Retrieval-Augmented Generation
- Retrieval: Find relevant document chunks
- Augmented: Add them to the prompt
- Generation: LLM generates answer based on retrieved context

WHY RAG:
Without RAG:
  User: "What is the remote work policy?"
  LLM: "I don't have access to your company's policies." ❌

With RAG:
  1. Retrieve relevant chunks from your documents
  2. Add them to the prompt as context
  3. LLM answers based on YOUR documents ✅

ARCHITECTURE:
┌─────────────────────────────────────────────────────────┐
│ User Question: "What is the remote work policy?"        │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ RETRIEVAL: Search vector store                          │
│ → Find top-k most similar chunks                        │
│ → "Remote work policy allows..."                        │
│ → "Employees may work from home..."                     │
│ → "Work from home guidelines..."                        │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ AUGMENTATION: Build prompt with context                 │
│ → System: "Answer based on these documents..."          │
│ → Context: [Retrieved chunks]                           │
│ → Question: "What is the remote work policy?"           │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ GENERATION: LLM generates answer                        │
│ → Ollama (Mistral) reads context                        │
│ → Generates accurate answer                             │
│ → Includes source citations                             │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ ANSWER: "According to the remote work policy..."        │
│ Sources: policy.pdf (page 1), handbook.pdf (page 12)    │
└─────────────────────────────────────────────────────────┘
"""

from typing import List, Dict, Optional
import requests
import json

from .vectorstore import VectorStore
from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RETRIEVAL_TOP_K,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    USE_OPENAI,
    OPENAI_API_KEY,
    OPENAI_MODEL
)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False



class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    
    WHAT IT DOES:
    1. Takes a user question
    2. Retrieves relevant document chunks
    3. Builds a prompt with context
    4. Generates an answer using LLM
    5. Returns answer with sources
    
    USAGE:
        rag = RAGPipeline()
        result = rag.query("What is the remote work policy?")
        
        print(result['answer'])
        print(result['sources'])
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        top_k: int = RETRIEVAL_TOP_K,
        similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: VectorStore instance (creates one if not provided)
            model: Ollama model for generation (e.g., "mistral:latest")
            base_url: Ollama server URL
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score (0-1)
        
        MODELS:
        - mistral:latest (recommended): Fast, accurate, 7B params
        - llama2:latest: Good alternative
        - codellama:latest: For code-related questions
        """
        self.vector_store = vector_store or VectorStore()
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # OpenAI Setup
        self.use_openai = USE_OPENAI and OPENAI_AVAILABLE and OPENAI_API_KEY != ""
        if self.use_openai:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.openai_model = OPENAI_MODEL
        
        # Verify Ollama connection if not using OpenAI
        if not self.use_openai:
            self._verify_ollama()
        
        print(f"🤖 RAG Pipeline initialized:")
        if self.use_openai:
            print(f"   LLM Model: {self.openai_model} (OpenAI)")
        else:
            print(f"   LLM Model: {self.model} (Ollama)")
        print(f"   Retrieval: top-{self.top_k} chunks")
        print(f"   Similarity threshold: {self.similarity_threshold}")
    
    
    def _verify_ollama(self):
        """
        Verify Ollama is running and model is available.
        
        RAISES:
            ConnectionError: If Ollama is not reachable
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            # Check if our model exists
            model_available = any(
                self.model in name or name.startswith(self.model.split(':')[0])
                for name in model_names
            )
            
            if not model_available:
                print(f"\n⚠️  Warning: Model '{self.model}' not found")
                print(f"   Available models: {', '.join(model_names)}")
                print(f"\n   To install: ollama pull {self.model}")
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}\n"
                f"Please ensure Ollama is running."
            )
    
    
    def query(
        self,
        question: str,
        filter_sources: Optional[List[str]] = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Answer a question using RAG.
        
        PROCESS:
        1. Retrieve relevant chunks
        2. Build prompt with context
        3. Generate answer
        4. Extract sources
        
        Args:
            question: User's question
            filter_sources: Only search these sources (e.g., ["policy.pdf"])
            include_sources: Include source citations in response
        
        RETURNS:
            Dictionary with:
            - answer: Generated answer
            - sources: List of source citations
            - retrieved_chunks: Raw chunks used (for debugging)
        
        EXAMPLE:
            rag = RAGPipeline()
            
            result = rag.query("What is the remote work policy?")
            print(result['answer'])
            # "According to the remote work policy, employees may..."
            
            print(result['sources'])
            # [{'source': 'policy.pdf', 'page': 1}, ...]
        """
        print(f"\n❓ Question: {question}")
        
        # Step 1: Retrieve relevant chunks
        print(f"   🔍 Retrieving relevant chunks...")
        
        filter_dict = None
        if filter_sources:
            # ChromaDB uses $in operator for list filtering
            filter_dict = {"source": {"$in": filter_sources}}
        
        retrieved_chunks = self.vector_store.search(
            query=question,
            top_k=self.top_k,
            filter_dict=filter_dict
        )
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer this question.",
                'sources': [],
                'retrieved_chunks': []
            }
        
        # Filter by similarity threshold
        filtered_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk['distance'] <= self.similarity_threshold
        ]
        
        if not filtered_chunks:
            return {
                'answer': "I found some documents, but they don't seem relevant enough to answer this question confidently.",
                'sources': [],
                'retrieved_chunks': retrieved_chunks
            }
        
        print(f"   ✓ Retrieved {len(filtered_chunks)} relevant chunk(s)")
        
        # Step 2: Build prompt
        prompt = self._build_prompt(question, filtered_chunks)
        
        # Step 3: Generate answer
        print(f"   🤖 Generating answer...")
        answer = self._generate(prompt)
        
        # Step 4: Extract sources
        sources = self._extract_sources(filtered_chunks) if include_sources else []
        
        print(f"   ✓ Answer generated")
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_chunks': filtered_chunks
        }


    def stream_query(
        self,
        question: str,
        filter_sources: Optional[List[str]] = None
    ):
        """
        Generator function to stream the RAG answer.
        
        Yields:
            Dictionary containing 'type' and 'content' (either chunk or final result)
        """
        # Step 1: Retrieve relevant chunks
        filter_dict = None
        if filter_sources:
            filter_dict = {"source": {"$in": filter_sources}}
        
        retrieved_chunks = self.vector_store.search(
            query=question,
            top_k=self.top_k,
            filter_dict=filter_dict
        )
        
        if not retrieved_chunks:
            yield {"type": "error", "content": "I couldn't find any relevant information."}
            return
        
        filtered_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk['distance'] <= self.similarity_threshold
        ]
        
        if not filtered_chunks:
            yield {"type": "error", "content": "No documents seem relevant enough."}
            return

        # Build prompt
        prompt = self._build_prompt(question, filtered_chunks)
        
        # Step 2: Stream generation
        yield {"type": "chunks", "content": filtered_chunks}
        
        full_answer = ""
        try:
            for text_chunk in self._generate_stream(prompt):
                full_answer += text_chunk
                yield {"type": "text", "content": text_chunk}
        
        except Exception as e:
            yield {"type": "error", "content": str(e)}
            return

        # Final metadata
        sources = self._extract_sources(filtered_chunks)
        yield {"type": "final", "content": {"answer": full_answer, "sources": sources}}


    def _generate_stream(self, prompt: str):
        """Internal generator for streaming from either Ollama or OpenAI."""
        if self.use_openai:
            yield from self._generate_openai_stream(prompt)
            return

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
            }
        }
        
        response = requests.post(
            self.generate_endpoint,
            json=payload,
            timeout=300,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if 'response' in chunk:
                    yield chunk['response']
                if chunk.get('done'):
                    break


    def _generate_openai(self, prompt: str) -> str:
        """Generate answer using OpenAI."""
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content


    def _generate_openai_stream(self, prompt: str):
        """Stream answer from OpenAI."""
        stream = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    
    
    def _build_prompt(self, question: str, chunks: List[Dict]) -> str:
        """
        Build prompt with retrieved context.
        
        PROMPT STRUCTURE:
        1. System instructions
        2. Context (retrieved chunks)
        3. Question
        4. Instructions for answer format
        
        Args:
            question: User's question
            chunks: Retrieved document chunks
        
        RETURNS:
            Formatted prompt string
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown')
            page = chunk['metadata'].get('page', 'N/A')
            content = chunk['content']
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source}, Page: {page}\n"
                f"Content: {content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Build complete prompt
        prompt = f"""You are a helpful assistant that answers questions based on provided documents.

INSTRUCTIONS:
- Answer the question using ONLY the information from the provided documents
- If the documents don't contain enough information, say so
- Be specific and cite which document you're referencing
- Keep your answer clear and concise
- If multiple documents provide relevant information, synthesize them

DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:
"""
        
        return prompt
    
    
    def _generate(self, prompt: str) -> str:
        """
        Generate answer using Ollama LLM.
        
        HOW IT WORKS:
        1. Send prompt to Ollama
        2. Stream response (or get complete response)
        3. Return generated text
        
        Args:
            prompt: Complete prompt with context
        
        RETURNS:
            Generated answer
        
        RAISES:
            RuntimeError: If generation fails
        """
        if self.use_openai:
            return self._generate_openai(prompt)
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,  # Get complete response at once
                "options": {
                    "temperature": 0.7,  # Creativity (0=deterministic, 1=creative)
                    "top_p": 0.9,        # Nucleus sampling
                    "top_k": 40,         # Top-k sampling
                }
            }
            
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=600  # 10 minute timeout for extremely complex generation
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get('response', '')
            
            if not answer:
                raise ValueError("Empty response from LLM")
            
            return answer.strip()
        
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Generation timed out. The knowledge base may be dense or the model "
                "is responding slowly on your hardware.\n\n"
                "💡 TIPS TO SPEED UP:\n"
                "1. Reduce 'Retrieval Count' in the sidebar Settings.\n"
                "2. Ensure no other heavy apps are running.\n"
                "3. Try a simpler question first."
            )
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to generate answer: {e}\n"
                f"Make sure Ollama is running and model '{self.model}' is installed."
            )
    
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract unique sources from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks with metadata
        
        RETURNS:
            List of source dictionaries with:
            - source: Filename
            - pages: List of page numbers
        
        EXAMPLE:
            [
                {'source': 'policy.pdf', 'pages': [1, 2]},
                {'source': 'handbook.pdf', 'pages': [12, 13]}
            ]
        """
        sources_dict = {}
        
        for chunk in chunks:
            source = chunk['metadata'].get('source', 'Unknown')
            page = chunk['metadata'].get('page', None)
            
            if source not in sources_dict:
                sources_dict[source] = {'source': source, 'pages': []}
            
            if page and page not in sources_dict[source]['pages']:
                sources_dict[source]['pages'].append(page)
        
        # Sort pages
        for source_info in sources_dict.values():
            source_info['pages'].sort()
        
        return list(sources_dict.values())
    
    
    def chat(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Chat with conversation history (future feature).
        
        PLANNED FEATURE:
        - Maintain conversation context
        - Follow-up questions
        - Reference previous answers
        
        Args:
            question: Current question
            conversation_history: Previous Q&A pairs
        
        RETURNS:
            Same as query()
        
        NOTE: Currently just calls query(), will be enhanced later
        """
        # For now, just call query
        # TODO: Implement conversation history
        return self.query(question)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_rag_pipeline(
    model: str = OLLAMA_MODEL,
    top_k: int = RETRIEVAL_TOP_K
) -> RAGPipeline:
    """
    Create a RAG pipeline instance (convenience function).
    
    Args:
        model: Ollama model name
        top_k: Number of chunks to retrieve
    
    RETURNS:
        RAGPipeline instance
    
    EXAMPLE:
        rag = create_rag_pipeline()
        result = rag.query("What is BABOK?")
    """
    return RAGPipeline(model=model, top_k=top_k)


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test the RAG pipeline.
    
    Run: python src/rag_pipeline.py
    """
    print("\n" + "="*60)
    print("TESTING RAG PIPELINE")
    print("="*60 + "\n")
    
    try:
        # Check if vector store has data
        from src.vectorstore import VectorStore
        vector_store = VectorStore()
        
        if vector_store.collection.count() == 0:
            print("❌ Vector store is empty!")
            print("\nPlease run the setup script first:")
            print("  python scripts/setup_vectorstore.py")
            exit(1)
        
        print(f"✓ Vector store has {vector_store.collection.count()} chunks")
        
        # Initialize RAG pipeline
        print("\n1. Initializing RAG pipeline...")
        rag = RAGPipeline(vector_store=vector_store)
        
        # Test questions
        print("\n2. Testing with sample questions...")
        
        test_questions = [
            "What is business analysis?",
            "How do I gather requirements?",
            "What are stakeholder management techniques?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"Question {i}: {question}")
            print('='*60)
            
            result = rag.query(question)
            
            print(f"\n📝 ANSWER:")
            print(result['answer'])
            
            if result['sources']:
                print(f"\n📚 SOURCES:")
                for source in result['sources']:
                    pages = ', '.join(map(str, source['pages']))
                    print(f"   - {source['source']} (pages: {pages})")
            
            print()
        
        print("\n" + "="*60)
        print("✓ RAG pipeline test complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
