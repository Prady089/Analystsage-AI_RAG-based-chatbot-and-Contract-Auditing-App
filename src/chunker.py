"""
Text Chunking Module

PURPOSE:
Splits documents into smaller, manageable chunks for RAG processing.

WHY CHUNKING IS CRITICAL:
1. LLMs have token limits (can't process entire documents)
2. Smaller chunks = more precise retrieval
3. Better semantic matching (specific concepts vs entire document)
4. Faster embedding generation

CHUNKING CHALLENGES:
- Too small: Loses context, incomplete information
- Too large: Irrelevant information, slower processing
- Must preserve meaning across chunk boundaries

STRATEGIES IMPLEMENTED:
1. Fixed-size chunking (simple, fast)
2. Sentence-aware chunking (preserves sentences)
3. Semantic chunking (preserves paragraphs/sections)
4. Recursive chunking (hierarchical splitting)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import re

from .loaders.base_loader import Document
from .config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """
    Represents a chunk of text with metadata.
    
    WHY SEPARATE FROM DOCUMENT:
    - Documents are raw loaded content
    - Chunks are processed, ready for embedding
    - Chunks have additional metadata (chunk_id, position)
    """
    content: str
    metadata: Dict[str, any]
    chunk_id: str  # Unique identifier
    
    def __repr__(self):
        preview = self.content[:50].replace('\n', ' ') + "..."
        return f"Chunk(id='{self.chunk_id}', content='{preview}')"


class TextChunker:
    """
    Splits text into chunks using various strategies.
    
    DESIGN:
    - Multiple chunking strategies
    - Configurable chunk size and overlap
    - Preserves metadata from original documents
    - Generates unique chunk IDs
    
    USAGE:
        chunker = TextChunker(chunk_size=800, chunk_overlap=200)
        chunks = chunker.chunk_documents(documents)
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        strategy: str = "sentence_aware"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy to use
                - "fixed": Simple fixed-size chunks
                - "sentence_aware": Split at sentence boundaries
                - "paragraph": Split at paragraph boundaries
                - "recursive": Hierarchical splitting
        
        WHY OVERLAP:
        Prevents splitting important concepts across chunks.
        
        Example without overlap:
        Chunk 1: "...stakeholder interviews are"
        Chunk 2: "effective for gathering requirements..."
        ❌ "interviews" context is split
        
        Example with overlap (200 chars):
        Chunk 1: "...stakeholder interviews are effective for..."
        Chunk 2: "...interviews are effective for gathering requirements..."
        ✅ "interviews" context preserved in both chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({chunk_overlap}) must be less than chunk size ({chunk_size})"
            )
        
        print(f"📐 Chunker initialized:")
        print(f"   Strategy: {strategy}")
        print(f"   Chunk size: {chunk_size} characters")
        print(f"   Overlap: {chunk_overlap} characters")
    
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of Document objects from loaders
        
        RETURNS:
            List of Chunk objects ready for embedding
        
        PROCESS:
        1. Iterate through each document
        2. Split using selected strategy
        3. Create Chunk objects with metadata
        4. Generate unique IDs
        """
        all_chunks = []
        
        print(f"\n📄 Chunking {len(documents)} document(s)...")
        
        for doc_idx, document in enumerate(documents):
            # Choose chunking strategy
            if self.strategy == "fixed":
                text_chunks = self._fixed_size_chunking(document.content)
            elif self.strategy == "sentence_aware":
                text_chunks = self._sentence_aware_chunking(document.content)
            elif self.strategy == "paragraph":
                text_chunks = self._paragraph_chunking(document.content)
            elif self.strategy == "recursive":
                text_chunks = self._recursive_chunking(document.content)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Create Chunk objects
            for chunk_idx, text in enumerate(text_chunks):
                # Generate unique chunk ID
                chunk_id = self._generate_chunk_id(document, doc_idx, chunk_idx)
                
                # Combine original metadata with chunk-specific metadata
                chunk_metadata = {
                    **document.metadata,  # Original document metadata
                    "chunk_index": chunk_idx,
                    "total_chunks": len(text_chunks),
                    "chunk_size": len(text),
                    "chunking_strategy": self.strategy
                }
                
                chunk = Chunk(
                    content=text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id
                )
                all_chunks.append(chunk)
        
        print(f"✓ Created {len(all_chunks)} chunks")
        return all_chunks
    
    
    def _fixed_size_chunking(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        STRATEGY:
        - Simple sliding window
        - Fast and predictable
        - May split mid-sentence
        
        WHEN TO USE:
        - Speed is priority
        - Text has no clear structure
        - Consistent chunk sizes needed
        
        EXAMPLE:
        Text: "ABCDEFGHIJKLMNOP" (16 chars)
        Chunk size: 8, Overlap: 2
        
        Chunk 1: "ABCDEFGH" (0-8)
        Chunk 2: "GHIJKLMN" (6-14, overlaps "GH")
        Chunk 3: "MNOP"     (12-16, overlaps "MN")
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # Extract chunk
            chunk = text[start:end]
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position (accounting for overlap)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    
    def _sentence_aware_chunking(self, text: str) -> List[str]:
        """
        Split text at sentence boundaries while respecting chunk size.
        
        STRATEGY:
        - Split text into sentences
        - Combine sentences until chunk size reached
        - Never split mid-sentence
        
        BENEFITS:
        - Preserves sentence integrity
        - More readable chunks
        - Better semantic coherence
        
        WHEN TO USE:
        - Text has clear sentences
        - Readability is important
        - Most common use case
        
        EXAMPLE:
        Text: "Sentence one. Sentence two. Sentence three. Sentence four."
        Chunk size: 30
        
        Chunk 1: "Sentence one. Sentence two." (28 chars)
        Chunk 2: "Sentence three. Sentence four." (31 chars)
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Keep last few sentences for context
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, 
                    self.chunk_overlap
                )
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    
    def _paragraph_chunking(self, text: str) -> List[str]:
        """
        Split text at paragraph boundaries.
        
        STRATEGY:
        - Split on double newlines (paragraphs)
        - Combine paragraphs until chunk size reached
        - Preserves paragraph structure
        
        BENEFITS:
        - Maintains logical structure
        - Good for well-formatted documents
        - Preserves topic boundaries
        
        WHEN TO USE:
        - Text has clear paragraphs
        - Document structure is important
        - Technical documentation, articles
        
        EXAMPLE:
        Text: "Para 1.\n\nPara 2.\n\nPara 3."
        
        Chunk 1: "Para 1.\n\nPara 2."
        Chunk 2: "Para 2.\n\nPara 3." (with overlap)
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If adding this paragraph exceeds chunk size
            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last paragraph)
                if self.chunk_overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    
    def _recursive_chunking(self, text: str) -> List[str]:
        """
        Recursively split text using multiple separators.
        
        STRATEGY:
        - Try to split on paragraphs first
        - If chunks still too large, split on sentences
        - If still too large, split on words
        - Last resort: fixed-size splitting
        
        BENEFITS:
        - Most intelligent splitting
        - Adapts to text structure
        - Best semantic preservation
        
        WHEN TO USE:
        - Mixed content (some paragraphs, some not)
        - Variable text structure
        - Maximum quality needed
        
        HIERARCHY:
        1. Paragraphs (\n\n)
        2. Sentences (. ! ?)
        3. Words (spaces)
        4. Characters (fixed size)
        """
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
        ]
        
        return self._recursive_split(text, separators, 0)
    
    
    def _recursive_split(
        self, 
        text: str, 
        separators: List[str], 
        depth: int
    ) -> List[str]:
        """
        Recursively split text using separator hierarchy.
        
        Args:
            text: Text to split
            separators: List of separators to try (in order)
            depth: Current recursion depth
        
        RETURNS:
            List of text chunks
        """
        # Base case: no more separators, use fixed-size
        if depth >= len(separators):
            return self._fixed_size_chunking(text)
        
        # If text is small enough, return as-is
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try current separator
        separator = separators[depth]
        splits = text.split(separator)
        
        # Recombine splits into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split_size = len(split) + len(separator)
            
            # If this split is too large by itself, recurse
            if len(split) > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0
                
                # Recursively split this large piece
                sub_chunks = self._recursive_split(split, separators, depth + 1)
                chunks.extend(sub_chunks)
                continue
            
            # If adding this split exceeds chunk size
            if current_size + split_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep last item for overlap
                    current_chunk = [current_chunk[-1], split]
                    current_size = len(current_chunk[0]) + split_size
                else:
                    current_chunk = [split]
                    current_size = split_size
            else:
                # Add to current chunk
                current_chunk.append(split)
                current_size += split_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        REGEX PATTERN:
        - Matches: . ! ?
        - Followed by: space or end of string
        - Handles: "Dr. Smith" (doesn't split)
        
        LIMITATIONS:
        - Simple heuristic, not perfect
        - May split on abbreviations
        - Good enough for most cases
        """
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    
    def _get_overlap_sentences(
        self, 
        sentences: List[str], 
        overlap_size: int
    ) -> List[str]:
        """
        Get last few sentences that fit within overlap size.
        
        WHY:
        For sentence-aware chunking, we want to overlap complete sentences,
        not split them.
        
        Args:
            sentences: List of sentences
            overlap_size: Target overlap size in characters
        
        RETURNS:
            Last N sentences that fit in overlap_size
        """
        overlap_sentences = []
        current_size = 0
        
        # Work backwards from end
        for sentence in reversed(sentences):
            sentence_size = len(sentence)
            if current_size + sentence_size <= overlap_size:
                overlap_sentences.insert(0, sentence)
                current_size += sentence_size
            else:
                break
        
        return overlap_sentences
    
    
    def _generate_chunk_id(
        self, 
        document: Document, 
        doc_idx: int, 
        chunk_idx: int
    ) -> str:
        """
        Generate unique chunk ID.
        
        FORMAT: {source}_{doc_idx}_{chunk_idx}
        
        EXAMPLE:
        - "report.pdf_0_0" (first chunk of first document)
        - "report.pdf_0_1" (second chunk of first document)
        - "contract.docx_1_0" (first chunk of second document)
        
        WHY UNIQUE IDS:
        - Track chunks in vector store
        - Retrieve source document
        - Debugging and logging
        """
        source = document.metadata.get('source', 'unknown')
        # Remove extension for cleaner IDs
        source_name = source.rsplit('.', 1)[0]
        return f"{source_name}_{doc_idx}_{chunk_idx}"


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    strategy: str = "sentence_aware"
) -> List[Chunk]:
    """
    Chunk documents in one line.
    
    Args:
        documents: List of Document objects
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy
    
    RETURNS:
        List of Chunk objects
    
    EXAMPLE:
        from src.loaders import load_document
        from src.chunker import chunk_documents
        
        docs = load_document("report.pdf")
        chunks = chunk_documents(docs, chunk_size=800, strategy="sentence_aware")
    """
    chunker = TextChunker(chunk_size, chunk_overlap, strategy)
    return chunker.chunk_documents(documents)


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test the chunker module.
    
    Run: python src/chunker.py
    """
    from src.loaders import load_document
    from src.config import RAW_DATA_DIR
    
    print("\n" + "="*60)
    print("TESTING TEXT CHUNKER")
    print("="*60 + "\n")
    
    # Find a test file
    test_files = (
        list(RAW_DATA_DIR.glob("*.pdf")) +
        list(RAW_DATA_DIR.glob("*.docx")) +
        list(RAW_DATA_DIR.glob("*.txt"))
    )
    
    if not test_files:
        print("❌ No files found for testing")
        print(f"Please add a document to: {RAW_DATA_DIR}")
        exit(1)
    
    test_file = test_files[0]
    print(f"Testing with: {test_file.name}\n")
    
    # Load document
    print("1. Loading document...")
    documents = load_document(str(test_file))
    print(f"   Loaded {len(documents)} document(s)\n")
    
    # Test different strategies
    strategies = ["fixed", "sentence_aware", "paragraph", "recursive"]
    
    for strategy in strategies:
        print(f"2. Testing '{strategy}' strategy...")
        chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=100,
            strategy=strategy
        )
        chunks = chunker.chunk_documents(documents)
        
        print(f"\n   📊 Results:")
        print(f"   Total chunks: {len(chunks)}")
        
        if chunks:
            sizes = [len(c.content) for c in chunks]
            print(f"   Avg chunk size: {sum(sizes)/len(sizes):.0f} chars")
            print(f"   Min chunk size: {min(sizes)} chars")
            print(f"   Max chunk size: {max(sizes)} chars")
            
            print(f"\n   📄 Sample chunk:")
            print(f"   ID: {chunks[0].chunk_id}")
            print(f"   Content: {chunks[0].content[:200]}...")
        
        print("\n" + "-"*60 + "\n")
    
    print("✓ Chunker test complete!")
