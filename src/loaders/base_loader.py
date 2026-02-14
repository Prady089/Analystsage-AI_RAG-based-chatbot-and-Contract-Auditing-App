"""
Base Document Loader Interface

PURPOSE:
Defines a common interface that all document loaders must implement.
This ensures consistency across different file formats (PDF, DOCX, TXT).

WHY USE AN INTERFACE:
- All loaders have the same methods (load, clean, etc.)
- Easy to add new formats (just implement this interface)
- Code that uses loaders doesn't care about file format
- Follows "Open/Closed Principle" - open for extension, closed for modification

DESIGN PATTERN: Abstract Base Class (ABC)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """
    Represents a loaded document with its content and metadata.
    
    WHY DATACLASS:
    - Automatic __init__, __repr__, __eq__ methods
    - Type hints for all fields
    - Immutable if needed (frozen=True)
    
    FIELDS:
    - content: The actual text content
    - metadata: Information about the document (source, page, etc.)
    """
    content: str
    metadata: Dict[str, any]
    
    def __repr__(self):
        """Custom representation for debugging"""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


class BaseDocumentLoader(ABC):
    """
    Abstract base class for all document loaders.
    
    WHY ABSTRACT:
    - Forces all loaders to implement required methods
    - Provides common functionality (like file validation)
    - Ensures consistent API across all loaders
    
    SUBCLASSES MUST IMPLEMENT:
    - load() method
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the loader with a file path.
        
        Args:
            file_path: Path to the document file
        
        WHAT HAPPENS:
        - Convert to Path object (better than strings)
        - Validate file exists
        - Store for later use
        """
        self.file_path = Path(file_path)
        self._validate_file()
    
    
    def _validate_file(self):
        """
        Validate that the file exists and is readable.
        
        WHY SEPARATE METHOD:
        - Called during initialization
        - Fails fast if file is missing
        - Clear error messages
        
        RAISES:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is a directory
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}\n"
                f"Please check the path and try again."
            )
        
        if self.file_path.is_dir():
            raise ValueError(
                f"Expected a file, got a directory: {self.file_path}\n"
                f"Please provide a file path, not a directory."
            )
        
        # Check if file is readable
        if not self.file_path.is_file():
            raise ValueError(f"Not a valid file: {self.file_path}")
    
    
    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load the document and return a list of Document objects.
        
        WHY ABSTRACT:
        - Each file format has different loading logic
        - Subclasses MUST implement this
        - Returns standardized Document objects
        
        RETURNS:
            List of Document objects with content and metadata
        
        EXAMPLE:
            loader = PDFLoader("report.pdf")
            documents = loader.load()
            for doc in documents:
                print(doc.content)
                print(doc.metadata)
        """
        pass
    
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text content.
        
        WHY STATIC:
        - Doesn't need instance data (self)
        - Can be used independently
        - Utility function
        
        WHAT IT DOES:
        - Remove excessive whitespace
        - Normalize line breaks
        - Remove common artifacts
        
        Args:
            text: Raw text to clean
        
        RETURNS:
            Cleaned text
        """
        import re
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove common artifacts
        text = text.replace('\x00', '')  # Null characters
        text = text.replace('\uf0b7', '•')  # Bullet points
        text = text.replace('\r\n', '\n')  # Windows line endings
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    
    def get_file_info(self) -> Dict[str, any]:
        """
        Get metadata about the file itself.
        
        WHY USEFUL:
        - Track document source
        - Debugging
        - Display in UI
        
        RETURNS:
            Dictionary with file information
        """
        return {
            "filename": self.file_path.name,
            "file_path": str(self.file_path),
            "file_size": self.file_path.stat().st_size,
            "file_extension": self.file_path.suffix,
            "file_type": self.__class__.__name__  # e.g., "PDFLoader"
        }
    
    
    def __repr__(self):
        """String representation for debugging"""
        return f"{self.__class__.__name__}(file_path='{self.file_path}')"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_file_type(file_path: str) -> str:
    """
    Detect file type based on extension.
    
    WHY THIS FUNCTION:
    - Auto-detect which loader to use
    - User doesn't need to specify format
    - Easy to add new formats
    
    Args:
        file_path: Path to file
    
    RETURNS:
        File type string: 'pdf', 'docx', 'txt', or 'unknown'
    
    EXAMPLE:
        file_type = detect_file_type("report.pdf")
        # Returns: 'pdf'
    """
    extension = Path(file_path).suffix.lower()
    
    type_mapping = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.txt': 'txt',
        '.md': 'markdown',
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.pptx': 'powerpoint'
    }
    
    return type_mapping.get(extension, 'unknown')


def get_loader_for_file(file_path: str) -> BaseDocumentLoader:
    """
    Get the appropriate loader for a file.
    
    WHY THIS FUNCTION:
    - Factory pattern - creates the right loader automatically
    - User doesn't need to know which loader to use
    - Easy to extend with new formats
    
    Args:
        file_path: Path to file
    
    RETURNS:
        Appropriate loader instance
    
    RAISES:
        ValueError: If file type is not supported
    
    EXAMPLE:
        loader = get_loader_for_file("report.pdf")
        documents = loader.load()
    """
    file_type = detect_file_type(file_path)
    
    # Import loaders here to avoid circular imports
    if file_type == 'pdf':
        from src.loaders.pdf_loader import PDFLoader
        return PDFLoader(file_path)
    
    elif file_type in ['docx', 'doc']:
        from src.loaders.docx_loader import DOCXLoader
        return DOCXLoader(file_path)
    
    elif file_type in ['txt', 'markdown']:
        from src.loaders.txt_loader import TXTLoader
        return TXTLoader(file_path)
    
    else:
        raise ValueError(
            f"Unsupported file type: {file_type}\n"
            f"Supported formats: PDF, DOCX, TXT, MD\n"
            f"File: {file_path}"
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def load_document(file_path: str) -> List[Document]:
    """
    Load any supported document in one line.
    
    WHY THIS FUNCTION:
    - Simplest API possible
    - Auto-detects file type
    - Returns standardized Document objects
    
    Args:
        file_path: Path to document file
    
    RETURNS:
        List of Document objects
    
    EXAMPLE:
        # Load any document
        documents = load_document("report.pdf")
        documents = load_document("contract.docx")
        documents = load_document("notes.txt")
        
        # Use the documents
        for doc in documents:
            print(f"Content: {doc.content[:100]}...")
            print(f"Source: {doc.metadata['source']}")
    """
    loader = get_loader_for_file(file_path)
    return loader.load()


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test the base loader interface.
    
    Run: python src/loaders/base_loader.py
    """
    print("\n" + "="*60)
    print("TESTING BASE DOCUMENT LOADER")
    print("="*60 + "\n")
    
    # Test file type detection
    test_files = [
        "report.pdf",
        "contract.docx",
        "notes.txt",
        "data.csv",
        "unknown.xyz"
    ]
    
    print("File Type Detection:")
    for file in test_files:
        file_type = detect_file_type(file)
        print(f"  {file:20} → {file_type}")
    
    print("\n✓ Base loader test complete!")
