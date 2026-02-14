"""
PDF Document Loader

PURPOSE:
Loads and extracts text from PDF files.
Supports multiple PDF libraries for maximum compatibility.

WHY MULTIPLE LIBRARIES:
- Different PDFs work better with different libraries
- Automatic fallback if one fails
- pdfplumber: Best for layout preservation
- pypdf: Lightweight and fast
- PyMuPDF: Most robust, handles complex PDFs
"""

from typing import List
from pathlib import Path

# Import base classes
from .base_loader import BaseDocumentLoader, Document

# PDF Processing Libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PDFLoader(BaseDocumentLoader):
    """
    Loads PDF documents and extracts text.
    
    INHERITANCE:
    - Extends BaseDocumentLoader
    - Inherits file validation, cleaning, etc.
    - Implements load() method
    
    USAGE:
        loader = PDFLoader("report.pdf")
        documents = loader.load()
        
        for doc in documents:
            print(f"Page {doc.metadata['page']}: {doc.content[:100]}...")
    """
    
    def __init__(self, file_path: str, method: str = "auto"):
        """
        Initialize PDF loader.
        
        Args:
            file_path: Path to PDF file
            method: Which library to use
                - "auto": Try pdfplumber, fallback to others
                - "pdfplumber": Use pdfplumber
                - "pypdf": Use pypdf
                - "pymupdf": Use PyMuPDF
        """
        super().__init__(file_path)  # Call parent __init__
        
        # Validate it's a PDF
        if self.file_path.suffix.lower() != '.pdf':
            raise ValueError(
                f"Expected PDF file, got: {self.file_path.suffix}\n"
                f"Use PDFLoader only for .pdf files"
            )
        
        self.method = method
    
    
    def load(self) -> List[Document]:
        """
        Load PDF and extract text from all pages.
        
        RETURNS:
            List of Document objects, one per page
        
        PROCESS:
        1. Try to load with specified method
        2. If auto mode, try libraries in order of preference
        3. Extract text page by page
        4. Clean text
        5. Create Document objects with metadata
        """
        if self.method == "auto":
            # Try libraries in order of preference
            if PDFPLUMBER_AVAILABLE:
                try:
                    return self._load_with_pdfplumber()
                except Exception as e:
                    print(f"⚠️ pdfplumber failed: {e}")
            
            if PYPDF_AVAILABLE:
                try:
                    return self._load_with_pypdf()
                except Exception as e:
                    print(f"⚠️ pypdf failed: {e}")
            
            if PYMUPDF_AVAILABLE:
                try:
                    return self._load_with_pymupdf()
                except Exception as e:
                    print(f"⚠️ PyMuPDF failed: {e}")
            
            raise RuntimeError(
                "All PDF libraries failed. Please install at least one:\n"
                "  pip install pdfplumber pypdf PyMuPDF"
            )
        
        elif self.method == "pdfplumber":
            return self._load_with_pdfplumber()
        elif self.method == "pypdf":
            return self._load_with_pypdf()
        elif self.method == "pymupdf":
            return self._load_with_pymupdf()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    
    def _load_with_pdfplumber(self) -> List[Document]:
        """
        Extract text using pdfplumber library.
        
        WHY PDFPLUMBER:
        - Best at preserving text layout
        - Good for tables and structured content
        - Handles most PDFs well
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber not installed. Install with:\n"
                "  pip install pdfplumber"
            )
        
        print(f"📖 Loading PDF with pdfplumber: {self.file_path.name}")
        documents = []
        
        with pdfplumber.open(self.file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"   Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text from this page
                text = page.extract_text()
                
                if text:  # Only include pages with text
                    # Clean the text
                    cleaned_text = self.clean_text(text)
                    
                    # Create Document object
                    doc = Document(
                        content=cleaned_text,
                        metadata={
                            "source": self.file_path.name,
                            "file_path": str(self.file_path),
                            "page": page_num,
                            "total_pages": total_pages,
                            "loader": "pdfplumber"
                        }
                    )
                    documents.append(doc)
                
                # Progress indicator every 50 pages
                if page_num % 50 == 0:
                    print(f"   Processed {page_num}/{total_pages} pages...")
        
        print(f"✓ Extracted text from {len(documents)} pages")
        return documents
    
    
    def _load_with_pypdf(self) -> List[Document]:
        """
        Extract text using pypdf library.
        
        WHY PYPDF:
        - Lightweight and fast
        - Good fallback if pdfplumber fails
        - Simple API
        """
        if not PYPDF_AVAILABLE:
            raise ImportError(
                "pypdf not installed. Install with:\n"
                "  pip install pypdf"
            )
        
        print(f"📖 Loading PDF with pypdf: {self.file_path.name}")
        documents = []
        
        reader = PdfReader(self.file_path)
        total_pages = len(reader.pages)
        print(f"   Total pages: {total_pages}")
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            
            if text:
                cleaned_text = self.clean_text(text)
                
                doc = Document(
                    content=cleaned_text,
                    metadata={
                        "source": self.file_path.name,
                        "file_path": str(self.file_path),
                        "page": page_num,
                        "total_pages": total_pages,
                        "loader": "pypdf"
                    }
                )
                documents.append(doc)
            
            if page_num % 50 == 0:
                print(f"   Processed {page_num}/{total_pages} pages...")
        
        print(f"✓ Extracted text from {len(documents)} pages")
        return documents
    
    
    def _load_with_pymupdf(self) -> List[Document]:
        """
        Extract text using PyMuPDF (fitz) library.
        
        WHY PYMUPDF:
        - Most robust, handles complex PDFs
        - Fast performance
        - Good for scanned documents (with OCR)
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF not installed. Install with:\n"
                "  pip install PyMuPDF"
            )
        
        print(f"📖 Loading PDF with PyMuPDF: {self.file_path.name}")
        documents = []
        
        doc = fitz.open(self.file_path)
        total_pages = len(doc)
        print(f"   Total pages: {total_pages}")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            
            if text:
                cleaned_text = self.clean_text(text)
                
                document = Document(
                    content=cleaned_text,
                    metadata={
                        "source": self.file_path.name,
                        "file_path": str(self.file_path),
                        "page": page_num + 1,  # 1-indexed
                        "total_pages": total_pages,
                        "loader": "pymupdf"
                    }
                )
                documents.append(document)
            
            if (page_num + 1) % 50 == 0:
                print(f"   Processed {page_num + 1}/{total_pages} pages...")
        
        doc.close()
        print(f"✓ Extracted text from {len(documents)} pages")
        return documents


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test the PDF loader.
    
    Run: python src/loaders/pdf_loader.py
    """
    import sys
    from src.config import RAW_DATA_DIR
    
    print("\n" + "="*60)
    print("TESTING PDF LOADER")
    print("="*60 + "\n")
    
    # Look for PDF files in data/raw/
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {RAW_DATA_DIR}")
        print(f"\nPlease place a PDF file there and try again.")
        sys.exit(1)
    
    # Test with first PDF found
    test_file = pdf_files[0]
    print(f"Testing with: {test_file.name}\n")
    
    try:
        # Load PDF
        loader = PDFLoader(str(test_file))
        documents = loader.load()
        
        # Show statistics
        print(f"\n📊 STATISTICS:")
        print(f"  Total pages loaded: {len(documents)}")
        
        if documents:
            total_chars = sum(len(doc.content) for doc in documents)
            avg_chars = total_chars / len(documents)
            print(f"  Total characters: {total_chars:,}")
            print(f"  Average chars/page: {avg_chars:.0f}")
            
            # Show sample from first page
            print(f"\n📄 SAMPLE FROM PAGE 1:")
            print("-" * 60)
            print(documents[0].content[:500] + "...")
            print("-" * 60)
            
            print(f"\n📋 METADATA:")
            for key, value in documents[0].metadata.items():
                print(f"  {key}: {value}")
        
        print("\n✓ PDF loader test successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
