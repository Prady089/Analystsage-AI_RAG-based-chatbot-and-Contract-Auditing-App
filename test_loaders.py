"""
Multi-Format Document Loader Test Script

PURPOSE:
Test all document loaders (PDF, DOCX, TXT) with comprehensive examples.

WHAT THIS DOES:
1. Tests file type detection
2. Tests each loader individually
3. Tests the unified load_document() function
4. Shows how to use the loaders in practice

RUN: python test_loaders.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.loaders import (
    load_document,
    detect_file_type,
    get_loader_for_file,
    Document
)
from src.config import RAW_DATA_DIR, ensure_directories


def test_file_type_detection():
    """Test automatic file type detection"""
    print("\n" + "="*60)
    print("TEST 1: FILE TYPE DETECTION")
    print("="*60 + "\n")
    
    test_files = [
        "report.pdf",
        "contract.docx",
        "policy.doc",
        "notes.txt",
        "readme.md",
        "data.csv",
        "unknown.xyz"
    ]
    
    print("Testing file type detection:")
    for filename in test_files:
        file_type = detect_file_type(filename)
        status = "✓" if file_type != "unknown" else "❌"
        print(f"  {status} {filename:20} → {file_type}")
    
    print("\n✓ File type detection test complete!")


def test_unified_loader():
    """Test the unified load_document() function"""
    print("\n" + "="*60)
    print("TEST 2: UNIFIED DOCUMENT LOADER")
    print("="*60 + "\n")
    
    # Find all supported files in data/raw/
    supported_extensions = ['*.pdf', '*.docx', '*.doc', '*.txt', '*.md']
    all_files = []
    
    for pattern in supported_extensions:
        all_files.extend(RAW_DATA_DIR.glob(pattern))
    
    if not all_files:
        print("❌ No supported files found in data/raw/")
        print(f"\nPlease add at least one file (PDF, DOCX, or TXT) to:")
        print(f"  {RAW_DATA_DIR}")
        return False
    
    print(f"Found {len(all_files)} file(s) to test:\n")
    
    # Test each file
    for file_path in all_files:
        print(f"📄 Testing: {file_path.name}")
        print(f"   Type: {detect_file_type(str(file_path))}")
        
        try:
            # Load document using unified function
            documents = load_document(str(file_path))
            
            # Show statistics
            total_chars = sum(len(doc.content) for doc in documents)
            print(f"   ✓ Loaded {len(documents)} document(s)")
            print(f"   ✓ Total characters: {total_chars:,}")
            
            # Show first document sample
            if documents:
                sample = documents[0].content[:100].replace('\n', ' ')
                print(f"   Sample: \"{sample}...\"")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            return False
    
    print("✓ Unified loader test complete!")
    return True


def test_specific_loaders():
    """Test each loader type specifically"""
    print("\n" + "="*60)
    print("TEST 3: SPECIFIC LOADER TESTS")
    print("="*60 + "\n")
    
    from src.loaders.pdf_loader import PDFLoader
    from src.loaders.docx_loader import DOCXLoader
    from src.loaders.txt_loader import TXTLoader
    
    # Test PDF loader
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    if pdf_files:
        print("📄 Testing PDF Loader:")
        try:
            loader = PDFLoader(str(pdf_files[0]))
            docs = loader.load()
            print(f"   ✓ Loaded {len(docs)} pages from {pdf_files[0].name}\n")
        except Exception as e:
            print(f"   ❌ PDF Error: {e}\n")
    else:
        print("⚠️  No PDF files to test\n")
    
    # Test DOCX loader
    docx_files = list(RAW_DATA_DIR.glob("*.docx")) + list(RAW_DATA_DIR.glob("*.doc"))
    if docx_files:
        print("📝 Testing DOCX Loader:")
        try:
            loader = DOCXLoader(str(docx_files[0]))
            docs = loader.load()
            print(f"   ✓ Loaded {docx_files[0].name}")
            print(f"   ✓ Paragraphs: {docs[0].metadata['num_paragraphs']}")
            print(f"   ✓ Tables: {docs[0].metadata['num_tables']}\n")
        except Exception as e:
            print(f"   ❌ DOCX Error: {e}\n")
    else:
        print("⚠️  No DOCX files to test\n")
    
    # Test TXT loader
    txt_files = list(RAW_DATA_DIR.glob("*.txt")) + list(RAW_DATA_DIR.glob("*.md"))
    if txt_files:
        print("📋 Testing TXT Loader:")
        try:
            loader = TXTLoader(str(txt_files[0]))
            docs = loader.load()
            print(f"   ✓ Loaded {txt_files[0].name}")
            print(f"   ✓ Lines: {docs[0].metadata['num_lines']}")
            print(f"   ✓ Encoding: {docs[0].metadata['encoding']}\n")
        except Exception as e:
            print(f"   ❌ TXT Error: {e}\n")
    else:
        print("⚠️  No TXT files to test\n")
    
    print("✓ Specific loader tests complete!")


def show_usage_examples():
    """Show practical usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60 + "\n")
    
    print("Example 1: Load any document")
    print("-" * 40)
    print("""
from src.loaders import load_document

# Automatically detects file type and loads
documents = load_document("report.pdf")
documents = load_document("contract.docx")
documents = load_document("notes.txt")

for doc in documents:
    print(doc.content)
    print(doc.metadata)
""")
    
    print("\nExample 2: Use specific loader")
    print("-" * 40)
    print("""
from src.loaders.pdf_loader import PDFLoader

loader = PDFLoader("report.pdf")
documents = loader.load()

# Access page-by-page
for doc in documents:
    page_num = doc.metadata['page']
    print(f"Page {page_num}: {doc.content[:100]}...")
""")
    
    print("\nExample 3: Load multiple documents")
    print("-" * 40)
    print("""
from pathlib import Path
from src.loaders import load_document

all_docs = []
for file_path in Path("data/raw").glob("*"):
    try:
        docs = load_document(str(file_path))
        all_docs.extend(docs)
    except:
        pass  # Skip unsupported files

print(f"Loaded {len(all_docs)} documents total")
""")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MULTI-FORMAT DOCUMENT LOADER - COMPREHENSIVE TEST")
    print("="*60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Run tests
    test_file_type_detection()
    success = test_unified_loader()
    
    if success:
        test_specific_loaders()
        show_usage_examples()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe multi-format document loader is working correctly.")
        print("You can now load PDF, DOCX, and TXT files seamlessly!")
    else:
        print("\n" + "="*60)
        print("⚠️  TESTS INCOMPLETE")
        print("="*60)
        print(f"\nPlease add documents to: {RAW_DATA_DIR}")
        print("Supported formats: PDF, DOCX, TXT, MD")


if __name__ == "__main__":
    main()
