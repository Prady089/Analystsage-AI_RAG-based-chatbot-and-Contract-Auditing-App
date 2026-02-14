"""
Text File Loader

PURPOSE:
Loads plain text files (.txt, .md, etc.).

WHY TEXT SUPPORT:
- Simplest format, no parsing needed
- Many documents are plain text (notes, documentation, code)
- Markdown files are increasingly common
- Fast and reliable

NO EXTERNAL LIBRARIES NEEDED:
- Uses Python's built-in file reading
- Handles different encodings
- Simple and robust
"""

from typing import List
from pathlib import Path

from .base_loader import BaseDocumentLoader, Document


class TXTLoader(BaseDocumentLoader):
    """
    Loads plain text files.
    
    WHAT IT DOES:
    - Reads text files with proper encoding
    - Handles UTF-8, ASCII, and other encodings
    - Preserves line breaks and formatting
    
    USAGE:
        loader = TXTLoader("notes.txt")
        documents = loader.load()
        
        for doc in documents:
            print(doc.content)
    """
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        """
        Initialize text loader.
        
        Args:
            file_path: Path to text file
            encoding: File encoding (default: utf-8)
                     Common encodings: utf-8, ascii, latin-1, cp1252
        """
        super().__init__(file_path)
        
        # Validate it's a text file
        valid_extensions = ['.txt', '.md', '.markdown', '.text', '.log']
        if self.file_path.suffix.lower() not in valid_extensions:
            print(f"⚠️ Warning: {self.file_path.suffix} may not be a text file")
            print(f"   Supported: {', '.join(valid_extensions)}")
        
        self.encoding = encoding
    
    
    def load(self) -> List[Document]:
        """
        Load text file.
        
        STRATEGY:
        - Read entire file as text
        - Handle encoding errors gracefully
        - Return as single document
        
        RETURNS:
            List with one Document object
        """
        print(f"📖 Loading text file: {self.file_path.name}")
        
        try:
            # Try to read with specified encoding
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                text = f.read()
        
        except UnicodeDecodeError:
            # If encoding fails, try common alternatives
            print(f"   ⚠️ {self.encoding} encoding failed, trying alternatives...")
            
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            text = None
            
            for enc in encodings_to_try:
                try:
                    with open(self.file_path, 'r', encoding=enc) as f:
                        text = f.read()
                    print(f"   ✓ Successfully read with {enc} encoding")
                    self.encoding = enc  # Update encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise ValueError(
                    f"Could not read file with any common encoding.\n"
                    f"Tried: {', '.join(encodings_to_try)}\n"
                    f"Please specify the correct encoding."
                )
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Count lines
        num_lines = len(cleaned_text.split('\n'))
        
        # Create Document object
        document = Document(
            content=cleaned_text,
            metadata={
                "source": self.file_path.name,
                "file_path": str(self.file_path),
                "encoding": self.encoding,
                "num_lines": num_lines,
                "loader": "txt"
            }
        )
        
        print(f"✓ Extracted {len(cleaned_text)} characters ({num_lines} lines)")
        
        return [document]


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test the text loader.
    
    Run: python src/loaders/txt_loader.py
    """
    import sys
    from src.config import RAW_DATA_DIR
    
    print("\n" + "="*60)
    print("TESTING TEXT LOADER")
    print("="*60 + "\n")
    
    # Look for text files in data/raw/
    txt_files = (
        list(RAW_DATA_DIR.glob("*.txt")) + 
        list(RAW_DATA_DIR.glob("*.md")) +
        list(RAW_DATA_DIR.glob("*.markdown"))
    )
    
    if not txt_files:
        print(f"❌ No text files found in {RAW_DATA_DIR}")
        print(f"\nCreating a test file...")
        
        # Create a test file
        test_file = RAW_DATA_DIR / "test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document.\n\n")
            f.write("It has multiple paragraphs.\n\n")
            f.write("And demonstrates the text loader.\n")
        
        print(f"✓ Created: {test_file}")
        txt_files = [test_file]
    
    # Test with first text file found
    test_file = txt_files[0]
    print(f"\nTesting with: {test_file.name}\n")
    
    try:
        # Load text file
        loader = TXTLoader(str(test_file))
        documents = loader.load()
        
        # Show statistics
        print(f"\n📊 STATISTICS:")
        print(f"  Documents: {len(documents)}")
        
        if documents:
            doc = documents[0]
            print(f"  Total characters: {len(doc.content):,}")
            print(f"  Lines: {doc.metadata['num_lines']}")
            print(f"  Encoding: {doc.metadata['encoding']}")
            
            # Show sample
            print(f"\n📄 SAMPLE CONTENT:")
            print("-" * 60)
            print(doc.content[:500] if len(doc.content) > 500 else doc.content)
            if len(doc.content) > 500:
                print("...")
            print("-" * 60)
            
            print(f"\n📋 METADATA:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
        
        print("\n✓ Text loader test successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
