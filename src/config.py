"""
Configuration Module for AnalystSage AI Intelligence Suite

PURPOSE:
This module centralizes all configuration settings for the RAG system.
By keeping settings in one place, we can easily adjust parameters without
modifying code throughout the project.

WHAT IT DOES:
- Loads environment variables from .env file
- Provides default values if .env doesn't exist
- Makes settings accessible to all other modules
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get the root directory of the project (parent of 'src' folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Define paths to important directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # Where you'll put raw PDFs
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Where chunked text will be saved
VECTOR_STORE_DIR = PROJECT_ROOT / "vectorstore"  # Where ChromaDB stores data

# ============================================================================
# OLLAMA SETTINGS
# ============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ============================================================================
# OPENAI SETTINGS
# ============================================================================

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ============================================================================
# VECTOR STORE SETTINGS
# ============================================================================

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(VECTOR_STORE_DIR))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_intelligence")

# ============================================================================
# TEXT CHUNKING SETTINGS
# ============================================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "800.0"))

# ============================================================================
# UI SETTINGS
# ============================================================================

APP_TITLE = os.getenv("APP_TITLE", "AnalystSage AI Intelligence Studio")
APP_PORT = int(os.getenv("APP_PORT", "8501"))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created/verified:")
    print(f"  - {RAW_DATA_DIR}")
    print(f"  - {PROCESSED_DATA_DIR}")
    print(f"  - {VECTOR_STORE_DIR}")


def print_config():
    """Display current configuration settings."""
    print("\n" + "="*60)
    print("ANALYSTSAGE AI - CONFIGURATION")
    print("="*60)
    print(f"\n📁 PROJECT PATHS:")
    print(f"  Root: {PROJECT_ROOT}")
    print(f"  Raw Data: {RAW_DATA_DIR}")
    print(f"  Vector Store: {VECTOR_STORE_DIR}")
    
    print(f"\n🤖 OLLAMA SETTINGS:")
    print(f"  Base URL: {OLLAMA_BASE_URL}")
    print(f"  LLM Model: {OLLAMA_MODEL}")
    print(f"  Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
    
    print(f"\n✨ OPENAI SETTINGS:")
    print(f"  Use OpenAI for Chat: {USE_OPENAI}")
    print(f"  Embedding Provider: {EMBEDDING_PROVIDER}")
    print(f"  API Key: {'Set' if OPENAI_API_KEY else 'NOT SET'}")
    
    print(f"\n🗄️ VECTOR STORE:")
    print(f"  Collection: {COLLECTION_NAME}")
    
    print(f"\n✂️ CHUNKING:")
    print(f"  Chunk Size: {CHUNK_SIZE} chars")
    print(f"  Overlap: {CHUNK_OVERLAP} chars")
    
    print(f"\n🔍 RETRIEVAL:")
    print(f"  Top K Results: {RETRIEVAL_TOP_K}")
    print(f"  Similarity Threshold: {RETRIEVAL_SIMILARITY_THRESHOLD}")
    
    print(f"\n🌐 UI:")
    print(f"  Title: {APP_TITLE}")
    print(f"  Port: {APP_PORT}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Testing configuration module...")
    ensure_directories()
    print_config()
    print("✓ Configuration loaded successfully!")
