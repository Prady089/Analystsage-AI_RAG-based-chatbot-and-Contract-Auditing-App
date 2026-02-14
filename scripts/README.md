# Scripts Directory

This directory contains utility scripts for the Document Q&A system.

## Available Scripts

### `setup_vectorstore.py`

**Purpose:** Index your documents into the vector store

**Usage:**
```bash
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
```

**When to run:**
- After adding new documents to `data/raw/`
- When you want to rebuild the index
- When changing chunking parameters

**What it does:**
1. Finds all documents in `data/raw/`
2. Loads them using appropriate loaders
3. Chunks them into smaller pieces
4. Generates embeddings
5. Stores in ChromaDB vector store

## Future Scripts

- `query_vectorstore.py` - Test queries from command line
- `export_data.py` - Export vector store data
- `import_data.py` - Import from other formats
