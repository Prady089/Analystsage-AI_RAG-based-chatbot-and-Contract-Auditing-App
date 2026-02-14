# 🔮 AnalystSage AI | Intelligence Suite

A powerful, privacy-focused document intelligence suite that lets you chat with your documents using AI. Built with RAG (Retrieval-Augmented Generation) and semantic analysis, running 100% locally with Ollama.

## ✨ Features

### 🎯 Core Capabilities
- **Multi-Format Support**: PDF, DOCX, TXT, Markdown
- **Semantic Search**: Find answers by meaning, not just keywords
- **Source Citations**: Know exactly where answers come from
- **Smart Contract Auditing**: Specialized legal logic for document comparison
- **100% Local**: All processing happens on your machine (private & free)

### 🖥️ User Interface
- **AnalystStage UI**: Branded, modern, glassmorphism-inspired design
- **Three-Pillar Navigation**: Ask Questions, Smart Auditor, and Studio Settings
- **KPI Dashboards**: Visual metrics for document compliance
- **Studio Settings**: Centralized library management and retrieval configuration

### 🔒 Privacy & Security
- **Local Processing**: No data sent to cloud
- **Private**: Your documents stay on your machine
- **Free**: No API costs, no subscriptions

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**

2. **Ollama** (for local AI)
   - Download from: https://ollama.ai
   - Install and start Ollama
   - Pull required models:
     ```bash
     ollama pull mistral
     ollama pull nomic-embed-text
     ```

### Installation

1. **Navigate to project**
   ```bash
   cd "C:\Automations\BABOK RAG"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Run the web interface:**
```bash
streamlit run app.py
```

Then open your browser to **http://localhost:8501** (or the port specified in the terminal).

---

## 📁 Project Structure

```
AnalystSage AI/
├── app.py                    # Streamlit web interface
├── requirements.txt          # Dependencies
├── README.md                # This file
│
├── data/raw/                # Place raw documents here
├── vectorstore/             # Vector database (auto-created)
│
├── src/
│   ├── loaders/             # Document loaders (PDF, DOCX, TXT)
│   ├── chunker.py           # Text chunking
│   ├── embeddings.py        # Ollama embeddings
│   ├── vectorstore.py       # ChromaDB vector store
│   └── rag_pipeline.py      # RAG pipeline
│
└── venv/                    # Virtual environment
```

---

## 🎯 Use Cases

- **Contract Review**: Automated audit against standard baseline agreements.
- **HR & Policies**: Employee handbook Q&A, policy lookups.
- **Research**: Research paper Q&A, knowledge extraction.
- **Technical Docs**: API documentation, user manuals.

---

## 📚 Technologies

- **Ollama**: Local LLM inference (Mistral)
- **ChromaDB**: Vector database
- **Streamlit**: Web interface
- **Python**: Core language

---

**Built with ❤️ for privacy-focused document intelligence.**
