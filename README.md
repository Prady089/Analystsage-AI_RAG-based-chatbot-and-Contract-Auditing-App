# Analystsage-AI_RAG-based-chatbot-and-Contract-Auditing-App

# AnalystSage AI  
**Semantic Intelligence for Your Business Documents**

<img width="1599" height="784" alt="image" src="https://github.com/user-attachments/assets/a685af96-1b83-483e-a56d-248b908946b8" />

<img width="1918" height="862" alt="image" src="https://github.com/user-attachments/assets/e01b83a0-4065-4233-808e-7975d179385d" />

<img width="1896" height="856" alt="image" src="https://github.com/user-attachments/assets/1954260c-66a1-46e2-a104-79d124720bed" />

<img width="1748" height="630" alt="image" src="https://github.com/user-attachments/assets/95205cb5-1963-4954-99b5-d36183e9ca47" />

<img width="1777" height="700" alt="image" src="https://github.com/user-attachments/assets/34aaf7ce-541a-48fd-8526-e771999d1567" />

<img width="1737" height="779" alt="image" src="https://github.com/user-attachments/assets/e4d0e11d-2ac6-40a8-b9f8-aee5685eebfd" />

<img width="1635" height="788" alt="image" src="https://github.com/user-attachments/assets/821d3074-1d20-4679-817f-9df372dbfb4f" />

AnalystSage AI is a RAG-powered platform that makes contracts, policies, and critical documents intelligent and instantly queryable.

Upload once → ask anything → get precise, source-grounded answers.  
Compare renewals against your golden standards → spot risks and gaps in seconds.

No more endless searching, manual redlining, or LLM hallucinations about your own files.

---

## ✨ Features

### 🎯 Core Capabilities
- **Audit Contract Mode**: Upload your **Golden Agreement** and any **Renewal Draft** to receive an instant Compliance Intelligence dashboard.
- **Chat Mode**: Select documents and ask natural questions. Get accurate answers pulled directly from your files with clause numbers and document citations.
- **Multi-Format Support**: PDF, DOCX, TXT, and Markdown.
- **Semantic Search**: Find answers by meaning, not just keywords.
- **100% Local Intelligence**: Powered by Ollama—all processing happens on your machine (private & free).

### 🖥️ User Interface
- **Modern Landing Page**: Glassmorphism-inspired design with a high-performance entry point.
- **Intelligence Chat V2**: Unified responsive header with a floating navigation pill. Jump between Chat, Audit, and Library seamlessly.
- **Compliance Matrix**: Visual pass/fail indicators and risk-count metrics.
- **Integrated Studio Settings**: Centralized library management to add, search, or clear your knowledge base.

### 🔒 Privacy & Security
- **Local Processing**: No data sent to the cloud.
- **Private**: Your documents stay on your machine.
- **Free**: No API costs, no subscriptions.

---

## 🚀 Quick Start

### Prerequisites

1.  **Python 3.10+**
2.  **Ollama** (for local AI)
    - Download from: [https://ollama.ai](https://ollama.ai)
    - Install and pull required models:
      ```bash
      ollama pull mistral
      ollama pull nomic-embed-text
      ```

### Installation

1.  **Clone & Navigate**
    ```bash
    git clone https://github.com/Prady089/Analystsage-AI_RAG-based-chatbot-and-Contract-Auditing-App.git
    cd Analystsage-AI_RAG-based-chatbot-and-Contract-Auditing-App
    ```

2.  **Setup Environment**
    ```bash
    python -m venv venv_app
    .\venv_app\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run the Platform**
    **FastAPI Backend (V2 UI Gateway):**
    ```bash
    python main.py
    ```
    Then open your browser to **http://localhost:8000** for the modern V2 experience.

    **Streamlit Interface (Backup UI):**
    ```bash
    streamlit run app.py
    ```

---

## 📁 Project Structure

```
AnalystSage AI/
├── main.py                  # FastAPI Backend & Unified Gateway
├── app.py                   # Streamlit fallback interface
├── requirements.txt         # Dependencies
├── static/                  # V2 UI Files (HTML/Glassmorphism)
│   ├── analyst_sage_v2.html # Hub Page
│   ├── chat_v2.html         # Intelligent Chat
│   ├── audit_v2.html        # Smart Auditor
│   └── library_v2.html      # Document Studio
├── src/
│   ├── loaders/             # Multi-format document ingestion
│   ├── chunker.py           # Text segmentation logic
│   ├── embeddings.py        # Ollama vector generation
│   ├── vectorstore.py       # ChromaDB persistence
│   └── rag_pipeline.py      # Core RAG orchestration
└── data/raw/                # Initial document landing zone
```

---

## 🎯 Use Cases

- **Legal / Compliance**: Verify NDAs, MSAs, and regulatory updates against baselines.
- **Procurement**: Enforce vendor governance and audit renewals vs. golden agreements.
- **HR & Policies**: Instantly query employee handbooks and manuals.
- **Technical Knowledge**: Turn API documentation and user manuals into interactive support agents.

---

**Built with ❤️ by Pradeep Kumar**
**Enhanced for Semantic Intelligence.**
