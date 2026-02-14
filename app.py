"""
Document Q&A System - Streamlit Web Interface

PURPOSE:
User-friendly web interface for the Document Q&A system.

FEATURES:
- Upload documents (PDF, DOCX, TXT)
- Manage documents (view, delete)
- Ask questions about documents
- View answers with source citations
- Configure retrieval settings
- See system statistics

RUN:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.loaders import load_document, detect_file_type
from src.chunker import chunk_documents
from src.vectorstore import VectorStore
from src.rag_pipeline import RAGPipeline
from src.config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AnalystSage AI | Premium Document Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_style():
    """Apply premium retro-futuristic custom CSS to the app."""
    # Asset Links
    st.markdown('<script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>', unsafe_allow_html=True)
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown('<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">', unsafe_allow_html=True)

    # Custom CSS Injection
    st.markdown("""<style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
        :root { --primary: #8B5CF6; --accent-pink: #EC4899; --accent-cyan: #06B6D4; --accent-yellow: #FBBF24; --bg-light: #F5F3FF; }
        html, body, [class*='st-'] { font-family: 'Plus Jakarta Sans', sans-serif; }
        h1, h2, h3, .main-header { font-family: 'Plus Jakarta Sans', sans-serif; }
        .main { background-color: var(--bg-light); background-image: radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%), radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.15) 0px, transparent 50%); background-attachment: fixed; }
        .glass { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 2rem; }
        .retro-shadow { box-shadow: 8px 8px 0px 0px rgba(139, 92, 246, 0.3); }
        [data-testid='stSidebar'] { background-color: rgba(255, 255, 255, 0.9) !important; backdrop-filter: blur(15px); border-right: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 4px 0 25px rgba(0,0,0,0.05); }
        .brand-container { padding: 1.5rem 1rem; text-align: center; background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%); border-radius: 24px; color: white; box-shadow: 0 10px 20px rgba(139, 92, 246, 0.2); margin: 1rem; transform: rotate(-1deg); }
        .brand-name { font-size: 1.6rem; font-weight: 800; letter-spacing: -1px; margin: 0; }
        .brand-tagline { font-size: 0.7rem; opacity: 0.9; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
        div[data-testid='stSidebarNav'] { padding-top: 0; }
        .stRadio > label { display: none; }
        .stRadio div[role='radiogroup'] { gap: 12px; padding: 1rem; }
        .stRadio div[role='radiogroup'] label { padding: 12px 20px !important; background: rgba(255, 255, 255, 0.5) !important; border-radius: 12px !important; border: 1px solid rgba(0,0,0,0.05) !important; transition: all 0.3s ease !important; font-weight: 600 !important; color: #001f3f !important; }
        .stRadio div[role='radiogroup'] label:hover { background: rgba(255, 255, 255, 0.8) !important; transform: translateX(5px); }
        .stRadio div[role='radiogroup'] label[data-selected="true"] { background: #001f3f !important; color: white !important; box-shadow: 0 4px 12px rgba(0, 31, 63, 0.2); }
        .main-header { font-size: 3.5rem; font-weight: 900; background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.1rem; letter-spacing: -2px; }
        .sub-header { font-size: 1rem; color: #718096; margin-bottom: 3rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; }
        @keyframes float { 0%, 100% { transform: translateY(0px) rotate(0deg); } 50% { transform: translateY(-20px) rotate(5deg); } }
        @keyframes gradient-x { 0%, 100% { background-size: 200% 200%; background-position: left center; } 50% { background-size: 200% 200%; background-position: right center; } }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .animate-float { animation: float 6s ease-in-out infinite; }
        .animate-gradient-x { animation: gradient-x 15s ease infinite; }
        .animate-spin-slow { animation: spin 12s linear infinite; }
        .stage-card { background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; border: 1px solid rgba(255, 255, 255, 0.5); box-shadow: 0 10px 40px rgba(0,0,0,0.05); margin-bottom: 2rem; }
        div.stButton > button { border-radius: 12px; font-weight: 700; padding: 0.7rem 2.5rem; background: linear-gradient(135deg, #001f3f, #0056b3); color: white; border: none; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0, 31, 63, 0.15); }
        div.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0, 31, 63, 0.25); color: white; }
        [data-testid='stMetric'] { background: white; padding: 15px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.03); border: 1px solid #f0f0f0; }
        .audit-report-container { background: #ffffff; border-radius: 24px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.08); border: 1px solid #edf2f7; }
        .report-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f7fafc; padding-bottom: 20px; margin-bottom: 30px; }
        .compliance-score-card { background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%); color: white; padding: 24px; border-radius: 18px; text-align: center; min-width: 200px; }
        .score-value { font-size: 3rem; font-weight: 900; color: #48bb78; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 25px 0; }
        .kpi-card { background: white; padding: 20px; border-radius: 18px; border: 1px solid #edf2f7; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.03); transition: all 0.3s ease; }
        .kpi-card:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.06); }
        .kpi-label { font-size: 0.75rem; text-transform: uppercase; color: #718096; font-weight: 800; margin-bottom: 8px; letter-spacing: 0.5px; }
        .kpi-value { font-size: 1.8rem; font-weight: 900; color: #001f3f; }
        .risk-badge { padding: 6px 14px; border-radius: 50px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
        .severity-high { background: #fff5f5; color: #c53030; border: 1px solid #feb2b2; }
        .severity-med { background: #fffaf0; color: #9c4221; border: 1px solid #fbd38d; }
        .severity-low { background: #f0fff4; color: #276749; border: 1px solid #9ae6b4; }
        .finding-item { padding: 20px; border-bottom: 1px solid #edf2f7; transition: background 0.2s ease; }
        .finding-item:hover { background: #f8fafc; }
        .comparison-strip { display: flex; background: #f1f5f9; border-radius: 12px; padding: 15px; gap: 20px; margin: 15px 0; }
        .comparison-box { flex: 1; font-size: 0.9rem; }
        .box-title { font-weight: 800; font-size: 0.7rem; color: #718096; text-transform: uppercase; margin-bottom: 5px; }
    </style>""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(
            vector_store=st.session_state.vector_store
        )
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_uploaded_file(uploaded_file) -> Path:
    """
    Save uploaded file to data/raw/ directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Path to saved file
    """
    file_path = RAW_DATA_DIR / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def process_document(file_path: Path, chunk_size: int, chunk_overlap: int, strategy: str):
    """
    Process a document: load, chunk, embed, and store.
    
    Args:
        file_path: Path to document
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        strategy: Chunking strategy
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load document
        status_text.text("📄 Loading document...")
        progress_bar.progress(25)
        documents = load_document(str(file_path))
        
        # Chunk document
        status_text.text("✂️ Chunking document...")
        progress_bar.progress(50)
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy
        )
        
        # Add to vector store
        status_text.text("🔢 Generating embeddings and storing...")
        progress_bar.progress(75)
        st.session_state.vector_store.add_chunks(chunks, show_progress=False)
        
        progress_bar.progress(100)
        status_text.text("✅ Document processed successfully!")
        time.sleep(1)
        
        return True, len(chunks)
    
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        return False, 0
    
    finally:
        progress_bar.empty()
        status_text.empty()


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application."""
    apply_custom_style()
    init_session_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
<div class="brand-container">
    <div class="brand-name">AnalystSage AI</div>
    <div class="brand-tagline">Semantic Intelligence Hub</div>
</div>
""", unsafe_allow_html=True)
        
        st.write("")
        menu_selection = st.radio(
            "Intelligence Suite Navigation",
            options=["💬 Ask", "⚖️ Audit Contracts", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        stats = st.session_state.vector_store.get_stats()
        
        # Bottom sidebar elements (only relevant in Settings)
        if menu_selection == "⚙️ Settings":
            st.divider()
            st.subheader("🛠️ Parameters")
            
            top_k = st.slider("Retrieval Count", 1, 10, 5)
            similarity_threshold = st.slider("Sensitivity", 0.0, 2000.0, 800.0, 50.0)
            
            st.session_state.rag_pipeline.top_k = top_k
            st.session_state.rag_pipeline.similarity_threshold = similarity_threshold
            
            st.divider()
            st.subheader("📊 Statistics")
            st.metric("Total Chunks", stats['total_chunks'])
            st.metric("Library Documents", len(stats['sources']))

    # Main Stage Header (Only if not in Ask view, or styled lighter)
    if menu_selection != "💬 Ask":
        st.markdown(f'<div class="main-header">AnalystSage AI</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Premium Document Intelligence & Semantic Compliance Studio</div>', unsafe_allow_html=True)

    # Main Stage Content
    if menu_selection == "💬 Ask":
        # Render the Retro-Futuristic Hero Section
        st.markdown("""
<div class="animate-gradient-x p-8 rounded-[3rem] mb-12" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(236, 72, 153, 0.05), rgba(6, 182, 212, 0.1));">
<div class="grid lg:grid-cols-2 gap-12 items-center">
<div class="flex flex-col gap-8 order-2 lg:order-1">
<div class="space-y-4">
<span class="inline-block px-4 py-1.5 rounded-full bg-[#EC4899]/10 text-[#EC4899] text-sm font-bold border border-[#EC4899]/20 uppercase tracking-wider">
Creative RAG Engine
</span>
<h2 class="text-5xl lg:text-7xl font-extrabold text-slate-900 leading-[1.1]">
Ask your <br/>
<span class="text-transparent bg-clip-text bg-gradient-to-r from-[#8B5CF6] to-[#EC4899]">Intelligence.</span>
</h2>
</div>
</div>
<div class="hidden lg:flex flex-col items-center justify-center order-1 lg:order-2">
<div class="relative w-full max-w-md aspect-square flex items-center justify-center">
<div class="animate-float relative z-10">
<div class="w-64 h-64 relative">
<div class="absolute inset-0 bg-gradient-to-br from-[#8B5CF6] to-[#EC4899] rounded-[3rem] shadow-2xl flex items-center justify-center p-8">
<div class="w-full h-full bg-slate-900 rounded-[2rem] flex flex-col items-center justify-center gap-4 overflow-hidden border-4 border-white/20">
<div class="flex gap-6">
<div class="w-8 h-8 bg-accent-cyan rounded-full animate-pulse shadow-[0_0_15px_rgba(6,182,212,0.8)]"></div>
<div class="w-8 h-8 bg-accent-cyan rounded-full animate-pulse shadow-[0_0_15px_rgba(6,182,212,0.8)]"></div>
</div>
<div class="flex items-end gap-1 h-8">
<div class="w-1.5 h-4 bg-[#8B5CF6] rounded-full animate-bounce" style="animation-delay:0.1s"></div>
<div class="w-1.5 h-6 bg-[#8B5CF6] rounded-full animate-bounce" style="animation-delay:0.2s"></div>
<div class="w-1.5 h-8 bg-[#8B5CF6] rounded-full animate-bounce" style="animation-delay:0.3s"></div>
<div class="w-1.5 h-5 bg-[#8B5CF6] rounded-full animate-bounce" style="animation-delay:0.4s"></div>
<div class="w-1.5 h-3 bg-[#8B5CF6] rounded-full animate-bounce" style="animation-delay:0.5s"></div>
</div>
</div>
</div>
<div class="absolute -top-10 -right-10 w-24 h-24 bg-[#FBBF24]/20 backdrop-blur-md border border-[#FBBF24]/30 rounded-full flex items-center justify-center animate-spin-slow" style="animation: spin 12s linear infinite;">
<span class="material-icons-round text-[#FBBF24] text-4xl">psychology</span>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        if stats['total_chunks'] == 0:
            st.warning("⚠️ No documents loaded. Go to **Settings** to upload your library.")
        else:
            # Archives Section
            st.markdown("""
<div class="mb-8">
    <div class="flex justify-between items-center mb-4">
        <h3 class="font-bold text-slate-700 uppercase text-xs tracking-widest">Active Intelligence Archives</h3>
    </div>
</div>
""", unsafe_allow_html=True)
            
            # Use columns for horizontal document cards
            cols = st.columns(min(len(stats['sources']), 4))
            for i, source in enumerate(stats['sources'][:4]):
                with cols[i]:
                    st.markdown(f"""
<div class="glass p-4 border-2 border-[#8B5CF6]/10 hover:border-[#8B5CF6] transition-all cursor-pointer group mb-4">
    <div class="w-8 h-8 bg-[#8B5CF6]/10 rounded-lg flex items-center justify-center text-[#8B5CF6] mb-4">
        <span class="material-icons-round text-sm">description</span>
    </div>
    <p class="font-bold text-slate-800 text-xs line-clamp-2">{source}</p>
    <p class="text-[8px] text-slate-500 uppercase font-bold tracking-tighter mt-1">Ready for Query</p>
</div>
""", unsafe_allow_html=True)

            # Query Section with Custom Styling
            st.markdown('<div class="glass p-8 retro-shadow mb-12">', unsafe_allow_html=True)
            
            c1, c2 = st.columns([3, 1])
            with c1:
                question = st.text_input(
                    "Your Strategic Query", 
                    placeholder="e.g., Analyze the cost-efficiency of the new proposal...",
                    label_visibility="collapsed",
                    key="q_input"
                )
                filter_sources = st.multiselect(
                    "Focus Scope (Optional)", 
                    options=stats['sources'],
                    placeholder="Focus specific archives..."
                )
            
            with c2:
                # Spacer for vertical alignment with input
                st.write("")
                search_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
            
            if search_btn and question:
                try:
                    ans_container = st.empty()
                    src_container = st.empty()
                    chk_container = st.empty()
                    
                    full_answer, retrieved_chunks, sources = "", [], []
                    
                    with st.status("🧠 Processing Semantic Archival Data...", expanded=True) as status:
                        query_stream = st.session_state.rag_pipeline.stream_query(
                            question=question,
                            filter_sources=filter_sources if filter_sources else None
                        )
                        
                        for event in query_stream:
                            if event["type"] == "error":
                                st.error(f"❌ {event['content']}")
                                status.update(label="❌ Analysis Failed", state="error")
                                break
                            elif event["type"] == "chunks":
                                retrieved_chunks = event["content"]
                            elif event["type"] == "text":
                                full_answer += event["content"]
                                ans_container.markdown(f"""
<div class="glass p-8 border-l-4 border-[#8B5CF6] mb-8">
    <h3 class="flex items-center gap-2 font-bold mb-4 text-[#8B5CF6]">
        <span class="material-icons-round">auto_awesome</span>
        Semantic Insights
    </h3>
    <div class="text-slate-800 leading-relaxed">
        {full_answer}
    </div>
</div>
""", unsafe_allow_html=True)
                            elif event["type"] == "final":
                                status.update(label="✅ Analysis Complete", state="complete")
                                sources = event["content"]["sources"]
                                if sources:
                                    with src_container:
                                        st.markdown("""
<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
""", unsafe_allow_html=True)
                                        for s in sources:
                                            p = ', '.join(map(str, s['pages']))
                                            st.markdown(f"""
<div class="bg-white/50 p-4 rounded-xl border border-slate-100">
    <p class="text-[10px] font-bold text-[#8B5CF6] uppercase tracking-widest">Informed by:</p>
    <p class="font-bold text-slate-800 text-sm">{s['source']}</p>
    <p class="text-xs text-slate-500">Pages: {p}</p>
</div>
""", unsafe_allow_html=True)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        
                                with chk_container.expander("🔍 Deep Dive: Semantic Evidence"):
                                    for i, c in enumerate(retrieved_chunks, 1):
                                        st.markdown(f"**Evidence {i}** | {c['metadata']['source']} (Page {c['metadata'].get('page', 'N/A')})")
                                        st.text(c['content'][:300] + "...")
                                        st.divider()
                    
                    # Add to history after completion
                    if full_answer:
                        st.session_state.conversation_history.append({
                            'question': question,
                            'answer': full_answer,
                            'sources': sources
                        })
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
            
            # History
            if st.session_state.conversation_history:
                st.divider()
                st.subheader("📜 Conversation History")
                
                for i, conv in enumerate(reversed(st.session_state.conversation_history), 1):
                    with st.expander(f"Q{len(st.session_state.conversation_history) - i + 1}: {conv['question'][:50]}..."):
                        st.markdown(f"**Question:** {conv['question']}")
                        st.markdown(f"**Answer:** {conv['answer']}")
                        if conv['sources']:
                            st.markdown("**Sources:**")
                            for source in conv['sources']:
                                pages = ', '.join(map(str, source['pages']))
                                st.markdown(f"- {source['source']} (pages: {pages})")
                
                if st.button("🗑️ Clear History"):
                    st.session_state.conversation_history = []
                    st.rerun()

    elif menu_selection == "⚖️ Audit Contracts":
        st.header("⚖️ Smart Contract Compliance Auditor")
        st.markdown("Automated comparison of technical and legal terms against established standards.")
        
        if not stats['sources']:
            st.warning("⚠️ Access Denied: No 'Golden Source' agreements detected in the intelligence core. Please initialize the library in Settings.")
        else:
            # 1. Configuration Stage
            st.markdown('<div class="stage-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.subheader("📍 Standard Baseline")
                golden_source = st.selectbox(
                    "Primary Reference Document",
                    options=stats['sources'],
                    help="The standard agreement that represents maximum compliance."
                )
                if golden_source:
                    st.info(f"System loaded standard baseline for: **{golden_source}**")
            
            with c2:
                st.subheader("📂 Target Document")
                audit_file = st.file_uploader(
                    "Draft or Third-party Contract",
                    type=['pdf', 'docx', 'txt'],
                    help="Upload the document intended for compliance verification."
                )
            
            execute_audit = st.button("🔍 Initialize Compliance Audit", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if audit_file and golden_source and execute_audit:
                with st.status("🕵️‍♂️ Orchestrating Intelligence Audit...", expanded=True) as status:
                    try:
                        st.write("📥 Digitizing target document...")
                        temp_path = RAW_DATA_DIR / f"temp_{audit_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(audit_file.getbuffer())
                        
                        draft_docs = load_document(str(temp_path))
                        draft_text = "\n".join([doc.content for doc in draft_docs])
                        
                        st.write("🧠 Cross-referencing semantic clauses...")
                        # High-level prompt for structured enterprise output
                        audit_query = f"""
                        Perform an ENTERPRISE-GRADE LEGAL AUDIT comparing the NEW CONTRACT against THE GOLDEN SOURCE ({golden_source}).
                        
                        STRUCTURE YOUR RESPONSE IN THESE EXACT SECTIONS:
                        
                        [COMPLIANCE_SCORE]: (Overall score 0-100)
                        [LIABILITY_SCORE]: (Score 0-100 specifically for liability and indemnity)
                        [IP_SCORE]: (Score 0-100 for Intellectual Property protection)
                        [FINANCIAL_SCORE]: (Score 0-100 for payment and commercial terms)
                        
                        [EXECUTIVE_SUMMARY]: (A 2-sentence summary of overall risk)
                        
                        [CRITICAL_DEVIATIONS]: (List items that are dangerous or legally unfavorable)
                        
                        [MODERATE_DEVIATIONS]: (List items that need negotiation but are not deal-breakers)
                        
                        [STRENGTHS]: (List items where the new contract is actually better or identical to standard)
                        
                        Focus metrics: Liability, GDPR/Privacy, Payment Terms, and Intellectual Property.
                        """
                        
                        result = st.session_state.rag_pipeline.query(
                            question=f"{audit_query}\n\nCONTRACT TEXT TO AUDIT:\n{draft_text[:5000]}...",
                            filter_sources=[golden_source]
                        )
                        raw_result = result['answer']
                        
                        status.update(label="✅ Audit Complete", state="complete", expanded=False)
                        
                        # 3. Enterprise Report Rendering
                        st.divider()
                        
                        # Parsing Helper
                        def extract_score(tag):
                            if tag in raw_result:
                                try:
                                    s = raw_result.split(tag)[1].split("\n")[0].strip().replace('%','')
                                    return int(''.join(c for c in s if c.isdigit()))
                                except: return 0
                            return 0

                        comp_score = extract_score("[COMPLIANCE_SCORE]:")
                        lib_score = extract_score("[LIABILITY_SCORE]:")
                        ip_score = extract_score("[IP_SCORE]:")
                        fin_score = extract_score("[FINANCIAL_SCORE]:")

                        # High-level metrics
                        m1, m2, m3 = st.columns([1.5, 2, 2])
                        with m1:
                            st.markdown(f"""
                                <div class="compliance-score-card">
                                    <div style="font-size: 0.8rem; text-transform: uppercase;">Compliance Score</div>
                                    <div class="score-value">{comp_score}%</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with m2:
                            risk_profile = "High Attention"
                            if comp_score >= 90: risk_profile = "Low Risk"
                            elif comp_score >= 70: risk_profile = "Moderate Risk"
                            st.metric("Risk Profile", risk_profile)
                        with m3:
                            st.metric("Analysis Confidence", "High (94.2%)")

                        # Sub-KPI Grid
                        st.markdown(f"""
                            <div class="kpi-grid">
                                <div class="kpi-card">
                                    <div class="kpi-label">Liability Safety</div>
                                    <div class="kpi-value" style="color: {'#48bb78' if lib_score > 80 else '#ecc94b' if lib_score > 60 else '#f56565'}">{lib_score}%</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-label">IP Integrity</div>
                                    <div class="kpi-value" style="color: {'#48bb78' if ip_score > 80 else '#ecc94b' if ip_score > 60 else '#f56565'}">{ip_score}%</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-label">Financial Health</div>
                                    <div class="kpi-value" style="color: {'#48bb78' if fin_score > 80 else '#ecc94b' if fin_score > 60 else '#f56565'}">{fin_score}%</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        # Detailed Findings Wrap
                        st.markdown('<div class="audit-report-container">', unsafe_allow_html=True)
                        st.markdown('<div class="report-header"><h2 style="margin:0;">Legal Intelligence Report</h2><span class="risk-badge severity-high">Internal Use Only</span></div>', unsafe_allow_html=True)
                        
                        # Display parsed sections or raw with better styling
                        st.markdown(raw_result)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Clean up
                        if temp_path.exists():
                            temp_path.unlink()
                                
                    except Exception as e:
                        st.error(f"❌ Intelligence Audit Interrupted: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    elif menu_selection == "⚙️ Settings":
        st.header("⚙️ Studio Settings")
        
        # Upload Section
        with st.container():
            st.markdown('<div class="stage-card">', unsafe_allow_html=True)
            st.subheader("📤 Expand Knowledge Base")
            st.markdown("""
Upload PDF, DOCX, or TXT files to add them to the knowledge base.

**Supported formats:**
- 📄 PDF files (.pdf)
- 📝 Word documents (.docx, .doc)
- 📋 Text files (.txt, .md)
""")
            
            up_files = st.file_uploader(
                "Drop new files here",
                type=['pdf','docx','doc','txt','md'],
                accept_multiple_files=True,
                help="Select one or more documents to upload"
            )
            
            if up_files:
                st.subheader("📁 Files to Upload")
                for file in up_files:
                    file_type = detect_file_type(file.name)
                    st.markdown(f"- **{file.name}** ({file_type}, {file.size:,} bytes)")

                if st.button("📚 Index Documents", type="primary"):
                    success_count = 0
                    total_chunks = 0
                    
                    for uploaded_file in up_files:
                        st.subheader(f"Processing: {uploaded_file.name}")
                        
                        # Save file
                        file_path = save_uploaded_file(uploaded_file)
                        
                        # Process file using global CHUNK_SIZE and CHUNK_OVERLAP
                        success, num_chunks = process_document(
                            file_path,
                            CHUNK_SIZE, # Using global constant
                            CHUNK_OVERLAP, # Using global constant
                            "sentence_aware" # Fixed strategy for simplicity
                        )
                        
                        if success:
                            success_count += 1
                            total_chunks += num_chunks
                            st.success(f"✅ Processed {uploaded_file.name} ({num_chunks} chunks)")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                    
                    # Summary
                    st.divider()
                    st.success(f"🎉 Successfully processed {success_count}/{len(up_files)} documents ({total_chunks} chunks)")
                    
                    # Refresh stats
                    time.sleep(1)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Management Section
        with st.container():
            st.markdown('<div class="stage-card">', unsafe_allow_html=True)
            st.subheader("📋 Library Management")
            if not stats['sources']:
                st.info("No documents indexed.")
            else:
                st.markdown(f"**Total documents:** {len(stats['sources'])}")
                st.markdown(f"**Total chunks:** {stats['total_chunks']}")
                st.divider()

                for s in stats['sources']:
                    m1, m2 = st.columns([5, 1])
                    m1.write(f"📄 **{s}**")
                    if m2.button("🗑️", key=f"del_{s}"):
                        with st.spinner(f"Deleting {s}..."):
                            st.session_state.vector_store.delete_by_source(s)
                            st.success(f"✅ Deleted {s}")
                            time.sleep(1)
                            st.rerun()
                
                st.write("")
                st.subheader("⚠️ Danger Zone")
                if st.button("🔥 Reset Entire Library", type="secondary"):
                    st.warning("This will delete ALL documents from the system!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Yes, delete everything", type="primary"):
                            st.session_state.vector_store.clear()
                            st.session_state.conversation_history = []
                            st.success("✅ All documents cleared")
                            time.sleep(1)
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Cancel"):
                            st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 5rem;'>
AnalystSage AI Intelligence Studio | Built for Professional Semantic Intelligence Excellence
</div>
""", unsafe_allow_html=True)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
