from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import json
import os
from pathlib import Path
import shutil
from typing import List, Optional

from src.loaders import load_document, detect_file_type
from src.chunker import chunk_documents
from src.vectorstore import VectorStore
from src.rag_pipeline import RAGPipeline
from src.config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

app = FastAPI(title="AnalystSage AI API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
vector_store = VectorStore()
rag_pipeline = RAGPipeline(vector_store=vector_store)

@app.get("/api/stats")
async def get_stats():
    return vector_store.get_stats()

@app.get("/api/sources")
async def get_sources():
    stats = vector_store.get_stats()
    return {"sources": stats["sources"]}

@app.post("/api/query")
async def query_endpoint(question: str = Form(...), filter_sources: Optional[str] = Form(None)):
    print(f"📥 Received query: {question}")
    filters = json.loads(filter_sources) if filter_sources else None
    
    def generate():
        try:
            for event in rag_pipeline.stream_query(question=question, filter_sources=filters):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            print(f"❌ Error in query stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    print(f"📤 Uploading {len(files)} files...")
    results = []
    for file in files:
        file_path = RAW_DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            docs = load_document(str(file_path))
            chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            vector_store.add_chunks(chunks)
            results.append({"filename": file.filename, "status": "success", "chunks": len(chunks)})
            print(f"✅ Indexed {file.filename}")
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "message": str(e)})
            print(f"❌ Error indexing {file.filename}: {e}")
            
    return {"results": results}

@app.delete("/api/sources/{source_name}")
async def delete_source(source_name: str):
    print(f"🗑️ Deleting source: {source_name}")
    try:
        vector_store.delete_by_source(source_name)
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Error deleting source: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_library():
    print("🔥 Clearing library...")
    vector_store.clear()
    return {"status": "success"}

@app.post("/api/audit")
async def audit_contract(
    golden_source: str = Form(...),
    audit_file: UploadFile = File(...)
):
    print(f"⚖️ Audit requested: {audit_file.filename} vs {golden_source}")
    # Save temp audit file
    temp_path = RAW_DATA_DIR / f"temp_{audit_file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audit_file.file, buffer)
        
    try:
        draft_docs = load_document(str(temp_path))
        draft_text = "\n".join([doc.content for doc in draft_docs])
        
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
        
        print("🧠 Running RAG pipeline for audit...")
        result = rag_pipeline.query(
            question=f"{audit_query}\n\nCONTRACT TEXT TO AUDIT:\n{draft_text[:5000]}...",
            filter_sources=[golden_source]
        )
        
        if temp_path.exists():
            temp_path.unlink()
            
        print("✅ Audit complete")
        return {"result": result}
    except Exception as e:
        print(f"❌ Error in audit: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (Frontend will go here)
if os.path.exists("static"):
    from fastapi.responses import FileResponse
    @app.get("/")
    async def read_index():
        return FileResponse("static/analyst_sage_v2.html")
        
    @app.get("/chat_v2.html")
    async def read_chat():
        return FileResponse("static/chat_v2.html")

    @app.get("/audit_v2.html")
    async def read_audit():
        return FileResponse("static/audit_v2.html")

    @app.get("/library_v2.html")
    async def read_library():
        return FileResponse("static/library_v2.html")
    
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
