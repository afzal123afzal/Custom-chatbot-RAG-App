"""
Main FastAPI application for the ChatRAG service.
"""
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from dotenv import load_dotenv

from backend.rag_pipeline import RAGPipeline
from backend.document_processor import extract_text_from_pdf

load_dotenv()

app = FastAPI(title="ChatRAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
try:
    rag_pipeline = RAGPipeline()
    # Load existing vector store if available
    rag_pipeline.load_existing_vectorstore()
except Exception as e:
    import traceback
    print(f"Error initializing RAG pipeline: {e}\n{traceback.format_exc()}")
    # Create a dummy pipeline that will fail gracefully
    rag_pipeline = None

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    Currently supports PDF files.
    """
    import traceback
    
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create uploads directory if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    try:
        print(f"Starting upload for file: {file.filename}")
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File saved to: {file_path}")
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        text_chunks = extract_text_from_pdf(file_path)
        print(f"Extracted {len(text_chunks)} text chunks")
        
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Combine all chunks
        full_text = "\n\n".join(text_chunks)
        print(f"Combined text length: {len(full_text)} characters")
        
        # Add to RAG pipeline
        if rag_pipeline is None:
            raise HTTPException(status_code=500, detail="RAG pipeline not initialized. Please check your GEMINI_API_KEY in .env file.")
        
        print("Adding documents to RAG pipeline...")
        try:
            rag_pipeline.add_documents(
                texts=[full_text],
                metadatas=[{"filename": file.filename, "source": file_path}]
            )
            print("Documents added successfully!")
        except Exception as e:
            error_trace = traceback.format_exc()
            error_msg = str(e) if str(e) else type(e).__name__
            print(f"ERROR adding documents to RAG pipeline:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {error_msg}")
            print(f"Full traceback:\n{error_trace}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error adding documents to RAG pipeline: {error_msg if error_msg else type(e).__name__}"
            )
        
        return JSONResponse(content={
            "message": f"Document '{file.filename}' processed successfully",
            "chunks": len(text_chunks)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = str(e) if str(e) else type(e).__name__
        print(f"ERROR processing document:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {error_msg}")
        print(f"Full traceback:\n{error_trace}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing document: {error_msg if error_msg else type(e).__name__}"
        )
    
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up file: {cleanup_error}")


@app.post("/api/query", response_model=QuestionResponse)
async def query(question_request: QuestionRequest):
    """
    Query the RAG system with a question.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline not initialized. Please check your GEMINI_API_KEY in .env file."
        )
    
    if not rag_pipeline.is_ready():
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded. Please upload documents first."
        )
    
    try:
        result = rag_pipeline.query(question_request.question)
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "ready": rag_pipeline.is_ready() if rag_pipeline else False,
        "pipeline_initialized": rag_pipeline is not None
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

