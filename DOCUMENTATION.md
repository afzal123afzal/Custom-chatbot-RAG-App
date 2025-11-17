# ChatRAG - Complete Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Application Architecture](#application-architecture)
3. [Features](#features)
4. [Technology Stack](#technology-stack)
5. [System Components](#system-components)
6. [How It Works](#how-it-works)
7. [Setup & Installation](#setup--installation)
8. [Usage Guide](#usage-guide)
9. [API Endpoints](#api-endpoints)
10. [Configuration](#configuration)
11. [Project Structure](#project-structure)
12. [Interview Preparation](#interview-preparation)

---

## 📖 Overview

**ChatRAG** is a sophisticated Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and interact with them through an intelligent question-answering interface. The application combines document processing, vector storage, and large language models to provide accurate, context-aware answers based on uploaded documents.

### Key Capabilities
- **Document Upload**: Supports PDF file uploads with text extraction
- **Vector Storage**: Uses ChromaDB for efficient document embedding storage
- **Semantic Search**: Retrieves relevant document chunks based on query similarity
- **AI-Powered Answers**: Generates context-aware responses using Google's Gemini API
- **Interactive UI**: Modern, responsive web interface for document interaction

---

## 🏗️ Application Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Frontend      │  (HTML/CSS/JavaScript)
│   (Browser)     │
└────────┬────────┘
         │ HTTP/REST API
         │
┌────────▼────────────────────────────────────────┐
│           FastAPI Backend                        │
│  ┌──────────────────────────────────────────┐   │
│  │  Document Upload Endpoint                │   │
│  │  - Receives PDF files                    │   │
│  │  - Extracts text using PyPDF2            │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  Query Endpoint                          │   │
│  │  - Receives user questions               │   │
│  │  - Returns AI-generated answers          │   │
│  └──────────────────────────────────────────┘   │
└────────┬─────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────┐
│          RAG Pipeline                           │
│  ┌──────────────────────────────────────────┐   │
│  │  Document Processing                     │   │
│  │  - Text splitting                        │   │
│  │  - Chunk creation                        │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  Embedding Generation                    │   │
│  │  - Sentence Transformers (local)         │   │
│  │  - Or Gemini Embeddings (optional)       │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  Vector Store (ChromaDB)                 │   │
│  │  - Stores document embeddings            │   │
│  │  - Enables similarity search             │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  LLM (Google Gemini)                     │   │
│  │  - Generates answers from context        │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Upload Flow**:
   ```
   User uploads PDF → FastAPI receives file → PyPDF2 extracts text 
   → Text split into chunks → Generate embeddings → Store in ChromaDB
   ```

2. **Query Flow**:
   ```
   User asks question → Generate query embedding → Similarity search in ChromaDB 
   → Retrieve top-k relevant chunks → Pass to LLM with context → Return answer
   ```

---

## ✨ Features

### 1. Document Management
- **PDF Upload**: Drag-and-drop or click-to-upload interface
- **Text Extraction**: Automatic text extraction from PDF pages
- **Chunk Processing**: Intelligent text chunking with overlap for better context retention
- **Persistent Storage**: Vector embeddings are stored persistently in ChromaDB

### 2. Intelligent Question Answering
- **Semantic Search**: Finds relevant document sections using vector similarity
- **Context-Aware Answers**: LLM generates answers based on retrieved document chunks
- **Source Citations**: Displays source documents used for each answer
- **Error Handling**: Graceful error handling with informative messages

### 3. User Interface
- **Modern Design**: Clean, ChatGPT-inspired interface
- **Responsive Layout**: Works on desktop and mobile devices
- **Real-time Feedback**: Upload progress indicators and status messages
- **Chat Interface**: Interactive chat experience with message history

### 4. Technical Features
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Checks**: API health check endpoint
- **Environment Configuration**: Secure API key management via `.env`
- **Error Recovery**: Robust error handling throughout the application

---

## 🔧 Technology Stack

### Backend
- **FastAPI** (v0.104.1): Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI application
- **Python 3.12**: Programming language

### Machine Learning & NLP
- **LangChain** (v0.1.0): Framework for building LLM applications
- **langchain-google-genai** (v0.0.6): LangChain integration for Google Gemini
- **langchain-community** (v0.0.10): Community integrations for LangChain
- **sentence-transformers** (v2.2.2): Local embedding models
- **google-generativeai** (v0.3.2): Google Gemini API client

### Vector Database
- **ChromaDB** (v0.4.22): Open-source embedding database

### Document Processing
- **PyPDF2** (v3.0.1): PDF text extraction library

### Other Dependencies
- **python-dotenv** (v1.0.0): Environment variable management
- **pydantic** (v2.5.0): Data validation using Python type annotations
- **python-multipart**: File upload support for FastAPI

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with modern features
- **JavaScript (Vanilla)**: Client-side interactivity

---

## 🧩 System Components

### 1. Main Application (`main.py`)
- **FastAPI App Instance**: Main application entry point
- **CORS Middleware**: Enables cross-origin requests
- **Global RAG Pipeline**: Initialized at startup
- **Static File Serving**: Serves frontend HTML/CSS/JS
- **API Routes**:
  - `GET /`: Serves the main HTML page
  - `POST /api/upload`: Handles document uploads
  - `POST /api/query`: Processes user questions
  - `GET /api/health`: Health check endpoint

### 2. RAG Pipeline (`backend/rag_pipeline.py`)
**Class: `RAGPipeline`**

**Responsibilities**:
- Initialize LLM (Google Gemini)
- Initialize embeddings (local or Gemini)
- Manage ChromaDB vector store
- Create QA chains using LangChain
- Handle document addition and querying

**Key Methods**:
- `__init__()`: Initialize pipeline with configuration
- `add_documents()`: Add documents to vector store
- `query()`: Query the RAG system
- `load_existing_vectorstore()`: Load persisted vector store
- `_create_qa_chain()`: Create retrieval QA chain
- `is_ready()`: Check if pipeline is ready

**Configuration Options**:
- `LLM_MODEL`: Gemini model to use (default: "gemini-2.0-flash")
- `EMBEDDINGS_PROVIDER`: "local" or "gemini" (default: "local")
- `persist_directory`: ChromaDB storage location (default: "./chroma_db")

### 3. Document Processor (`backend/document_processor.py`)
**Functions**:
- `extract_text_from_pdf()`: Extract text from PDF files page by page
- `split_text_into_chunks()`: Split text into smaller chunks (currently not used, splitting handled by LangChain)

### 4. Frontend (`frontend/index.html`)
**Features**:
- Single-page application with inline CSS and JavaScript
- File upload area with drag-and-drop support
- Chat interface for questions and answers
- Status message system for user feedback
- Mobile-responsive design with hamburger menu

**JavaScript Functions**:
- `handleFileUpload()`: Handles PDF file uploads
- `sendMessage()`: Sends questions to backend
- `addMessage()`: Displays messages in chat interface
- `showStatus()`: Shows status notifications

---

## 🔄 How It Works

### Document Processing Pipeline

1. **Upload**: User uploads a PDF file through the web interface
2. **Text Extraction**: PyPDF2 extracts text from each PDF page
3. **Text Splitting**: LangChain's `RecursiveCharacterTextSplitter` splits text into chunks:
   - Chunk size: 1000 characters
   - Chunk overlap: 200 characters (prevents context loss)
4. **Embedding Generation**: Each chunk is converted to a vector embedding:
   - Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
   - Alternative: Gemini embeddings (API-based)
5. **Vector Storage**: Embeddings are stored in ChromaDB with metadata (filename, source)
6. **Persistence**: Vector store is saved to disk for future use

### Query Processing Pipeline

1. **Question Input**: User submits a question through the chat interface
2. **Query Embedding**: Question is converted to an embedding using the same model
3. **Similarity Search**: ChromaDB performs vector similarity search:
   - Retrieves top-k most similar document chunks (k=3)
   - Uses cosine similarity by default
4. **Context Assembly**: Retrieved chunks are assembled as context
5. **Prompt Construction**: Custom prompt template combines:
   - Retrieved document chunks (context)
   - User question
   - Instructions for the LLM
6. **LLM Generation**: Google Gemini generates an answer based on:
   - The provided context
   - The user's question
   - Custom prompt instructions
7. **Response**: Answer and source documents are returned to the user

### RAG Chain Configuration

The application uses LangChain's `RetrievalQA` chain with:
- **Chain Type**: "stuff" (concatenates all retrieved documents)
- **Retriever**: Similarity search retriever (k=3)
- **Return Source Documents**: Enabled for citation display
- **Custom Prompt**: Instructs LLM to use context, be concise, and admit uncertainty

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))

### Step-by-Step Setup

1. **Clone/Navigate to Project**:
   ```bash
   cd /home/seq_afzal/Documents/LLm_Project/Month_3/ChatRAG
   ```

2. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Create Environment File**:
   ```bash
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```
   Or manually create `.env` file:
   ```
   GEMINI_API_KEY=your_actual_api_key
   LLM_MODEL=gemini-2.0-flash  # Optional
   EMBEDDINGS_PROVIDER=local  # Optional: "local" or "gemini"
   ```

5. **Run the Application**:
   ```bash
   python main.py
   ```

6. **Access the Application**:
   Open your browser and navigate to: `http://localhost:8000`

### First-Time Setup Notes
- The sentence-transformers model will download automatically on first use (~90MB)
- ChromaDB will create its database files in `./chroma_db/` directory
- Uploaded files are temporarily stored in `./uploads/` and cleaned up after processing

---

## 📝 Usage Guide

### Uploading Documents

1. **Via Click**:
   - Click the upload area in the sidebar
   - Select a PDF file from your computer
   - Wait for "Document processed successfully" message

2. **Via Drag-and-Drop**:
   - Drag a PDF file over the upload area
   - Drop the file
   - Wait for processing confirmation

### Asking Questions

1. After uploading a document, the chat interface becomes active
2. Type your question in the input field
3. Press Enter or click the send button
4. The AI will:
   - Search through your documents
   - Find relevant sections
   - Generate an answer
   - Display source citations

### Example Questions
- "What is the main topic of this document?"
- "Summarize the key points"
- "What does the document say about [specific topic]?"
- "List all important dates mentioned"

### Best Practices
- Upload documents before asking questions
- Ask specific questions for better answers
- Check source citations to verify information
- You can upload multiple documents for broader knowledge base

---

## 🌐 API Endpoints

### 1. `GET /`
- **Description**: Serves the main HTML page
- **Response**: HTML content
- **Authentication**: None required

### 2. `POST /api/upload`
- **Description**: Upload and process a PDF document
- **Request**:
  - Method: POST
  - Content-Type: multipart/form-data
  - Body: PDF file
- **Response**:
  ```json
  {
    "message": "Document 'filename.pdf' processed successfully",
    "chunks": 15
  }
  ```
- **Error Responses**:
  - `400`: Invalid file type or no text extracted
  - `500`: Processing error or RAG pipeline not initialized

### 3. `POST /api/query`
- **Description**: Query the RAG system with a question
- **Request**:
  - Method: POST
  - Content-Type: application/json
  - Body:
    ```json
    {
      "question": "What is the main topic?"
    }
    ```
- **Response**:
  ```json
  {
    "answer": "The main topic is...",
    "sources": [
      {
        "content": "Relevant text excerpt...",
        "metadata": {
          "filename": "document.pdf",
          "source": "./uploads/document.pdf"
        }
      }
    ]
  }
  ```
- **Error Responses**:
  - `400`: No documents uploaded
  - `500`: Query processing error or RAG pipeline not initialized

### 4. `GET /api/health`
- **Description**: Check application health status
- **Response**:
  ```json
  {
    "status": "healthy",
    "ready": true,
    "pipeline_initialized": true
  }
  ```
- **Use Case**: Monitoring and debugging

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | None | Yes |
| `LLM_MODEL` | Gemini model to use | `gemini-2.0-flash` | No |
| `EMBEDDINGS_PROVIDER` | Embedding provider: `local` or `gemini` | `local` | No |

### Configuration in Code

**RAG Pipeline Settings** (`backend/rag_pipeline.py`):
- `chunk_size`: 1000 characters
- `chunk_overlap`: 200 characters
- `temperature`: 0.7 (LLM creativity)
- `k`: 3 (number of document chunks to retrieve)

**Text Splitter** (`RecursiveCharacterTextSplitter`):
- Recursively splits text by characters
- Preserves paragraph and sentence boundaries when possible
- Overlap ensures context continuity

---

## 📁 Project Structure

```
ChatRAG/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # Basic setup guide
├── DOCUMENTATION.md                 # This comprehensive documentation
├── .env                             # Environment variables (create this)
├── backend/
│   ├── __init__.py                  # Package initialization
│   ├── rag_pipeline.py             # RAG pipeline implementation
│   └── document_processor.py       # PDF text extraction utilities
├── frontend/
│   └── index.html                   # Single-page application UI
├── chroma_db/                       # ChromaDB persistent storage
│   ├── chroma.sqlite3              # Database file
│   └── [collection_id]/            # Vector collection data
├── uploads/                         # Temporary file upload storage
└── venv/                            # Python virtual environment
```

---

## 🎯 Interview Preparation

### Core Concepts to Understand

#### 1. RAG (Retrieval-Augmented Generation)
**What to know:**
- **Definition**: RAG combines information retrieval with generative AI
- **Why RAG?**: Allows LLMs to access external knowledge without retraining
- **Components**: Document processing, embedding, vector store, retrieval, generation
- **Advantages**: 
  - Reduces hallucinations
  - Enables domain-specific knowledge
  - Provides citations
  - Keeps model knowledge up-to-date

**Expected Questions:**
- "What is RAG and why use it?"
- "How does RAG differ from fine-tuning?"
- "What are the limitations of RAG?"

#### 2. Vector Embeddings & Similarity Search
**What to know:**
- **Embeddings**: Dense vector representations of text (384-dim for this app)
- **Similarity Metrics**: Cosine similarity, Euclidean distance
- **Why embeddings?**: Capture semantic meaning, enable similarity search
- **Embedding Models**: 
  - Local: `sentence-transformers/all-MiniLM-L6-v2` (fast, free)
  - API-based: Gemini embeddings (higher quality, costs API calls)

**Expected Questions:**
- "What are embeddings and why are they important?"
- "How does similarity search work?"
- "What's the difference between local and API-based embeddings?"
- "How would you choose chunk size and overlap?"

#### 3. LangChain Framework
**What to know:**
- **Purpose**: Framework for building LLM applications
- **Components Used**:
  - `RetrievalQA`: Chain that combines retrieval and generation
  - `RecursiveCharacterTextSplitter`: Text chunking utility
  - `Chroma`: Vector store integration
  - `PromptTemplate`: Custom prompt construction
- **Chain Types**: "stuff", "map_reduce", "refine", "map_rerank"

**Expected Questions:**
- "Why use LangChain?"
- "Explain the RetrievalQA chain"
- "What are different chain types and when to use each?"

#### 4. ChromaDB
**What to know:**
- **Purpose**: Vector database for storing embeddings
- **Features**: Persistent storage, similarity search, metadata filtering
- **Advantages**: Lightweight, easy to use, good for small-medium datasets
- **Alternatives**: Pinecone, Weaviate, Qdrant, Milvus

**Expected Questions:**
- "Why ChromaDB over other vector databases?"
- "How does ChromaDB store and query vectors?"
- "What are the limitations of ChromaDB?"

#### 5. Document Processing
**What to know:**
- **Text Extraction**: PyPDF2 for PDF processing
- **Text Chunking**: 
  - Why chunk? LLMs have token limits, improves retrieval precision
  - Chunk size: Balance between context and precision
  - Overlap: Preserves context across chunk boundaries
- **Challenges**: PDF formatting, tables, images (not handled currently)

**Expected Questions:**
- "How do you handle different document formats?"
- "What's the optimal chunk size?"
- "How does chunk overlap help?"

#### 6. FastAPI & Web Development
**What to know:**
- **FastAPI**: Modern Python web framework
- **Features**: Automatic API documentation, async support, type validation
- **Endpoints**: RESTful API design
- **File Upload**: Multipart form data handling
- **CORS**: Cross-origin resource sharing for frontend-backend communication

**Expected Questions:**
- "Why FastAPI over Flask/Django?"
- "How do you handle file uploads in FastAPI?"
- "Explain CORS and why it's needed"

### Technical Deep Dives

#### Architecture & Design Decisions

**Q: Why separate document processing from RAG pipeline?**
- **A**: Separation of concerns, easier testing, modularity, allows reuse

**Q: Why persist the vector store?**
- **A**: Avoid reprocessing documents, faster startup, maintains state across sessions

**Q: How do you handle multiple document uploads?**
- **A**: Documents are added incrementally to existing vector store, embeddings are appended

**Q: What are potential bottlenecks?**
- **A**: 
  - Large PDF processing (time)
  - Embedding generation (local is fast, API has rate limits)
  - LLM API calls (cost and latency)
  - Vector search with large collections (scales with document count)

#### Optimization Strategies

**Q: How would you improve performance?**
- **A**:
  - Async processing for document uploads
  - Caching frequently asked questions
  - Batch embedding generation
  - Using GPU for local embeddings
  - Connection pooling for database

**Q: How would you scale this application?**
- **A**:
  - Replace ChromaDB with distributed vector DB (Pinecone, Weaviate)
  - Use load balancer for multiple FastAPI instances
  - Implement Redis for caching
  - Queue system for document processing (Celery, RQ)
  - Microservices architecture

**Q: How would you improve answer quality?**
- **A**:
  - Reranking retrieved documents
  - Using larger embedding models
  - Prompt engineering
  - Fine-tuning LLM on domain data
  - Multi-query retrieval (reformulate question multiple ways)

#### Error Handling & Robustness

**Q: What error cases did you handle?**
- **A**:
  - Missing API key
  - Invalid file types
  - PDF extraction failures
  - Empty documents
  - Query failures
  - Network errors

**Q: How do you validate user inputs?**
- **A**: Pydantic models for request validation, file type checking, error responses

### System Design Questions

**Q: Design a system to handle 1 million documents**
- **A**: Discuss:
  - Distributed vector database
  - Sharding strategies
  - Caching layer
  - Async processing pipeline
  - Load balancing
  - Horizontal scaling

**Q: How would you add multi-user support?**
- **A**: 
  - User authentication
  - Separate vector collections per user
  - Access control
  - Document ownership
  - Shared documents feature

**Q: How would you implement document versioning?**
- **A**:
  - Metadata tracking (version, timestamp)
  - Soft deletion
  - Update existing embeddings
  - Version comparison

### Code-Specific Questions

**Be prepared to explain:**
- How the RAG pipeline is initialized
- How documents are added to the vector store
- How queries are processed end-to-end
- The prompt template and its purpose
- Why certain parameters were chosen (chunk_size=1000, k=3, etc.)

### Demonstration Preparation

**Be ready to:**
- Run the application and demonstrate it
- Upload a sample PDF
- Ask questions and show answers
- Explain the source citations
- Show the vector store persistence
- Discuss any limitations you've identified

### Common Questions & Answers

**Q: What are the limitations of this application?**
- **A**: 
  - Only supports PDFs (no Word, images, etc.)
  - No user authentication
  - Single-tenant (no multi-user support)
  - ChromaDB may not scale to very large datasets
  - No conversation memory (each query is independent)
  - No handling of PDFs with images/tables

**Q: What would you improve next?**
- **A**: 
  - Support more file formats (Word, TXT, Markdown)
  - Add conversation history/memory
  - Implement document management (delete, list uploaded docs)
  - Add user authentication
  - Improve UI/UX (better source display, markdown rendering)
  - Add streaming responses
  - Implement reranking for better retrieval

**Q: How do you handle edge cases?**
- **A**: 
  - Empty PDFs: Check if text extraction returns content
  - Very large PDFs: Could add chunking limits or file size limits
  - Questions outside document scope: LLM admits uncertainty
  - Malformed PDFs: Error handling and user feedback

### Practice Scenarios

1. **Code Walkthrough**: Walk through the entire codebase explaining each component
2. **Live Coding**: Be ready to modify code (e.g., change chunk size, add a new endpoint)
3. **Debugging**: Know how to troubleshoot common issues (API key, ChromaDB errors)
4. **Architecture Discussion**: Explain the architecture and justify design choices

---

## 🔍 Troubleshooting

### Common Issues

1. **"RAG pipeline not initialized"**
   - Check `.env` file exists and contains `GEMINI_API_KEY`
   - Verify API key is valid

2. **"No text could be extracted from PDF"**
   - PDF might be image-based (scanned)
   - Try OCR or different PDF
   - Check if PDF is corrupted

3. **Embedding model download fails**
   - Check internet connection
   - Verify HuggingFace access

4. **ChromaDB errors**
   - Check write permissions in `./chroma_db/`
   - Try deleting `chroma_db/` folder and restarting

5. **Port 8000 already in use**
   - Change port in `main.py`: `uvicorn.run(app, host="0.0.0.0", port=8001)`

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://www.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Gemini API](https://ai.google.dev/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

---

## 📄 License & Credits

This is a demonstration project for educational purposes, showcasing RAG implementation using modern Python frameworks and LLM APIs.

---

**Last Updated**: 2024
**Version**: 1.0.0




