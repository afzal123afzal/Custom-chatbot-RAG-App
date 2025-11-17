# ChatRAG - Custom Chatbot Q&A Application

A RAG (Retrieval-Augmented Generation) application built with LangChain, ChromaDB, and Google's Gemini API for document-based question answering.

## Features

- 📄 Document upload and processing (PDF support)
- 🔍 Vector-based document retrieval using ChromaDB
- 💬 Interactive chat interface with AI-powered responses
- 🧠 LangChain integration for RAG pipeline
- 🤖 Google Gemini API for natural language understanding

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   Create a `.env` file in the project root:
   ```bash
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```
   Or manually create `.env` and add:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8000`

## Usage

1. **Upload Documents:**
   - Click "Upload Document" on the main page
   - Select a PDF file
   - Wait for processing to complete

2. **Ask Questions:**
   - Type your question in the chat interface
   - The AI will search through your documents and provide answers based on the content

## Project Structure

```
ChatRAG/
├── main.py              # FastAPI application entry point
├── backend/
│   ├── __init__.py
│   ├── rag_pipeline.py  # RAG pipeline implementation
│   └── document_processor.py  # Document processing utilities
├── frontend/
│   └── index.html       # Chat interface
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this file, see Setup)
├── DOCUMENTATION.md     # Comprehensive documentation
└── README.md           # This file
```

## Technology Stack

- **Backend:** FastAPI, LangChain, ChromaDB
- **LLM:** Google Gemini API
- **Frontend:** HTML, CSS, JavaScript
- **Vector Store:** ChromaDB

## Documentation

For comprehensive documentation including architecture details, API reference, troubleshooting, and **interview preparation guide**, see [DOCUMENTATION.md](./DOCUMENTATION.md).

