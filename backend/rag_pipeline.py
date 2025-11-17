"""
RAG Pipeline implementation using LangChain, ChromaDB, and Gemini.
"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()


class RAGPipeline:
    """RAG Pipeline for document Q&A using LangChain, ChromaDB, and Gemini."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the RAG pipeline."""
        self.persist_directory = persist_directory
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Gemini LLM (use supported model; configurable via env)
        llm_model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=self.api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Initialize embeddings
        embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
        if embeddings_provider == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
        else:
            # Default to local embeddings to avoid API quotas
            # Model downloads on first run and caches locally
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.qa_chain = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text documents
            metadatas: Optional list of metadata dictionaries
        """
        import traceback
        
        try:
            print(f"Creating documents from {len(texts)} texts...")
            # Create documents
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                documents.append(Document(page_content=text, metadata=metadata))
            
            print(f"Created {len(documents)} documents, splitting into chunks...")
            # Split documents into chunks
            doc_chunks = self.text_splitter.split_documents(documents)
            print(f"Split into {len(doc_chunks)} chunks")
            
            # Create or get vector store
            if self.vectorstore is None:
                print("Creating new vector store...")
                self.vectorstore = Chroma.from_documents(
                    documents=doc_chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                print("Vector store created successfully")
            else:
                print("Adding documents to existing vector store...")
                # Add to existing vector store
                self.vectorstore.add_documents(doc_chunks)
                print("Documents added to vector store")
            
            print("Persisting vector store...")
            # Persist the vector store
            self.vectorstore.persist()
            
            print("Creating QA chain...")
            # Create QA chain
            self._create_qa_chain()
            print("QA chain created successfully")
            
        except Exception as e:
            error_trace = traceback.format_exc()
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
            print(f"ERROR in add_documents:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {error_msg}")
            print(f"Full traceback:\n{error_trace}")
            raise Exception(f"Failed to add documents: {error_msg}")
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        if self.vectorstore is None:
            return
        
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer concise and focused.

        Context: {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def load_existing_vectorstore(self):
        """Load existing vector store if it exists."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                self._create_qa_chain()
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return False
        return False
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            return {
                "answer": "No documents have been uploaded yet. Please upload documents first.",
                "sources": []
            }
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    
    def is_ready(self) -> bool:
        """Check if the RAG pipeline is ready to answer questions."""
        return self.qa_chain is not None

