"""
Document processing utilities for extracting text from various file formats.
"""
import os
from typing import List
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_path: str) -> List[str]:
    """
    Extract text from a PDF file and split into chunks.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of text chunks
    """
    try:
        reader = PdfReader(file_path)
        chunks = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                chunks.append(text)
        
        return chunks
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into smaller chunks for better processing.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = end - chunk_overlap
    
    return chunks











