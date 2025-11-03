#!/usr/bin/env python3
"""
Document Processor for extracting text from various document types
"""

import logging
import json
import tempfile
import os
from typing import List, Dict, Any, Optional
import PyPDF2
from docx import Document
from pptx import Presentation
from google_cloud_storage_service import GoogleCloudStorageService

# Configure logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents and extract text content for Q&A"""
    
    def __init__(self):
        """Initialize document processor"""
        self.gcs_service = GoogleCloudStorageService()
        logger.info("✅ Document Processor initialized")
    
    def process_document_from_content(
        self, 
        company_name: str, 
        qudemo_id: str, 
        document_id: str, 
        file_content: bytes, 
        mime_type: str, 
        filename: str
    ) -> bool:
        """
        Process document from file content and store in GCS
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            document_id: Document ID
            file_content: File content as bytes
            mime_type: MIME type of the file
            filename: Original filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"📄 Processing document: {document_id} for {company_name}/{qudemo_id}")
            
            # Extract text from document
            extracted_text = self.extract_text_from_content(file_content, mime_type, filename)
            
            if not extracted_text:
                logger.error(f"❌ No text extracted from document: {document_id}")
                return False
            
            # Structure the extracted text
            document_data = {
                'document_id': document_id,
                'filename': filename,
                'mime_type': mime_type,
                'extracted_text': extracted_text,
                'processed_at': json.dumps({'timestamp': 'now'})  # Simple timestamp
            }
            
            # Store in GCS
            file_path = f"{company_name}/{qudemo_id}/documents/{document_id}/extracted_text.json"
            success = self.gcs_service.upload_file_content(
                content=json.dumps(document_data, indent=2),
                file_path=file_path,
                content_type='application/json'
            )
            
            if success:
                logger.info(f"✅ Document processed and stored: {document_id}")
                return True
            else:
                logger.error(f"❌ Failed to store document: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing document {document_id}: {e}")
            return False
    
    def extract_text_from_content(self, file_content: bytes, mime_type: str, filename: str) -> Optional[str]:
        """
        Extract text from file content based on MIME type
        
        Args:
            file_content: File content as bytes
            mime_type: MIME type of the file
            filename: Original filename
            
        Returns:
            Extracted text or None if extraction failed
        """
        try:
            logger.info(f"📄 Extracting text from {filename} (type: {mime_type})")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Extract text based on MIME type
                if mime_type == 'application/pdf':
                    return self._extract_pdf_text(temp_file_path)
                elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
                    return self._extract_docx_text(temp_file_path)
                elif mime_type in ['application/vnd.openxmlformats-officedocument.presentationml.presentation', 'application/vnd.ms-powerpoint']:
                    return self._extract_pptx_text(temp_file_path)
                elif mime_type == 'text/plain':
                    return self._extract_txt_text(temp_file_path)
                else:
                    logger.warning(f"⚠️ Unsupported file type: {mime_type}")
                    return None
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"❌ Error extracting text from {filename}: {e}")
            return None
    
    def _extract_pdf_text(self, file_path: str) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"📄 PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Page {page_num + 1} ---\n\n"
                        text += page_text
                
                logger.info(f"📄 Extracted {len(text)} characters from PDF")
                return text.strip()
                
        except Exception as e:
            logger.error(f"❌ Error extracting PDF text: {e}")
            return None
    
    def _extract_docx_text(self, file_path: str) -> Optional[str]:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            logger.info(f"📄 Extracted {len(text)} characters from DOCX")
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Error extracting DOCX text: {e}")
            return None
    
    def _extract_pptx_text(self, file_path: str) -> Optional[str]:
        """Extract text from PPTX file"""
        try:
            prs = Presentation(file_path)
            text = ""
            
            for slide_num, slide in enumerate(prs.slides):
                text += f"\n\n--- Slide {slide_num + 1} ---\n\n"
                
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text += shape.text + "\n"
            
            logger.info(f"📄 Extracted {len(text)} characters from PPTX")
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Error extracting PPTX text: {e}")
            return None
    
    def _extract_txt_text(self, file_path: str) -> Optional[str]:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            
            logger.info(f"📄 Extracted {len(text)} characters from TXT")
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Error extracting TXT text: {e}")
            return None
    
    def search_document_content(self, company_name: str, qudemo_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Search for content in all documents for a QuDemo
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            query: Search query
            
        Returns:
            list: List of matching document content
        """
        try:
            logger.info(f"🔍 Searching documents for: {company_name}/{qudemo_id} - Query: '{query}'")
            print(f"DEBUG: Document search called with query: {query}")
            
            # Get all documents for this QuDemo
            documents_path = f"{company_name}/{qudemo_id}/documents/"
            print(f"DEBUG: Looking for documents in path: {documents_path}")
            document_folders = self.gcs_service.list_files(documents_path)
            
            logger.info(f"📁 Found {len(document_folders)} document folders: {document_folders}")
            print(f"DEBUG: Found {len(document_folders)} document folders: {document_folders}")
            
            results = []
            query_lower = query.lower()
            
            print(f"DEBUG: Processing {len(document_folders)} document folders")
            for folder in document_folders:
                print(f"DEBUG: Processing folder: {folder}")
                if folder.endswith('/'):
                    document_id = folder.rstrip('/').split('/')[-1]
                    text_file_path = f"{folder}extracted_text.json"
                elif folder.endswith('extracted_text.json'):
                    # Handle direct file paths
                    document_id = folder.split('/')[-2]  # Get document ID from parent folder
                    text_file_path = folder
                    print(f"DEBUG: Direct file path detected, document_id: {document_id}")
                else:
                    print(f"DEBUG: Skipping non-folder, non-extracted_text.json path: {folder}")
                    continue
                
                print(f"DEBUG: About to download document content from: {text_file_path}")
                logger.info(f"📄 Downloading document content from: {text_file_path}")
                document_data = self.gcs_service.download_file_content(text_file_path)
                
                if document_data:
                    print(f"DEBUG: Successfully downloaded document content, length: {len(document_data)}")
                    try:
                        doc_info = json.loads(document_data)
                        text = doc_info.get('extracted_text', '')
                        
                        logger.info(f"📄 Document {document_id}: {len(text)} characters of text")
                        
                        # If query is empty, return ALL document content (for FAQ generation)
                        if not query or query.strip() == "":
                            logger.info(f"📋 Empty query - returning ALL content from document {document_id}")
                            results.append({
                                'document_id': document_id,
                                'filename': doc_info.get('filename', '').split('/')[-1],
                                'mime_type': doc_info.get('mime_type', ''),
                                'content': text,  # Return full text
                                'source': 'document',
                                'matched_terms': []
                            })
                            logger.info(f"✅ Added full document content ({len(text)} chars) from {document_id}")
                        else:
                            # Intelligent text search using key terms
                            search_terms = self._extract_search_terms(query_lower)
                            logger.info(f"🔍 Extracted search terms: {search_terms}")
                            print(f"DEBUG: Extracted {len(search_terms)} search terms: {search_terms[:5]}...")
                            
                            # Check if any search terms are found in the text
                            found_terms = []
                            for term in search_terms:
                                if term in text.lower():
                                    found_terms.append(term)
                            
                            print(f"DEBUG: Found {len(found_terms)} matching terms: {found_terms}")
                            
                            if found_terms:
                                logger.info(f"✅ Found terms {found_terms} in document {document_id}")
                                # Find relevant sections using the most relevant term
                                best_term = found_terms[0]  # Use the first found term
                                relevant_sections = self._find_relevant_sections(text, best_term)
                                
                                results.append({
                                    'document_id': document_id,
                                    'filename': doc_info.get('filename', '').split('/')[-1],
                                    'mime_type': doc_info.get('mime_type', ''),
                                    'relevant_sections': relevant_sections,
                                    'source': 'document',
                                    'matched_terms': found_terms
                                })
                                
                                logger.info(f"📄 Added {len(relevant_sections)} relevant sections from document {document_id}")
                            else:
                                logger.info(f"❌ No search terms found in document {document_id}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error for document {document_id}: {e}")
                        continue
                else:
                    logger.error(f"❌ Failed to download document content from: {text_file_path}")
            
            logger.info(f"📊 Document search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching document content: {e}")
            return []
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract key search terms from a query"""
        try:
            # Remove common question words and get meaningful terms
            stop_words = {
                'what', 'is', 'the', 'and', 'of', 'in', 'to', 'for', 'with', 'by', 'from', 'on', 'at', 'as', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'can', 'a', 'an', 'this', 'that', 'these', 'those'
            }
            
            # Split query into words and filter out stop words
            words = query.lower().split()
            meaningful_words = [word.strip('.,!?;:') for word in words if word not in stop_words and len(word) > 2]
            
            # Create search terms of different lengths (1-3 words)
            search_terms = []
            
            # Single words
            search_terms.extend(meaningful_words)
            
            # Two-word phrases
            for i in range(len(meaningful_words) - 1):
                phrase = f"{meaningful_words[i]} {meaningful_words[i+1]}"
                search_terms.append(phrase)
            
            # Three-word phrases for important terms
            if len(meaningful_words) >= 3:
                for i in range(len(meaningful_words) - 2):
                    phrase = f"{meaningful_words[i]} {meaningful_words[i+1]} {meaningful_words[i+2]}"
                    search_terms.append(phrase)
            
            # Remove duplicates and return
            return list(dict.fromkeys(search_terms))  # Preserves order while removing duplicates
            
        except Exception as e:
            logger.error(f"❌ Error extracting search terms: {e}")
            return [query]  # Fallback to original query
    
    def _find_relevant_sections(self, text: str, search_term: str) -> List[str]:
        """Find relevant sections of text around search terms"""
        try:
            sections = []
            text_lower = text.lower()
            term_lower = search_term.lower()
            
            # Find all occurrences of the search term
            start = 0
            while True:
                pos = text_lower.find(term_lower, start)
                if pos == -1:
                    break
                
                # Extract context around the term (200 characters before and after)
                context_start = max(0, pos - 200)
                context_end = min(len(text), pos + len(search_term) + 200)
                
                context = text[context_start:context_end]
                sections.append(context.strip())
                
                start = pos + 1
            
            # Remove duplicates and limit to 5 sections
            unique_sections = list(dict.fromkeys(sections))
            return unique_sections[:5]
            
        except Exception as e:
            logger.error(f"❌ Error finding relevant sections: {e}")
            return []
