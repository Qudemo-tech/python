#!/usr/bin/env python3
"""
Document Processor
Handles text extraction from various document formats
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from google_cloud_storage_service import GoogleCloudStorageService

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents and extracts text content"""
    
    def __init__(self, gcs_bucket_name: str = None):
        """Initialize document processor"""
        self.gcs_service = GoogleCloudStorageService(
            bucket_name=gcs_bucket_name,
            service_account_path='service-account-key.json'
        )
        logger.info("✅ Document Processor initialized")
    
    def process_document_from_content(self, company_name: str, qudemo_id: str, document_id: str, 
                                     file_content: bytes, mime_type: str, filename: str) -> bool:
        """
        Process a document from file content and extract text
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            document_id: Document ID
            file_content: Document content as bytes
            mime_type: MIME type of the document
            filename: Original filename
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            logger.info(f"📄 Processing document: {document_id} ({mime_type})")
            
            # Extract text from content
            extracted_text = self.extract_text_from_content(file_content, mime_type)
            if not extracted_text:
                logger.error(f"❌ Failed to extract text from document: {document_id}")
                return False
            
            # Structure the extracted text
            document_data = {
                "document_id": document_id,
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "filename": filename,
                "mime_type": mime_type,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "processing_status": "completed"
            }
            
            # Store extracted text in GCS
            text_file_path = f"{company_name}/{qudemo_id}/documents/{document_id}/extracted_text.json"
            success = self.gcs_service.upload_file_content(
                json.dumps(document_data, indent=2),
                text_file_path,
                'application/json'
            )
            
            if success:
                logger.info(f"✅ Document processed successfully: {document_id}")
                logger.info(f"📊 Extracted {len(extracted_text)} characters of text")
                return True
            else:
                logger.error(f"❌ Failed to store extracted text: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing document {document_id}: {e}")
            return False

    def process_document(self, company_name: str, qudemo_id: str, document_id: str, 
                        file_path: str, mime_type: str) -> bool:
        """
        Process a document and extract text content
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            document_id: Document ID
            file_path: Path to the document in GCS
            mime_type: MIME type of the document
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            logger.info(f"📄 Processing document: {document_id} ({mime_type})")
            
            # Download document from GCS
            document_content = self.gcs_service.download_file(file_path)
            if not document_content:
                logger.error(f"❌ Failed to download document: {file_path}")
                return False
            
            # Extract text based on file type
            extracted_text = self.extract_text_from_content(document_content, mime_type)
            if not extracted_text:
                logger.error(f"❌ Failed to extract text from document: {document_id}")
                return False
            
            # Structure the extracted text
            document_data = {
                "document_id": document_id,
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "file_path": file_path,
                "mime_type": mime_type,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "processing_status": "completed"
            }
            
            # Store extracted text in GCS
            text_file_path = f"{company_name}/{qudemo_id}/documents/{document_id}/extracted_text.json"
            success = self.gcs_service.upload_file_content(
                json.dumps(document_data, indent=2),
                text_file_path,
                'application/json'
            )
            
            if success:
                logger.info(f"✅ Document processed successfully: {document_id}")
                logger.info(f"📊 Extracted {len(extracted_text)} characters of text")
                return True
            else:
                logger.error(f"❌ Failed to store extracted text: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing document {document_id}: {e}")
            return False
    
    def extract_text_from_content(self, content: bytes, mime_type: str) -> Optional[str]:
        """
        Extract text from document content based on MIME type
        
        Args:
            content: Document content as bytes
            mime_type: MIME type of the document
            
        Returns:
            str: Extracted text or None if failed
        """
        try:
            if mime_type == 'application/pdf':
                return self._extract_pdf_text(content)
            elif mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                return self._extract_word_text(content, mime_type)
            elif mime_type in ['application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation']:
                return self._extract_powerpoint_text(content, mime_type)
            elif mime_type == 'text/plain':
                return self._extract_plain_text(content)
            else:
                logger.warning(f"⚠️ Unsupported file type: {mime_type}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error extracting text: {e}")
            return None
    
    def _extract_pdf_text(self, content: bytes) -> Optional[str]:
        """Extract text from PDF content"""
        try:
            import PyPDF2
            import io
            
            logger.info(f"📄 Extracting text from PDF ({len(content)} bytes)")
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            
            logger.info(f"📄 PDF has {len(pdf_reader.pages)} pages")
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"
                logger.info(f"📄 Page {page_num + 1}: {len(page_text)} characters")
            
            extracted_text = text.strip()
            logger.info(f"📄 Total extracted text: {len(extracted_text)} characters")
            return extracted_text
            
        except ImportError:
            logger.error("❌ PyPDF2 not installed. Install with: pip install PyPDF2")
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting PDF text: {e}")
            return None
    
    def _extract_word_text(self, content: bytes, mime_type: str) -> Optional[str]:
        """Extract text from Word document content"""
        try:
            if mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # DOCX file
                import docx
                import io
                
                doc = docx.Document(io.BytesIO(content))
                text = ""
                
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                
                return text.strip()
                
            elif mime_type == 'application/msword':
                # DOC file - requires additional library
                logger.warning("⚠️ DOC files require python-docx2txt or similar library")
                return None
                
        except ImportError:
            logger.error("❌ python-docx not installed. Install with: pip install python-docx")
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting Word text: {e}")
            return None
    
    def _extract_powerpoint_text(self, content: bytes, mime_type: str) -> Optional[str]:
        """Extract text from PowerPoint content"""
        try:
            if mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
                # PPTX file
                from pptx import Presentation
                import io
                
                prs = Presentation(io.BytesIO(content))
                text = ""
                
                for slide_num, slide in enumerate(prs.slides):
                    text += f"--- Slide {slide_num + 1} ---\n"
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text += shape.text + "\n"
                    text += "\n"
                
                return text.strip()
                
            elif mime_type == 'application/vnd.ms-powerpoint':
                # PPT file - requires additional library
                logger.warning("⚠️ PPT files require python-pptx or similar library")
                return None
                
        except ImportError:
            logger.error("❌ python-pptx not installed. Install with: pip install python-pptx")
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting PowerPoint text: {e}")
            return None
    
    def _extract_plain_text(self, content: bytes) -> Optional[str]:
        """Extract text from plain text content"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    return content.decode(encoding).strip()
                except UnicodeDecodeError:
                    continue
            
            logger.error("❌ Could not decode text file with any supported encoding")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting plain text: {e}")
            return None
    
    def get_document_text(self, company_name: str, qudemo_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve extracted text for a document
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            document_id: Document ID
            
        Returns:
            dict: Document data with extracted text or None if not found
        """
        try:
            text_file_path = f"{company_name}/{qudemo_id}/documents/{document_id}/extracted_text.json"
            content = self.gcs_service.download_file_content(text_file_path)
            
            if content:
                return json.loads(content)
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving document text: {e}")
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
                                'filename': doc_info.get('file_path', '').split('/')[-1],
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

    def _find_relevant_sections(self, text: str, query: str, max_sections: int = 3) -> List[str]:
        """Find relevant sections of text containing the query"""
        try:
            sentences = text.split('.')
            relevant_sections = []
            
            for sentence in sentences:
                if query in sentence.lower() and len(sentence.strip()) > 20:
                    relevant_sections.append(sentence.strip())
                    if len(relevant_sections) >= max_sections:
                        break
            
            return relevant_sections
            
        except Exception as e:
            logger.error(f"❌ Error finding relevant sections: {e}")
            return []
