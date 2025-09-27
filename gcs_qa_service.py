#!/usr/bin/env python3
"""
Google Cloud Storage Q&A Service
Handles Q&A functionality using Google Cloud Storage instead of Pinecone
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from google_cloud_storage_service import GoogleCloudStorageService
from direct_transcript_qa import DirectTranscriptQA
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class GCSQAService:
    """Q&A service using Google Cloud Storage for transcript storage and retrieval"""
    
    def __init__(self, gcs_bucket_name: str = None):
        """Initialize GCS Q&A service"""
        self.gcs_service = GoogleCloudStorageService(
            bucket_name=gcs_bucket_name,
            service_account_path='service-account-key.json'
        )
        self.direct_qa = DirectTranscriptQA()
        self.document_processor = DocumentProcessor()
    
    async def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict[str, Any]:
        """
        Answer a question using both transcript and document data from Google Cloud Storage
        
        Args:
            question: The question to answer
            company_name: Company name
            qudemo_id: QuDemo ID
            
        Returns:
            Dict with answer, timestamp, and metadata
        """
        try:
            logger.info(f"❓ Processing question: {question}")
            logger.info(f"🏢 Company: {company_name}, QuDemo: {qudemo_id}")
            
            # Search for relevant content in video transcripts
            video_answer_data = self.gcs_service.search_transcript_directly(
                company_name=company_name,
                qudemo_id=qudemo_id,
                question=question
            )
            
            # Search for relevant content in documents
            print(f"DEBUG: Calling document search with company_name='{company_name}', qudemo_id='{qudemo_id}', query='{question}'")
            document_results = self.document_processor.search_document_content(
                company_name=company_name,
                qudemo_id=qudemo_id,
                query=question
            )
            
            logger.info(f"📄 Document search results: {len(document_results)} documents found")
            if document_results:
                for i, result in enumerate(document_results):
                    logger.info(f"📄 Document {i+1}: {result.get('filename', 'Unknown')} - {len(result.get('relevant_sections', []))} sections")
            
            # Combine results and determine best response
            combined_answer = self._combine_video_and_document_results(
                video_answer_data, document_results, question
            )
            
            if not combined_answer:
                return {
                    'success': False,
                    'error': 'No relevant information found',
                    'answer': 'No relevant information found',
                    'timestamp': 0,
                    'formatted_timestamp': '00:00',
                    'confidence': 0.0,
                    'video_url': '',
                    'video_title': ''
                }
            
            # Store the Q&A answer
            if combined_answer:
                self.gcs_service.store_qa_answer(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    question=question,
                    answer_data=combined_answer
                )
            
            # Get video URL and title from combined_answer or video_answer_data
            video_url = combined_answer.get('video_url', '') if combined_answer else ''
            video_title = combined_answer.get('video_title', '') if combined_answer else ''
            
            # If no video URL found, try to get it from the transcript data
            if not video_url:
                transcript_data = self.gcs_service.get_video_transcript(company_name, qudemo_id)
                if transcript_data and 'videos' in transcript_data:
                    # Multi-video format - find the video that matches the timestamp
                    start_timestamp = combined_answer.get('timestamp', 0) if combined_answer else 0
                    for video in transcript_data['videos']:
                        video_timestamps = video.get('timestamps', [])
                        for segment in video_timestamps:
                            if abs(segment.get('start_timestamp', 0) - start_timestamp) < 5:  # Within 5 seconds
                                video_url = video.get('video_url', '')
                                video_title = video.get('video_title', '')
                                break
                        if video_url:
                            break
                elif transcript_data:
                    # Single video format
                    video_url = transcript_data.get('video_url', '')
                    video_title = transcript_data.get('video_title', '')
            
            # Calculate end timestamp from formatted timestamp or add duration
            start_timestamp = combined_answer.get('timestamp', 0) if combined_answer else 0
            formatted_timestamp = combined_answer.get('formatted_timestamp', '00:00-00:00') if combined_answer else '00:00-00:00'
            
            # Try to extract end time from formatted timestamp
            if '-' in formatted_timestamp:
                try:
                    # Parse the end time from formatted timestamp like "00:22-00:30"
                    end_part = formatted_timestamp.split('-')[1]
                    if ':' in end_part:
                        parts = end_part.split(':')
                        if len(parts) == 2:  # MM:SS
                            end_timestamp = int(parts[0]) * 60 + int(parts[1])
                        elif len(parts) == 3:  # HH:MM:SS
                            end_timestamp = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        else:
                            end_timestamp = start_timestamp + 30
                    else:
                        end_timestamp = start_timestamp + 30
                except:
                    end_timestamp = start_timestamp + 30
            else:
                end_timestamp = start_timestamp + 30
            
            return {
                'success': True,
                'answer': combined_answer.get('answer', '') if combined_answer else 'No relevant information found',
                'timestamp': start_timestamp,
                'end': end_timestamp,
                'formatted_timestamp': combined_answer.get('formatted_timestamp', '00:00') if combined_answer else '00:00',
                'video_url': video_url,
                'video_title': video_title,
                'confidence': combined_answer.get('confidence', 0.0) if combined_answer else 0.0,
                'sources': combined_answer.get('sources', []) if combined_answer else [],
                'answer_source': 'gcs_transcript_search'
            }
            
        except Exception as e:
            logger.error(f"❌ Q&A processing failed: {e}")
            return {
                'success': False,
                'error': f'Q&A processing failed: {str(e)}',
                'answer': 'Sorry, I encountered an error while processing your question.',
                'timestamp': 0,
                'formatted_timestamp': '00:00',
                'confidence': 0.0
            }
    
    def get_transcript_info(self, company_name: str, qudemo_id: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a transcript"""
        try:
            transcript_data = self.gcs_service.get_video_transcript(company_name, qudemo_id)
            if not transcript_data:
                return None
            
            return {
                'video_title': transcript_data.get('video_title', 'Unknown'),
                'video_url': transcript_data.get('video_url', ''),
                'word_count': transcript_data.get('metadata', {}).get('word_count', 0),
                'language': transcript_data.get('metadata', {}).get('language', 'en'),
                'processed_at': transcript_data.get('processed_at', ''),
                'chunks_count': len(transcript_data.get('chunks', []))
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get transcript info: {e}")
            return None
    
    def list_qudemos(self, company_name: str) -> List[str]:
        """List all qudemo IDs for a company"""
        return self.gcs_service.list_qudemos(company_name)
    
    def delete_qudemo(self, company_name: str, qudemo_id: str) -> bool:
        """Delete all data for a specific qudemo"""
        return self.gcs_service.delete_qudemo_data(company_name, qudemo_id)
    
    def generate_and_store_suggested_questions(self, company_name: str, qudemo_id: str) -> List[str]:
        """Generate and store suggested questions for a QuDemo"""
        try:
            logger.info(f"🤖 Generating suggested questions for {company_name}/{qudemo_id}")
            
            # Get transcript data
            transcript_data = self.gcs_service.get_video_transcript(company_name, qudemo_id)
            if not transcript_data:
                logger.warning(f"⚠️ No transcript found for {company_name}/{qudemo_id}")
                return []
            
            # Generate suggested questions using the direct QA system
            suggested_questions = self.direct_qa.generate_suggested_questions(transcript_data)
            
            if suggested_questions:
                # Store suggested questions in GCS
                success = self.gcs_service.store_suggested_questions(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    suggested_questions=suggested_questions
                )
                
                if success:
                    logger.info(f"✅ Generated and stored {len(suggested_questions)} suggested questions for {company_name}/{qudemo_id}")
                    return suggested_questions
                else:
                    logger.error(f"❌ Failed to store suggested questions for {company_name}/{qudemo_id}")
                    return []
            else:
                logger.warning(f"⚠️ No suggested questions generated for {company_name}/{qudemo_id}")
                return []
            
        except Exception as e:
            logger.error(f"❌ Failed to generate suggested questions: {e}")
            return []
    
    def _combine_video_and_document_results(self, video_data: Optional[Dict], document_results: List[Dict], question: str) -> Optional[Dict]:
        """
        Combine video transcript and document search results to provide the best answer
        
        Args:
            video_data: Video transcript search results
            document_results: Document search results
            question: Original question
            
        Returns:
            Combined answer data or None if no relevant information found
        """
        try:
            has_video_answer = video_data and video_data.get('answer') and video_data.get('answer') != 'No relevant information found'
            has_document_answer = document_results and len(document_results) > 0
            
            if not has_video_answer and not has_document_answer:
                return None
            
            # If only video answer exists
            if has_video_answer and not has_document_answer:
                logger.info("📹 Using video-only answer")
                return {
                    'answer': video_data['answer'],
                    'timestamp': video_data.get('timestamp', 0),
                    'source': 'video',
                    'confidence': video_data.get('confidence', 0.8),
                    'metadata': {
                        'video_sources': video_data.get('metadata', {}).get('sources', []),
                        'document_sources': []
                    }
                }
            
            # If only document answer exists
            if not has_video_answer and has_document_answer:
                logger.info("📄 Using document-only answer")
                # Combine all document sections
                combined_document_text = []
                document_sources = []
                
                for doc_result in document_results:
                    if doc_result.get('relevant_sections'):
                        combined_document_text.extend(doc_result['relevant_sections'])
                        document_sources.append({
                            'filename': doc_result.get('filename', 'Unknown'),
                            'document_id': doc_result.get('document_id', ''),
                            'mime_type': doc_result.get('mime_type', '')
                        })
                
                # Format document answer using LLM for better quality (similar to video answers)
                formatted_answer = self._format_document_answer(question, combined_document_text)
                
                return {
                    'answer': formatted_answer,
                    'timestamp': 0,  # No video timestamp for document-only answers
                    'source': 'document',
                    'confidence': 0.7,
                    'metadata': {
                        'video_sources': [],
                        'document_sources': document_sources
                    }
                }
            
            # If both video and document answers exist, combine them
            if has_video_answer and has_document_answer:
                logger.info("📹📄 Combining video and document answers")
                
                # Combine document sections
                document_text = []
                document_sources = []
                
                for doc_result in document_results:
                    if doc_result.get('relevant_sections'):
                        document_text.extend(doc_result['relevant_sections'])
                        document_sources.append({
                            'filename': doc_result.get('filename', 'Unknown'),
                            'document_id': doc_result.get('document_id', ''),
                            'mime_type': doc_result.get('mime_type', '')
                        })
                
                # Create combined answer with intelligent formatting
                combined_answer = self._format_combined_answer(question, video_data['answer'], document_text[:2])
                
                return {
                    'answer': combined_answer,
                    'timestamp': video_data.get('timestamp', 0),
                    'source': 'combined',
                    'confidence': max(video_data.get('confidence', 0.8), 0.7),
                    'metadata': {
                        'video_sources': video_data.get('metadata', {}).get('sources', []),
                        'document_sources': document_sources
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error combining video and document results: {e}")
            return None
    
    def _format_document_answer(self, question: str, document_sections: List[str]) -> str:
        """
        Format document answer using LLM to match video answer quality
        
        Args:
            question: The original question
            document_sections: List of relevant document sections
            
        Returns:
            Formatted answer string
        """
        try:
            import openai
            
            # Combine document sections for context
            document_text = '\n\n'.join(document_sections[:5])  # Limit to first 5 sections
            
            # Create prompt similar to video answer formatting
            prompt = f"""You are an expert assistant that provides concise, professional answers based on document content.

Document Content:
{document_text}

Question: {question}

INSTRUCTIONS:
- Provide a direct, concise answer in 2-3 sentences maximum
- Use professional, business-ready language
- Focus on the core essence and key information
- Be intelligent and insightful (like ChatGPT)
- Interpret and analyze the content, don't just quote it
- Provide processed insights, not raw document text
- If the document doesn't contain relevant information, say "No relevant information found"

Return ONLY the answer text, no explanations or formatting."""
            
            # Call OpenAI API for formatting
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                top_p=0.95
            )
            
            formatted_answer = response.choices[0].message.content.strip()
            logger.info(f"📄 Formatted document answer: {formatted_answer[:100]}...")
            return formatted_answer
            
        except Exception as e:
            logger.error(f"❌ Failed to format document answer: {e}")
            # Fallback to simple truncation
            combined_text = ' '.join(document_sections[:2])
            if len(combined_text) > 500:
                combined_text = combined_text[:500] + "..."
            return combined_text
    
    def _format_combined_answer(self, question: str, video_answer: str, document_sections: List[str]) -> str:
        """
        Format combined video and document answer using LLM
        
        Args:
            question: The original question
            video_answer: Video-based answer
            document_sections: List of relevant document sections
            
        Returns:
            Formatted combined answer string
        """
        try:
            import openai
            
            # Combine document sections for context
            document_text = '\n\n'.join(document_sections[:3])  # Limit to first 3 sections
            
            # Create prompt for combined answer formatting
            prompt = f"""You are an expert assistant that combines video and document information to provide comprehensive answers.

Video Answer:
{video_answer}

Additional Document Information:
{document_text}

Question: {question}

INSTRUCTIONS:
- Combine the video and document information into a single, coherent answer
- Provide a direct, concise answer in 2-3 sentences maximum
- Use professional, business-ready language
- Focus on the core essence and key information
- Be intelligent and insightful (like ChatGPT)
- Integrate both sources naturally, don't just concatenate them
- Provide processed insights, not raw text
- Prioritize the most relevant information from both sources

Return ONLY the combined answer text, no explanations or formatting."""
            
            # Call OpenAI API for formatting
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250,
                top_p=0.95
            )
            
            formatted_answer = response.choices[0].message.content.strip()
            logger.info(f"📹📄 Formatted combined answer: {formatted_answer[:100]}...")
            return formatted_answer
            
        except Exception as e:
            logger.error(f"❌ Failed to format combined answer: {e}")
            # Fallback to simple combination
            return f"{video_answer}\n\nAdditional information from documentation: {' '.join(document_sections[:2])}"
    
    def get_suggested_questions(self, company_name: str, qudemo_id: str) -> List[str]:
        """Get suggested questions for a QuDemo"""
        try:
            suggested_questions_data = self.gcs_service.get_suggested_questions(company_name, qudemo_id)
            if suggested_questions_data:
                return suggested_questions_data.get('suggested_questions', [])
            else:
                return []
        except Exception as e:
            logger.error(f"❌ Failed to get suggested questions: {e}")
            return []