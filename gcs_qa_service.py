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

logger = logging.getLogger(__name__)

class GCSQAService:
    """Q&A service using Google Cloud Storage for transcript storage and retrieval"""
    
    def __init__(self, gcs_bucket_name: str = None):
        """Initialize GCS Q&A service"""
        self.gcs_service = GoogleCloudStorageService(
            bucket_name=gcs_bucket_name,
            service_account_path='service-account-key.json'
        )
    
    async def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict[str, Any]:
        """
        Answer a question using transcript data from Google Cloud Storage
        
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
            
            # Search for relevant content in the transcript using direct method
            answer_data = self.gcs_service.search_transcript_directly(
                company_name=company_name,
                qudemo_id=qudemo_id,
                question=question
            )
            
            if not answer_data:
                return {
                    'success': False,
                    'error': 'No relevant content found for this question',
                    'answer': 'I could not find relevant information to answer your question. The available video content does not contain information about this topic.',
                    'timestamp': 0,
                    'formatted_timestamp': '00:00',
                    'confidence': 0.0,
                    'video_url': '',
                    'video_title': ''
                }
            
            # Store the Q&A answer
            self.gcs_service.store_qa_answer(
                company_name=company_name,
                qudemo_id=qudemo_id,
                question=question,
                answer_data=answer_data
            )
            
            # Get video URL and title from answer_data (which comes from direct transcript search)
            # Handle multiple videos by getting the correct one from the answer data
            video_url = answer_data.get('video_url', '')
            video_title = answer_data.get('video_title', '')
            
            # If no video URL found, try to get it from the transcript data
            if not video_url:
                transcript_data = self.gcs_service.get_video_transcript(company_name, qudemo_id)
                if transcript_data and 'videos' in transcript_data:
                    # Multi-video format - find the video that matches the timestamp
                    start_timestamp = answer_data.get('timestamp', 0)
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
            start_timestamp = answer_data.get('timestamp', 0)
            formatted_timestamp = answer_data.get('formatted_timestamp', '00:00-00:00')
            
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
                'answer': answer_data.get('answer', ''),
                'timestamp': start_timestamp,
                'end': end_timestamp,
                'formatted_timestamp': answer_data.get('formatted_timestamp', '00:00'),
                'video_url': video_url,
                'video_title': video_title,
                'confidence': answer_data.get('confidence', 0.0),
                'sources': answer_data.get('sources', []),
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
