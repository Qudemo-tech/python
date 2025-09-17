#!/usr/bin/env python3
"""
Google Cloud Storage Service
Handles storing and retrieving video transcripts and Q&A data from Google Cloud Storage
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from google.cloud import storage
from google.oauth2 import service_account
from direct_transcript_qa import DirectTranscriptQA

logger = logging.getLogger(__name__)

class GoogleCloudStorageService:
    """Service for managing video transcripts and Q&A data in Google Cloud Storage"""
    
    def __init__(self, bucket_name: str = None, service_account_path: str = None):
        """Initialize Google Cloud Storage service"""
        # Use a default bucket name, but we'll create company-specific buckets
        self.default_bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'qudemo-video-transcripts')
        self.service_account_path = service_account_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service-account-key.json')
        self.bucket = None  # Will be set when we create/access a company bucket
        
        # Initialize direct transcript Q&A system
        self.direct_qa = DirectTranscriptQA()
        
        # Initialize Google Cloud Storage client
        try:
            if self.service_account_path and os.path.exists(self.service_account_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_path
                )
                self.client = storage.Client(credentials=credentials)
            else:
                # Use default credentials (for production)
                self.client = storage.Client()
            
            # Initialize client only, buckets will be created per company
            logger.info(f"✅ Google Cloud Storage client initialized")
            logger.info(f"📦 Will create company-specific buckets as needed")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Cloud Storage: {e}")
            raise
    
    def _get_company_bucket(self, company_name: str):
        """Get or create a bucket for a specific company"""
        try:
            # Create company-specific bucket name
            safe_company_name = company_name.lower().replace(' ', '-').replace('_', '-')
            bucket_name = f"qudemo-{safe_company_name}"
            
            # Get or create the bucket
            bucket = self.client.bucket(bucket_name)
            
            if not bucket.exists():
                # Create the bucket
                bucket = self.client.create_bucket(bucket_name)
                logger.info(f"✅ Created company bucket: {bucket_name}")
            else:
                logger.info(f"✅ Connected to existing company bucket: {bucket_name}")
            
            return bucket
            
        except Exception as e:
            logger.error(f"❌ Failed to get/create company bucket for {company_name}: {e}")
            raise
    
    def store_video_transcript(self, company_name: str, qudemo_id: str, 
                             transcript_data: Dict[str, Any]) -> bool:
        """Store video transcript data in Google Cloud Storage with company/qudemo structure"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: qudemo_id/transcript.json (within company bucket)
            file_path = f"{qudemo_id}/transcript.json"
            
            # Get existing transcript data or create new
            existing_data = self.get_video_transcript(company_name, qudemo_id) or {
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "processed_at": datetime.now().isoformat(),
                "videos": []
            }
            
            # Add new video transcript
            video_transcript = {
                "video_url": transcript_data.get('video_url', ''),
                "video_title": transcript_data.get('video_title', ''),
                "transcript": transcript_data.get('transcript', ''),
                "timestamps": transcript_data.get('timestamps', []),
                "segments": transcript_data.get('segments', []),
                "topics": transcript_data.get('topics', []),
                "chunks": transcript_data.get('chunks', []),
                "processed_at": datetime.now().isoformat()
            }
            
            # Add to videos array
            existing_data['videos'].append(video_transcript)
            existing_data['updated_at'] = datetime.now().isoformat()
            
            # Use existing_data as transcript_with_metadata
            transcript_with_metadata = existing_data
            
            # Upload to Google Cloud Storage
            blob = bucket.blob(file_path)
            blob.upload_from_string(
                json.dumps(transcript_with_metadata, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"✅ Stored transcript for {company_name}/{qudemo_id} in GCS")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store transcript: {e}")
            return False
    
    def get_video_transcript(self, company_name: str, qudemo_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve video transcript data from Google Cloud Storage with company/qudemo structure"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: qudemo_id/transcript.json (within company bucket)
            file_path = f"{qudemo_id}/transcript.json"
            blob = bucket.blob(file_path)
            
            if not blob.exists():
                logger.warning(f"⚠️ Transcript not found: {file_path}")
                return None
            
            # Download and parse JSON
            content = blob.download_as_text()
            transcript_data = json.loads(content)
            
            logger.info(f"✅ Retrieved transcript for {company_name}/{qudemo_id} from GCS")
            return transcript_data
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve transcript: {e}")
            return None
    
    def store_qa_answer(self, company_name: str, qudemo_id: str, 
                       question: str, answer_data: Dict[str, Any]) -> bool:
        """Store Q&A answer in Google Cloud Storage with company/qudemo structure"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: qudemo_id/qa_answers.json (within company bucket)
            file_path = f"{qudemo_id}/qa_answers.json"
            
            # Get existing Q&A data or create new
            existing_data = self.get_qa_answers(company_name, qudemo_id) or {"answers": []}
            
            # Add new answer
            answer_entry = {
                "question": question,
                "answer": answer_data.get('answer', ''),
                "timestamp": answer_data.get('start', 0),
                "formatted_timestamp": answer_data.get('formatted_timestamp', ''),
                "confidence": answer_data.get('confidence', 0),
                "sources": answer_data.get('sources', []),
                "created_at": datetime.now().isoformat()
            }
            
            existing_data["answers"].append(answer_entry)
            
            # Upload updated data
            blob = bucket.blob(file_path)
            blob.upload_from_string(
                json.dumps(existing_data, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"✅ Stored Q&A answer for {company_name}/{qudemo_id} in GCS")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store Q&A answer: {e}")
            return False
    
    def get_qa_answers(self, company_name: str, qudemo_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve Q&A answers from Google Cloud Storage with company/qudemo structure"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: qudemo_id/qa_answers.json (within company bucket)
            file_path = f"{qudemo_id}/qa_answers.json"
            blob = bucket.blob(file_path)
            
            if not blob.exists():
                return None
            
            content = blob.download_as_text()
            qa_data = json.loads(content)
            
            return qa_data
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve Q&A answers: {e}")
            return None
    
    
    def search_transcript_directly(self, company_name: str, qudemo_id: str, 
                                 question: str) -> Optional[Dict[str, Any]]:
        """Search transcript directly using raw transcript data instead of chunks"""
        try:
            # Get transcript data
            transcript_data = self.get_video_transcript(company_name, qudemo_id)
            if not transcript_data:
                return None
            
            logger.info(f"🔍 Searching directly through transcript for question: {question}")
            
            # Use direct transcript Q&A system
            result = self.direct_qa.search_transcript_directly(transcript_data, question)
            
            if result:
                # Add metadata (video_url and video_title are already set by direct_qa)
                result['processed_at'] = transcript_data.get('processed_at', '')
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to search transcript directly: {e}")
            return None
    
    def _get_anchor_chunks(self, chunks: List[Dict], question: str, k: int = 60) -> List[Dict]:
        """Get top k chunks by keyword score (anchor step)"""
        question_lower = question.lower()
        question_keywords = [word for word in question_lower.split() if len(word) > 2]
        
        # Calculate keyword scores for all chunks
        chunk_scores = []
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            keyword_matches = sum(1 for keyword in question_keywords if keyword in chunk_text)
            base_relevance = keyword_matches / len(question_keywords) if question_keywords else 0
            
            # Add metadata for timestamp calculation
            chunk_metadata = chunk.get('metadata', {})
            segment_start = chunk_metadata.get('segment_start', '00:00')
            segment_end = chunk_metadata.get('segment_end', '00:00')
            
            # Convert timestamp to seconds
            timestamp_seconds = 0
            if ':' in segment_start:
                parts = segment_start.split(':')
                if len(parts) == 2:  # MM:SS
                    timestamp_seconds = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:  # HH:MM:SS
                    timestamp_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            
            chunk_scores.append({
                'text': chunk.get('text', ''),
                'timestamp': timestamp_seconds,
                'formatted_timestamp': f"{segment_start}-{segment_end}",
                'relevance_score': base_relevance,
                'chunk_id': chunk.get('id', ''),
                'metadata': chunk_metadata
            })
        
        # Sort by relevance score and return top k
        chunk_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
        return chunk_scores[:k]
    
    def _calculate_bm25_mean(self, section: Dict, question: str) -> float:
        """Calculate BM25 mean score for a section"""
        # Simple implementation - in production, use proper BM25
        question_lower = question.lower()
        question_keywords = [word for word in question_lower.split() if len(word) > 2]
        
        section_text = section.get('text', '').lower()
        keyword_matches = sum(1 for keyword in question_keywords if keyword in section_text)
        return keyword_matches / len(question_keywords) if question_keywords else 0
    
    def _log_section_selection(self, question: str, scored_sections: List[Tuple[Dict, float]]):
        """Log detailed section selection process"""
        print(f"\n🔍 DETAILED SECTION SELECTION LOG for: '{question}'")
        print("=" * 80)
        print(f"📊 Total sections found: {len(scored_sections)}")
        print("-" * 80)
        
        for i, (section, score) in enumerate(scored_sections):
            start_time = section.get('start_seconds', 0)
            end_time = section.get('end_seconds', 0)
            chunk_count = section.get('chunk_count', 0)
            text_preview = section.get('text', '')[:100]
            
            print(f"Section {i+1} [{start_time//60:02d}:{start_time%60:02d}-{end_time//60:02d}:{end_time%60:02d}] Score: {score:.3f}")
            print(f"  Chunks: {chunk_count}, Text: {text_preview}...")
            
            # Show why this section got this score
            section_text_lower = section.get('text', '').lower()
            if 'disqualified' in section_text_lower:
                print(f"  ✅ Contains 'disqualified' keyword")
            if 'build a new agent' in section_text_lower:
                print(f"  ✅ Contains 'build a new agent' phrase")
            if 'let me show you' in section_text_lower:
                print(f"  ✅ Contains 'let me show you' phrase")
            
            print()
        
        print("=" * 80)
    
    def _find_timestamp_for_sentence(self, sentence: str, timestamps: List[Dict]) -> float:
        """Find the timestamp for a given sentence"""
        try:
            # Simple approach: find the first timestamp that appears before this sentence
            # In production, you might want more sophisticated matching
            sentence_position = 0  # This would need to be calculated based on actual position
            
            for timestamp_info in timestamps:
                if timestamp_info.get('position', 0) <= sentence_position:
                    return timestamp_info.get('timestamp', 0)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Failed to find timestamp: {e}")
            return 0
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into MM:SS or HH:MM:SS format"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                return f"{minutes:02d}:{seconds:02d}"
                
        except Exception as e:
            logger.error(f"❌ Failed to format timestamp: {e}")
            return "00:00"
    
    def list_qudemos(self, company_name: str) -> List[str]:
        """List all qudemo IDs for a company using company-specific bucket"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # List all blobs in the company bucket
            blobs = self.client.list_blobs(bucket.name)
            
            qudemo_ids = set()
            for blob in blobs:
                # Extract qudemo_id from path: qudemo_id/filename
                path_parts = blob.name.split('/')
                if len(path_parts) >= 1:
                    qudemo_ids.add(path_parts[0])
            
            return list(qudemo_ids)
            
        except Exception as e:
            logger.error(f"❌ Failed to list qudemos: {e}")
            return []
    
    def delete_qudemo_data(self, company_name: str, qudemo_id: str) -> bool:
        """Delete all data for a specific qudemo using company-specific bucket"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # List all blobs for this qudemo
            prefix = f"{qudemo_id}/"
            blobs = self.client.list_blobs(bucket.name, prefix=prefix)
            
            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            
            logger.info(f"✅ Deleted {deleted_count} files for {company_name}/{qudemo_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete qudemo data: {e}")
            return False
    
    def list_companies(self) -> List[str]:
        """List all companies by checking for qudemo-* buckets"""
        try:
            # List all buckets and filter for qudemo-* buckets
            buckets = self.client.list_buckets()
            
            companies = []
            for bucket in buckets:
                if bucket.name.startswith('qudemo-'):
                    # Extract company name from bucket name: qudemo-company-name
                    company_name = bucket.name[7:]  # Remove 'qudemo-' prefix
                    companies.append(company_name)
            
            return companies
            
        except Exception as e:
            logger.error(f"❌ Failed to list companies: {e}")
            return []
    
    def get_storage_structure(self) -> Dict[str, Any]:
        """Get the complete storage structure for debugging"""
        try:
            # List all qudemo-* buckets
            buckets = self.client.list_buckets()
            
            structure = {}
            for bucket in buckets:
                if bucket.name.startswith('qudemo-'):
                    company_name = bucket.name[7:]  # Remove 'qudemo-' prefix
                    
                    # List all blobs in this company bucket
                    blobs = self.client.list_blobs(bucket.name)
                    
                    structure[company_name] = {}
                    for blob in blobs:
                        path_parts = blob.name.split('/')
                        if len(path_parts) >= 1:
                            qudemo = path_parts[0]
                            filename = path_parts[1] if len(path_parts) > 1 else 'unknown'
                            
                            if qudemo not in structure[company_name]:
                                structure[company_name][qudemo] = []
                            
                            structure[company_name][qudemo].append(filename)
            
            return structure
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage structure: {e}")
            return {}
    
    def _format_professional_answer(self, raw_answer: str, question: str, sources: list) -> str:
        """Return raw answer directly without template overwriting"""
        return raw_answer
    
    def _format_how_to_answer(self, content: str, question: str, has_steps: bool) -> str:
        """Return content directly without template overwriting"""
        return content
    
    def _format_manual_steps(self, content: str, question: str) -> str:
        """Return content directly without template overwriting"""
        return content
    
    def _format_testing_answer(self, content: str, question: str) -> str:
        """Return content directly without template overwriting"""
        return content
    
    def _format_explanation_answer(self, content: str, question: str) -> str:
        """Return content directly without template overwriting"""
        return content
    
    def _format_general_answer(self, content: str, question: str, has_steps: bool) -> str:
        """Return content directly without template overwriting"""
        return content
    
    def _extract_steps(self, content: str) -> list:
        """Extract step-by-step instructions from content"""
        import re
        steps = []
        
        # Clean up content first
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                not any(skip in sentence.lower() for skip in [
                    'next video', 'check out', 'watch us', 'if you want', 'see our next',
                    'to get started', 'now we get to test', 'if you want to watch'
                ]) and
                any(word in sentence.lower() for word in [
                    'first', 'next', 'then', 'find', 'click', 'set', 'create', 'open', 'enter',
                    'contact record', 'lead status', 'sidebar', 'information', 'task', 'follow up'
                ])):
                valid_sentences.append(sentence)
        
        # Look for step patterns in valid sentences
        for sentence in valid_sentences:
            # Look for "first we'll" patterns
            if re.search(r'first,?\s+we\'ll?', sentence, re.IGNORECASE):
                clean_step = re.sub(r'^first,?\s*we\'ll?\s*', '', sentence, flags=re.IGNORECASE)
                if clean_step and len(clean_step) > 10:
                    steps.append(clean_step)
            
            # Look for "next we'll" patterns
            elif re.search(r'next,?\s+we\'ll?', sentence, re.IGNORECASE):
                clean_step = re.sub(r'^next,?\s*we\'ll?\s*', '', sentence, flags=re.IGNORECASE)
                if clean_step and len(clean_step) > 10:
                    steps.append(clean_step)
            
            # Look for "then we'll" or "then it" patterns
            elif re.search(r'then,?\s+(?:we\'ll?|it)', sentence, re.IGNORECASE):
                clean_step = re.sub(r'^then,?\s+(?:we\'ll?\s*|it\s*)', '', sentence, flags=re.IGNORECASE)
                if clean_step and len(clean_step) > 10:
                    steps.append(clean_step)
            
            # Look for other action patterns
            elif any(word in sentence.lower() for word in ['find', 'click', 'set', 'create', 'open']):
                # Clean up common prefixes
                clean_step = re.sub(r'^(first|next|then|and now|now),?\s*', '', sentence, flags=re.IGNORECASE)
                if clean_step and len(clean_step) > 10:
                    steps.append(clean_step)
        
        # If we still don't have enough steps, add remaining valid sentences
        if len(steps) < 3:
            for sentence in valid_sentences:
                if sentence not in steps and len(sentence) > 15:
                    steps.append(sentence)
        
        return steps[:4]  # Limit to 4 steps for better readability
    
    def delete_company_bucket(self, company_name: str) -> bool:
        """Delete entire company bucket and all its contents"""
        try:
            # Get company-specific bucket name
            bucket_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
            
            # Check if bucket exists
            bucket = self.client.bucket(bucket_name)
            if not bucket.exists():
                logger.warning(f"⚠️ Bucket {bucket_name} does not exist")
                return True  # Consider it successful if bucket doesn't exist
            
            # Delete all blobs in the bucket first
            blobs = self.client.list_blobs(bucket_name)
            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            
            logger.info(f"🗑️ Deleted {deleted_count} files from bucket {bucket_name}")
            
            # Delete the bucket itself
            bucket.delete()
            logger.info(f"✅ Successfully deleted bucket {bucket_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete company bucket {company_name}: {e}")
            return False