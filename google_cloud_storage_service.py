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
            # Check for service account JSON in environment variable first (for Render)
            service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
            if service_account_json:
                import json
                service_account_info = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(service_account_info)
                self.client = storage.Client(credentials=credentials)
                logger.info(f"✅ Google Cloud Storage client initialized using environment JSON")
            elif self.service_account_path and os.path.exists(self.service_account_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_path
                )
                self.client = storage.Client(credentials=credentials)
                logger.info(f"✅ Google Cloud Storage client initialized using service account file: {self.service_account_path}")
            else:
                # Check Render secret files directory as fallback
                render_secret_path = '/etc/secrets/service-account-key.json'
                if os.path.exists(render_secret_path):
                    credentials = service_account.Credentials.from_service_account_file(render_secret_path)
                    self.client = storage.Client(credentials=credentials)
                    logger.info(f"✅ Google Cloud Storage client initialized using Render secret file: {render_secret_path}")
                else:
                    # Use default credentials (for production)
                    self.client = storage.Client()
                    logger.info(f"✅ Google Cloud Storage client initialized using default credentials")
            
            # Initialize client only, buckets will be created per company
            logger.info(f"📦 Will create company-specific buckets as needed")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Cloud Storage: {e}")
            logger.error(f"❌ Service account path checked: {self.service_account_path}")
            logger.error(f"❌ Render secret path checked: /etc/secrets/service-account-key.json")
            raise
    
    def _get_company_bucket(self, company_name: str):
        """Get or create a bucket for a specific company"""
        try:
            # Create company-specific bucket name with proper sanitization
            safe_company_name = self._sanitize_bucket_name(company_name)
            bucket_name = f"qudemo-{safe_company_name}"
            
            # Get or create the bucket
            bucket = self.client.bucket(bucket_name)
            
            if not bucket.exists():
                # Create the bucket
                bucket = self.client.create_bucket(bucket_name)
                logger.info(f"✅ Created company bucket: {bucket_name}")
                
                # Make bucket publicly readable for all objects
                try:
                    policy = bucket.get_iam_policy(requested_policy_version=3)
                    policy.bindings.append({
                        "role": "roles/storage.objectViewer",
                        "members": {"allUsers"}
                    })
                    bucket.set_iam_policy(policy)
                    logger.info(f"🌍 Made bucket publicly readable: {bucket_name}")
                except Exception as policy_error:
                    logger.warning(f"⚠️ Could not set public access on bucket: {policy_error}")
            else:
                logger.info(f"✅ Connected to existing company bucket: {bucket_name}")
            
            return bucket
            
        except Exception as e:
            logger.error(f"❌ Failed to get/create company bucket for {company_name}: {e}")
            raise
    
    def _sanitize_bucket_name(self, company_name: str) -> str:
        """Sanitize company name to create a valid GCS bucket name"""
        try:
            import re
            import hashlib
            
            # Convert to lowercase
            safe_name = company_name.lower()
            
            # Replace problematic characters with hyphens
            # GCS bucket names cannot contain dots (.) as they're interpreted as domain names
            # Also replace other special characters that might cause issues
            safe_name = re.sub(r'[^a-z0-9-]', '-', safe_name)
            
            # Remove multiple consecutive hyphens
            safe_name = re.sub(r'-+', '-', safe_name)
            
            # Remove leading/trailing hyphens
            safe_name = safe_name.strip('-')
            
            # Ensure it's not empty
            if not safe_name:
                # Fallback: use hash of original name
                safe_name = hashlib.md5(company_name.encode()).hexdigest()[:8]
            
            # GCS bucket names must be 3-63 characters
            if len(safe_name) > 63 - len("qudemo-"):
                # Truncate and add hash suffix to ensure uniqueness
                hash_suffix = hashlib.md5(company_name.encode()).hexdigest()[:8]
                safe_name = safe_name[:63 - len("qudemo-") - len(hash_suffix) - 1] + "-" + hash_suffix
            
            # Ensure minimum length (GCS requires 3+ chars)
            if len(safe_name) < 3:
                safe_name = safe_name + "-" + hashlib.md5(company_name.encode()).hexdigest()[:6]
            
            logger.info(f"🔄 Sanitized company name: '{company_name}' -> '{safe_name}'")
            return safe_name
            
        except Exception as e:
            logger.error(f"❌ Error sanitizing bucket name for '{company_name}': {e}")
            # Fallback: use hash of original name
            import hashlib
            return hashlib.md5(company_name.encode()).hexdigest()[:12]
    
    def store_video_transcript(self, company_name: str, qudemo_id: str, 
                             transcript_data: Dict[str, Any]) -> bool:
        """Store video transcript data in Google Cloud Storage with company/qudemo structure"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: company_name/qudemo_id/transcript.json (within company bucket)
            file_path = f"{company_name}/{qudemo_id}/transcript.json"
            
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
            
            # Create file path: company_name/qudemo_id/transcript.json (within company bucket)
            file_path = f"{company_name}/{qudemo_id}/transcript.json"
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
    
    def get_faqs(self, company_name: str, qudemo_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve FAQs data from Google Cloud Storage with company/qudemo structure"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: company_name/qudemo_id/faqs_{company_name}.json
            faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
            file_path = f"{company_name}/{qudemo_id}/{faq_filename}"
            blob = bucket.blob(file_path)
            
            if not blob.exists():
                logger.warning(f"⚠️ FAQs not found: {file_path}")
                return None
            
            # Download and parse JSON
            content = blob.download_as_text()
            faqs_data = json.loads(content)
            
            # Extract the 'faqs' array from the structure
            faqs_list = faqs_data.get('faqs', []) if isinstance(faqs_data, dict) else faqs_data
            
            logger.info(f"✅ Retrieved {len(faqs_list)} FAQs for {company_name}/{qudemo_id} from GCS")
            return faqs_list
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve FAQs: {e}")
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
    
    def store_suggested_questions(self, company_name: str, qudemo_id: str, suggested_questions: List[str]) -> bool:
        """Store suggested questions in Google Cloud Storage with company/qudemo structure (legacy format)"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: qudemo_id/suggested_questions.json (within company bucket)
            file_path = f"{qudemo_id}/suggested_questions.json"
            
            # Create suggested questions data
            suggested_questions_data = {
                "suggested_questions": suggested_questions,
                "created_at": datetime.now().isoformat(),
                "qudemo_id": qudemo_id,
                "company_name": company_name
            }
            
            # Upload data
            blob = bucket.blob(file_path)
            blob.upload_from_string(
                json.dumps(suggested_questions_data, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"✅ Stored {len(suggested_questions)} suggested questions for {company_name}/{qudemo_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store suggested questions: {e}")
            return False
    
    def store_suggested_questions_with_metadata(self, company_name: str, qudemo_id: str, 
                                                video_questions: List[Dict]) -> bool:
        """Store suggested questions WITH VIDEO METADATA + CACHED ANSWERS for instant retrieval"""
        try:
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: qudemo_id/suggested_questions.json (within company bucket)
            file_path = f"{qudemo_id}/suggested_questions.json"
            
            # Create suggested questions data with video metadata + cached answers
            suggested_questions_data = {
                "version": "2.1",  # Updated version for cached answers
                "video_questions": video_questions,  # Array of {video_id, video_index, video_title, questions_with_answers}
                "created_at": datetime.now().isoformat(),
                "qudemo_id": qudemo_id,
                "company_name": company_name
            }
            
            # Upload data
            blob = bucket.blob(file_path)
            blob.upload_from_string(
                json.dumps(suggested_questions_data, indent=2),
                content_type='application/json'
            )
            
            total_questions = sum(len(vq.get('questions_with_answers', [])) for vq in video_questions)
            logger.info(f"✅ Stored {total_questions} questions WITH CACHED ANSWERS from {len(video_questions)} video(s) for {company_name}/{qudemo_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store suggested questions with metadata: {e}")
            return False
    
    def get_suggested_questions_with_metadata(self, company_name: str, qudemo_id: str) -> Optional[Dict[str, Any]]:
        """Get stored suggested questions WITH FULL METADATA (questions + answers + video info)"""
        try:
            logger.info(f"🔍 GETTING suggested questions with metadata for company: '{company_name}', qudemo: '{qudemo_id}'")
            
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            
            # Create file path: qudemo_id/suggested_questions.json (within company bucket)
            file_path = f"{qudemo_id}/suggested_questions.json"
            
            # Check if file exists
            blob = bucket.blob(file_path)
            if not blob.exists():
                logger.warning(f"⚠️ No stored suggested questions found for {company_name}/{qudemo_id}")
                return None
            
            # Download and parse the file
            content = blob.download_as_text()
            suggested_questions_data = json.loads(content)
            
            logger.info(f"✅ Retrieved suggested questions with metadata from GCS")
            return suggested_questions_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving suggested questions with metadata: {e}")
            return None
    
    def get_suggested_questions(self, company_name: str, qudemo_id: str) -> Optional[List[str]]:
        """Get stored suggested questions with INTELLIGENT SHUFFLING for multiple videos"""
        try:
            logger.info(f"🔍 GETTING suggested questions for company: '{company_name}', qudemo: '{qudemo_id}'")
            
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            logger.info(f"🔍 Using bucket: {bucket.name}")
            
            # Create file path: qudemo_id/suggested_questions.json (within company bucket)
            file_path = f"{qudemo_id}/suggested_questions.json"
            logger.info(f"🔍 Looking for file at path: {file_path}")
            
            # Check if file exists
            blob = bucket.blob(file_path)
            if not blob.exists():
                logger.warning(f"⚠️ No stored suggested questions found for {company_name}/{qudemo_id} at path: {file_path}")
                logger.info(f"🔍 Listing all blobs in bucket to debug:")
                try:
                    blobs_list = list(bucket.list_blobs(prefix=f"{qudemo_id}/", max_results=10))
                    if blobs_list:
                        logger.info(f"🔍 Found {len(blobs_list)} files with prefix '{qudemo_id}/':")
                        for b in blobs_list:
                            logger.info(f"   - {b.name}")
                    else:
                        logger.info(f"🔍 No files found with prefix '{qudemo_id}/'")
                except Exception as list_err:
                    logger.error(f"❌ Error listing blobs: {list_err}")
                return None
            
            # Download and parse the file
            content = blob.download_as_text()
            suggested_questions_data = json.loads(content)
            
            # Check version to handle both old and new formats
            version = suggested_questions_data.get('version', '1.0')
            
            if version in ['2.0', '2.1'] and 'video_questions' in suggested_questions_data:
                # NEW FORMAT: Video-specific questions with metadata (and cached answers in v2.1)
                video_questions = suggested_questions_data.get('video_questions', [])
                logger.info(f"✅ Found NEW FORMAT (v{version}) questions from {len(video_questions)} video(s)")
                
                # Apply intelligent shuffling (returns just question strings)
                shuffled_questions = self._shuffle_video_questions(video_questions)
                logger.info(f"✅ Shuffled {len(shuffled_questions)} questions for display")
                return shuffled_questions
                
            else:
                # OLD FORMAT: Flat list of questions (backward compatibility)
                questions = suggested_questions_data.get('suggested_questions', [])
                logger.info(f"✅ Retrieved {len(questions)} stored suggested questions (OLD FORMAT)")
                logger.info(f"📝 Questions: {questions}")
                return questions
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve suggested questions for {company_name}/{qudemo_id}: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None
    
    def _shuffle_video_questions(self, video_questions: List[Dict]) -> List[str]:
        """
        Intelligently shuffle questions from multiple videos.
        - If multiple videos: Interleave questions (round-robin from different videos)
        - If single video: Return questions as-is
        - Supports both v2.0 (questions array) and v2.1 (questions_with_answers array)
        """
        try:
            if not video_questions:
                return []
            
            # Extract questions (handle both v2.0 and v2.1 formats)
            def extract_questions(video_data):
                # v2.1 format: questions_with_answers = [{question, answer, ...}]
                if 'questions_with_answers' in video_data:
                    return [qa['question'] for qa in video_data['questions_with_answers']]
                # v2.0 format: questions = ["question1", "question2"]
                elif 'questions' in video_data:
                    return video_data['questions']
                return []
            
            # Single video: No shuffling needed
            if len(video_questions) == 1:
                questions = extract_questions(video_questions[0])
                logger.info(f"🎯 Single video: Returning {len(questions)} questions as-is")
                return questions
            
            # Multiple videos: Interleave questions (round-robin)
            logger.info(f"🔀 Multiple videos ({len(video_questions)}): Applying round-robin shuffling")
            
            # Get questions for each video
            all_video_questions = [extract_questions(vq) for vq in video_questions]
            max_questions = max(len(qs) for qs in all_video_questions)
            
            shuffled = []
            # Round-robin through all videos
            for i in range(max_questions):
                for video_idx, questions in enumerate(all_video_questions):
                    if i < len(questions):
                        shuffled.append(questions[i])
                        video_title = video_questions[video_idx].get('video_title', 'Unknown')
                        logger.info(f"  Added Q{i+1} from video: {video_title}")
            
            logger.info(f"✅ Shuffled result: {len(shuffled)} questions from {len(video_questions)} videos")
            return shuffled
            
        except Exception as e:
            logger.error(f"❌ Error shuffling questions: {e}")
            # Fallback: Just flatten all questions
            flat = []
            for vq in video_questions:
                if 'questions_with_answers' in vq:
                    flat.extend([qa['question'] for qa in vq['questions_with_answers']])
                elif 'questions' in vq:
                    flat.extend(vq['questions'])
            return flat
    
    def get_cached_answer_for_suggested_question(self, company_name: str, qudemo_id: str, 
                                                  question: str) -> Optional[Dict[str, Any]]:
        """Get CACHED answer for a suggested question (INSTANT - NO LLM CALL!)"""
        try:
            logger.info(f"⚡ Looking for CACHED answer for: '{question}'")
            
            # Get company-specific bucket
            bucket = self._get_company_bucket(company_name)
            file_path = f"{qudemo_id}/suggested_questions.json"
            
            # Check if file exists
            blob = bucket.blob(file_path)
            if not blob.exists():
                logger.warning(f"⚠️ No suggested questions file found")
                return None
            
            # Download and parse the file
            content = blob.download_as_text()
            suggested_questions_data = json.loads(content)
            
            # Check version
            version = suggested_questions_data.get('version', '1.0')
            
            if version == '2.1' and 'video_questions' in suggested_questions_data:
                # v2.1 format has cached answers!
                video_questions = suggested_questions_data.get('video_questions', [])
                
                # Search for the question in all videos
                for video_data in video_questions:
                    questions_with_answers = video_data.get('questions_with_answers', [])
                    for qa in questions_with_answers:
                        if qa.get('question', '').strip().lower() == question.strip().lower():
                            logger.info(f"⚡ FOUND CACHED ANSWER! (instant retrieval)")
                            return {
                                'answer': qa.get('answer', ''),
                                'timestamp': qa.get('timestamp', 0),
                                'formatted_timestamp': qa.get('formatted_timestamp', '00:00'),
                                'video_url': qa.get('video_url', ''),
                                'video_title': qa.get('video_title', 'Video'),
                                'is_cached': True  # Flag to indicate this was instant
                            }
                
                logger.info(f"⚠️ Question not found in cached answers")
                return None
            else:
                # Old format doesn't have cached answers
                logger.info(f"⚠️ Old format (v{version}) - no cached answers available")
                return None
            
        except Exception as e:
            logger.error(f"❌ Error retrieving cached answer: {e}")
            return None
    
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
    
    def search_video_transcript(self, company_name: str, qudemo_id: str, question: str) -> Optional[Dict[str, Any]]:
        """Search video transcript for a question - wrapper for search_transcript_directly"""
        try:
            logger.info(f"🎥 Searching video transcript for question: {question}")
            return self.search_transcript_directly(company_name, qudemo_id, question)
        except Exception as e:
            logger.error(f"❌ Failed to search video transcript: {e}")
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
    
    def upload_file_content(self, content: str, file_path: str, content_type: str = 'application/octet-stream') -> bool:
        """Upload file content to Google Cloud Storage"""
        try:
            # Extract company name from file path (format: company_name/qudemo_id/documents/...)
            path_parts = file_path.split('/')
            if len(path_parts) >= 1:
                company_name = path_parts[0]
                # Get company-specific bucket
                bucket = self._get_company_bucket(company_name)
            else:
                # Fallback to default bucket
                bucket = self.client.bucket(self.default_bucket_name)
            
            # Create blob and upload content
            blob = bucket.blob(file_path)
            blob.upload_from_string(content, content_type=content_type)
            
            logger.info(f"✅ Uploaded file content to GCS: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload file content to {file_path}: {e}")
            return False

    def list_files(self, prefix: str) -> List[str]:
        """List files in GCS with a given prefix"""
        try:
            # Extract company name from prefix (format: company_name/qudemo_id/documents/)
            path_parts = prefix.split('/')
            if len(path_parts) >= 1:
                company_name = path_parts[0]
                # Get company-specific bucket
                bucket = self._get_company_bucket(company_name)
            else:
                # Fallback to default bucket
                bucket = self.client.bucket(self.default_bucket_name)
            
            # List blobs with prefix
            blobs = bucket.list_blobs(prefix=prefix)
            file_paths = [blob.name for blob in blobs]
            
            logger.info(f"📁 Listed {len(file_paths)} files with prefix: {prefix}")
            return file_paths
            
        except Exception as e:
            logger.error(f"❌ Failed to list files with prefix {prefix}: {e}")
            return []

    def download_file_content(self, file_path: str) -> Optional[str]:
        """Download file content from Google Cloud Storage"""
        try:
            # Extract company name from file path (format: company_name/qudemo_id/documents/...)
            path_parts = file_path.split('/')
            if len(path_parts) >= 1:
                company_name = path_parts[0]
                # Get company-specific bucket
                bucket = self._get_company_bucket(company_name)
            else:
                # Fallback to default bucket
                bucket = self.client.bucket(self.default_bucket_name)
            
            # Get blob and download content
            blob = bucket.blob(file_path)
            content = blob.download_as_text()
            
            logger.info(f"✅ Downloaded file content from GCS: {file_path}")
            return content
            
        except Exception as e:
            logger.error(f"❌ Failed to download file content from {file_path}: {e}")
            return None

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
    
    def store_website_content(self, company_name: str, qudemo_id: str, website_data: Dict[str, Any]) -> bool:
        """Store website scraped content in GCS"""
        try:
            bucket = self._get_company_bucket(company_name)
            
            # Create website storage path
            website_id = website_data.get('website_id', f"website_{int(datetime.now().timestamp())}")
            file_path = f"{qudemo_id}/websites/{website_id}.json"
            
            # Prepare data for storage
            storage_data = {
                'website_id': website_id,
                'base_url': website_data.get('base_url'),
                'scraped_pages': website_data.get('scraped_pages', []),
                'total_pages': website_data.get('total_pages', 0),
                'scraping_status': website_data.get('scraping_status', 'unknown'),
                'errors': website_data.get('errors', []),
                'analysis': website_data.get('analysis', {}),
                'scraped_at': website_data.get('scraped_at', datetime.now().isoformat()),
                'stored_at': datetime.now().isoformat()
            }
            
            # Upload to GCS
            blob = bucket.blob(file_path)
            blob.upload_from_string(
                json.dumps(storage_data, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"✅ Stored website content: {website_id} ({storage_data['total_pages']} pages)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store website content: {e}")
            return False
    
    def get_website_content(self, company_name: str, qudemo_id: str, website_id: str = None) -> Optional[Dict[str, Any]]:
        """Get website content from GCS"""
        try:
            bucket = self._get_company_bucket(company_name)
            
            if website_id:
                # Get specific website
                file_path = f"{qudemo_id}/websites/{website_id}.json"
                blob = bucket.blob(file_path)
                
                if blob.exists():
                    content = blob.download_as_text()
                    return json.loads(content)
                else:
                    return None
            else:
                # Get all websites for the QuDemo
                prefix = f"{qudemo_id}/websites/"
                blobs = bucket.list_blobs(prefix=prefix)
                
                websites = []
                for blob in blobs:
                    if blob.name.endswith('.json'):
                        content = blob.download_as_text()
                        website_data = json.loads(content)
                        websites.append(website_data)
                
                return {
                    'websites': websites,
                    'total_websites': len(websites)
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get website content: {e}")
            return None
    
    def search_website_content(self, company_name: str, qudemo_id: str, query: str) -> List[Dict[str, Any]]:
        """Search website content for a query"""
        try:
            # Get all website content
            website_data = self.get_website_content(company_name, qudemo_id)
            if not website_data:
                logger.info(f"🌐 No website data found for {company_name}/{qudemo_id}")
                return []
            
            results = []
            query_lower = query.lower()
            
            # Handle the data structure returned by get_website_content
            if isinstance(website_data, dict) and 'websites' in website_data:
                # Multiple websites structure: {'websites': [...], 'total_websites': ...}
                websites = website_data['websites']
            elif isinstance(website_data, list):
                # Direct list of websites
                websites = website_data
            else:
                # Single website structure
                websites = [website_data]
            
            logger.info(f"🌐 Searching through {len(websites)} website(s) for query: {query}")
            
            for website in websites:
                logger.info(f"🌐 Searching website: {website.get('website_id', 'unknown')} with {len(website.get('scraped_pages', []))} pages")
                
                for page in website.get('scraped_pages', []):
                    content = page.get('content', '').lower()
                    title = page.get('title', '').lower()
                    
                    # Enhanced text matching with keyword extraction
                    query_keywords = [word for word in query_lower.split() if len(word) > 2]
                    content_matches = sum(1 for keyword in query_keywords if keyword in content)
                    title_matches = sum(1 for keyword in query_keywords if keyword in title)
                    
                    # Match if at least 2 keywords are found, or exact phrase match
                    if (content_matches >= 2 or title_matches >= 2 or 
                        query_lower in content or query_lower in title):
                        logger.info(f"🌐 Found match in page: {page.get('title', 'untitled')}")
                        logger.info(f"🌐 Content matches: {content_matches}, Title matches: {title_matches}")
                        logger.info(f"🌐 Query keywords: {query_keywords}")
                        results.append({
                            'source_type': 'website',
                            'url': page.get('url'),
                            'title': page.get('title'),
                            'content': page.get('content'),
                            'relevance_score': self._calculate_relevance_score(query, page.get('content', ''), page.get('title', '')),
                            'website_id': website.get('website_id'),
                            'base_url': website.get('base_url')
                        })
                    else:
                        logger.info(f"🌐 No match in page: {page.get('title', 'untitled')} (content: {content_matches}, title: {title_matches})")
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            logger.info(f"🌐 Website search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search website content: {e}")
            return []
    
    def get_website_count(self, company_name: str, qudemo_id: str) -> int:
        """Get count of websites processed for a QuDemo"""
        try:
            bucket = self._get_company_bucket(company_name)
            
            # List all website files for this QuDemo
            prefix = f"{qudemo_id}/websites/"
            blobs = bucket.list_blobs(prefix=prefix)
            
            count = 0
            for blob in blobs:
                if blob.name.endswith('.json'):
                    count += 1
            
            logger.info(f"📊 Website count for {company_name}/{qudemo_id}: {count}")
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to get website count: {e}")
            return 0
    
    def _calculate_relevance_score(self, query: str, content: str, title: str = "") -> float:
        """Calculate relevance score for website content with title prioritization"""
        try:
            query_lower = query.lower()
            content_lower = content.lower()
            title_lower = title.lower()
            
            query_words = set(query_lower.split())
            content_words = content_lower.split()
            title_words = title_lower.split()
            
            if not content_words:
                return 0.0
            
            # Count query word matches in content and title
            content_matches = sum(1 for word in query_words if word in content_words)
            title_matches = sum(1 for word in query_words if word in title_words)
            
            # Calculate base score
            match_ratio = content_matches / len(query_words) if query_words else 0
            content_factor = min(len(content_words) / 100, 1.0)  # Normalize content length
            
            base_score = match_ratio * content_factor
            
            # Boost score for title matches (exact title match gets highest priority)
            if title_lower == query_lower:
                # Exact title match - highest priority
                return base_score + 10.0
            elif title_matches >= len(query_words) * 0.8:  # 80% of keywords in title
                # Most keywords in title - high priority
                return base_score + 5.0
            elif title_matches >= 2:
                # Some keywords in title - medium priority
                return base_score + 2.0
            else:
                # No title matches - base score only
                return base_score
            
        except Exception as e:
            logger.warning(f"⚠️ Relevance score calculation failed: {e}")
            return 0.0