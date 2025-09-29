#!/usr/bin/env python3
"""
Simple Gemini Transcriber with GCS Storage
Based on user's test file - clean and simple approach
"""

import os
import logging
import requests
import json
import re
import time
from typing import Optional, Dict, List, Any
from datetime import datetime
from google_cloud_storage_service import GoogleCloudStorageService
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class SimpleGeminiTranscriber:
    """Simple Gemini transcriber that stores results in GCS"""
    
    def __init__(self, api_key: str, gcs_bucket_name: str = None):
        """
        Initialize simple transcriber
        
        Args:
            api_key: Gemini API key
            gcs_bucket_name: GCS bucket name for storage
        """
        self.api_key = api_key
        self.gcs_service = GoogleCloudStorageService(
            bucket_name=gcs_bucket_name,
            service_account_path=os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service-account-key.json')
        )
        logger.info("✅ Simple Gemini Transcriber initialized with GCS storage")
    
    def transcribe_video(
        self,
        video_url: str,
        model: str = "gemini-2.0-flash",
        mime_type: str = "video/mp4",
        timeout: int = 1800,  # Increased to 30 minutes for long video processing
    ) -> Optional[str]:
        """
        Transcribe spoken words from a video URL using Gemini.
        Returns plain text (with timestamps if the model adds them), or None on failure.
        """
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        prompt = (
            "Transcribe the spoken words from this video.\n"
            "Write continuous paragraphs with complete thoughts.\n"
            "Add timestamps at the start of each new segment in [MM:SS-MM:SS] for videos under 1 hour, "
            "or [HH:MM:SS] if longer.\n"
            "Each chunk to be maximum 30 seconds.\n"
            "No summaries or commentary—only the transcript."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"fileData": {"mimeType": mime_type, "fileUri": video_url}},
                    ]
                }
            ]
        }

        try:
            logger.info(f"🎥 Transcribing video: {video_url}")
            
            # Check if this might be a long video and warn user
            if "youtu.be" in video_url or "youtube.com" in video_url:
                logger.info("📺 YouTube video detected - processing may take longer for videos over 30 minutes")
                logger.info("⏱️ For 1-hour videos, expect 15-30 minutes of processing time")
            
            # Retry logic for timeout issues with better handling for long videos
            max_retries = 3  # Increased retries for long videos
            r = None
            
            for attempt in range(max_retries):
                try:
                    # Estimate processing time based on attempt
                    if attempt == 0:
                        time_estimate = "5-15 minutes"
                    elif attempt == 1:
                        time_estimate = "10-20 minutes"
                    else:
                        time_estimate = "15-30 minutes"
                    
                    logger.info(f"🔄 Attempt {attempt + 1}/{max_retries} - Processing video (this may take {time_estimate})...")
                    logger.info(f"📡 Making request to: {endpoint}")
                    logger.info(f"📦 Payload structure: {list(payload.keys())}")
                    
                    r = requests.post(
                        endpoint,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.api_key
                        },
                        json=payload,
                        timeout=timeout,
                    )
                    logger.info(f"✅ Request completed successfully on attempt {attempt + 1} - Status: {r.status_code}")
                    break  # Success, exit retry loop
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Timeout on attempt {attempt + 1}, retrying with longer timeout...")
                        timeout = timeout + 600  # Add 10 more minutes for retry (was 5 minutes)
                        continue
                    else:
                        logger.error(f"❌ All {max_retries} attempts timed out (video processing may be too slow for this video length)")
                        return None
                except requests.exceptions.RequestException as e:
                    logger.error(f"❌ Request failed on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"🔄 Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        return None
            
            if r is None or r.status_code != 200:
                logger.error(f"❌ API request failed with status {r.status_code if r else 'No response'}")
                if r:
                    logger.error(f"Response content: {r.text[:500]}...")
                    
                    # Provide specific guidance for common errors
                    if r.status_code == 500:
                        logger.error("💡 500 Error: This usually means the video is too long or complex for processing")
                        logger.error("💡 Try with a shorter video (under 30 minutes) or check if the video URL is accessible")
                    elif r.status_code == 400:
                        logger.error("💡 400 Error: Invalid request - check video URL format and accessibility")
                    elif r.status_code == 403:
                        logger.error("💡 403 Error: API quota exceeded or invalid API key")
                return None

            try:
                data = r.json()
                logger.info(f"📊 Response keys: {list(data.keys())}")
            except Exception as e:
                logger.error(f"❌ Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {r.text[:500]}...")
                return None
                
            cands = data.get("candidates", [])
            if not cands:
                logger.error("❌ No candidates in response")
                logger.error(f"Full response: {data}")
                return None

            parts = cands[0].get("content", {}).get("parts", [])
            if not parts or "text" not in parts[0]:
                logger.error("❌ No text content in response")
                return None

            transcript = (parts[0]["text"] or "").strip()
            if not transcript:
                logger.error("❌ Empty transcript")
                return None
                
            logger.info(f"✅ Transcription successful: {len(transcript)} characters")
            
            # Log the full transcript text for debugging
            logger.info("📄 FULL TRANSCRIPT TEXT:")
            logger.info("=" * 80)
            logger.info(transcript)
            logger.info("=" * 80)
            
            return transcript
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None
    
    def parse_timestamped_segments(self, transcript: str) -> List[Dict[str, Any]]:
        """
        Parse timestamped segments from transcript
        
        Args:
            transcript: Raw transcript with timestamps
            
        Returns:
            List of segments with timestamps and text
        """
        segments = []
        
        # Split by timestamp patterns
        # Pattern 1: [MM:SS-MM:SS] or [HH:MM:SS-HH:MM:SS]
        timestamp_pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)-(\d{1,2}:\d{2}(?::\d{2})?)\]'
        
        parts = re.split(timestamp_pattern, transcript)
        
        i = 0
        while i < len(parts) - 2:
            if re.match(timestamp_pattern, f"[{parts[i+1]}-{parts[i+2]}]"):
                start_time = self._parse_timestamp(parts[i+1])
                end_time = self._parse_timestamp(parts[i+2])
                text = parts[i+3].strip() if i+3 < len(parts) else ""
                
                if text:  # Only add segments with text
                    segments.append({
                        'start_timestamp': start_time,
                        'end_timestamp': end_time,
                        'text': text,
                        'formatted_start': parts[i+1],
                        'formatted_end': parts[i+2]
                    })
                i += 4
            else:
                i += 1
        
        # If no timestamped segments found, create one big segment
        if not segments and transcript.strip():
            segments.append({
                'start_timestamp': 0,
                'end_timestamp': 0,
                'text': transcript.strip(),
                'formatted_start': "00:00",
                'formatted_end': "00:00"
            })
        
        logger.info(f"📊 Parsed {len(segments)} timestamped segments")
        return segments
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Convert timestamp string to seconds"""
        try:
            parts = timestamp_str.split(':')
            if len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                return 0.0
        except:
            return 0.0
    
    def create_chunks_from_segments(self, segments: List[Dict[str, Any]], chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Create chunks from segments for better Q&A performance
        
        Args:
            segments: List of timestamped segments
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for i, segment in enumerate(segments):
            text = segment['text']
            start_time = segment['start_timestamp']
            end_time = segment['end_timestamp']
            
            # If segment is too long, split it
            if len(text) > chunk_size:
                words = text.split()
                current_chunk = ""
                current_start = start_time
                
                for j, word in enumerate(words):
                    if len(current_chunk + " " + word) > chunk_size and current_chunk:
                        # Save current chunk
                        chunks.append({
                            'id': f"chunk_{len(chunks)}",
                            'text': current_chunk.strip(),
                            'start_timestamp': current_start,
                            'end_timestamp': start_time + (j / len(words)) * (end_time - start_time),
                            'segment_id': i,
                            'metadata': {
                                'segment_start': segment['formatted_start'],
                                'segment_end': segment['formatted_end'],
                                'chunk_type': 'transcript'
                            }
                        })
                        current_chunk = word
                        current_start = start_time + (j / len(words)) * (end_time - start_time)
                    else:
                        current_chunk += " " + word if current_chunk else word
                
                # Add remaining text as final chunk
                if current_chunk.strip():
                    chunks.append({
                        'id': f"chunk_{len(chunks)}",
                        'text': current_chunk.strip(),
                        'start_timestamp': current_start,
                        'end_timestamp': end_time,
                        'segment_id': i,
                        'metadata': {
                            'segment_start': segment['formatted_start'],
                            'segment_end': segment['formatted_end'],
                            'chunk_type': 'transcript'
                        }
                    })
            else:
                # Segment is small enough, use as single chunk
                chunks.append({
                    'id': f"chunk_{len(chunks)}",
                    'text': text,
                    'start_timestamp': start_time,
                    'end_timestamp': end_time,
                    'segment_id': i,
                    'metadata': {
                        'segment_start': segment['formatted_start'],
                        'segment_end': segment['formatted_end'],
                        'chunk_type': 'transcript'
                    }
                })
        
        logger.info(f"📦 Created {len(chunks)} chunks from {len(segments)} segments")
        return chunks
    
    async def process_video_with_gcs_storage(
        self, 
        video_url: str, 
        company_name: str, 
        qudemo_id: str
    ) -> Dict[str, Any]:
        """
        Process video and store in GCS
        
        Args:
            video_url: Video URL to process
            company_name: Company name for bucket organization
            qudemo_id: QuDemo ID for folder organization
            
        Returns:
            Processing result dictionary
        """
        try:
            logger.info(f"🎬 Processing video for {company_name}/{qudemo_id}: {video_url}")
            
            # Step 1: Transcribe video
            transcript = self.transcribe_video(video_url)
            if not transcript:
                return {
                    'success': False,
                    'error': 'Failed to transcribe video',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            # Step 2: Parse timestamped segments
            segments = self.parse_timestamped_segments(transcript)
            
            # Log parsed segments
            logger.info(f"📊 PARSED SEGMENTS ({len(segments)} total):")
            for i, segment in enumerate(segments[:5]):  # Show first 5 segments
                formatted_timestamp = f"{segment.get('formatted_start', '00:00')}-{segment.get('formatted_end', '00:00')}"
                logger.info(f"  Segment {i+1} [{formatted_timestamp}]: {segment['text'][:100]}...")
            if len(segments) > 5:
                logger.info(f"  ... and {len(segments) - 5} more segments")
            
            # Step 3: Create chunks
            chunks = self.create_chunks_from_segments(segments)
            
            # Log created chunks
            logger.info(f"📦 CREATED CHUNKS ({len(chunks)} total):")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                metadata = chunk.get('metadata', {})
                formatted_timestamp = f"{metadata.get('segment_start', '00:00')}-{metadata.get('segment_end', '00:00')}"
                logger.info(f"  Chunk {i+1} [{formatted_timestamp}]: {chunk['text'][:100]}...")
            if len(chunks) > 3:
                logger.info(f"  ... and {len(chunks) - 3} more chunks")
            
            # Step 4: Prepare data for GCS storage
            transcript_data = {
                'video_url': video_url,
                'video_title': f'Video for {company_name}',
                'transcript': transcript,
                'timestamps': segments,  # Keep original segments
                'segments': segments,   # Alias for compatibility
                'chunks': chunks,
                'topics': [],  # No topic analysis in simple version
                'metadata': {
                    'word_count': len(transcript.split()),
                    'language': 'en',
                    'method': 'simple_gemini_transcriber',
                    'processed_at': datetime.now().isoformat(),
                    'total_segments': len(segments),
                    'total_chunks': len(chunks)
                }
            }
            
            # Step 5: Store in GCS
            storage_success = self.gcs_service.store_video_transcript(
                company_name=company_name,
                qudemo_id=qudemo_id,
                transcript_data=transcript_data
            )
            
            if not storage_success:
                return {
                    'success': False,
                    'error': 'Failed to store in Google Cloud Storage',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            logger.info(f"✅ Video processed and stored successfully: {len(chunks)} chunks")
            
            return {
                'success': True,
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'chunks_created': len(chunks),
                'segments_created': len(segments),
                'transcript_length': len(transcript),
                'method': 'simple_gemini_transcriber'
            }
            
        except Exception as e:
            logger.error(f"❌ Video processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }

# Example usage function
def test_transcriber():
    """Test the simple transcriber"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return
    
    transcriber = SimpleGeminiTranscriber(api_key)
    
    # Test with your example video
    video_url = "https://www.youtube.com/watch?v=B-nEYsyRlYo"
    
    print(f"🎥 Testing transcription for: {video_url}")
    transcript = transcriber.transcribe_video(video_url)
    
    if transcript:
        print("✅ Transcription successful!")
        print(f"📏 Length: {len(transcript)} characters")
        print(f"📄 Preview: {transcript[:200]}...")
        
        # Parse segments
        segments = transcriber.parse_timestamped_segments(transcript)
        print(f"📊 Segments found: {len(segments)}")
        
        # Create chunks
        chunks = transcriber.create_chunks_from_segments(segments)
        print(f"📦 Chunks created: {len(chunks)}")
    else:
        print("❌ Transcription failed")

if __name__ == "__main__":
    test_transcriber()
