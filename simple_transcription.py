#!/usr/bin/env python3
"""
Simple Transcription Service for Loom Videos
Uses OpenAI Whisper API instead of local Whisper to avoid FFmpeg issues
"""

import os
import logging
import tempfile
import requests
from typing import Dict, Optional, List

# Configure logging
logger = logging.getLogger(__name__)

class SimpleTranscriptionService:
    """Simple transcription service using OpenAI Whisper API"""
    
    def __init__(self, openai_api_key: str):
        """Initialize the transcription service"""
        self.openai_api_key = openai_api_key
        self.api_url = "https://api.openai.com/v1/audio/transcriptions"
        
    def transcribe_video(self, video_path: str) -> Optional[Dict]:
        """Transcribe video using OpenAI Whisper API"""
        try:
            logger.info(f"🎤 Transcribing video using OpenAI Whisper API: {video_path}")
            
            # Check if file exists
            if not os.path.exists(video_path):
                raise Exception(f"Video file does not exist: {video_path}")
            
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"Video file size: {file_size_mb:.1f} MB")
            
            # OpenAI has a 25MB limit for audio files
            if file_size_mb > 25:
                logger.warning(f"Video file is large ({file_size_mb:.1f}MB), may need compression")
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            # Read the video file
            with open(video_path, 'rb') as audio_file:
                files = {
                    'file': ('video.mp4', audio_file, 'video/mp4'),
                    'model': (None, 'whisper-1'),
                    'response_format': (None, 'verbose_json'),
                    'timestamp_granularities[]': (None, 'word')
                }
                
                logger.info("📡 Sending request to OpenAI Whisper API...")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    timeout=300  # 5 minute timeout
                )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ OpenAI Whisper API transcription successful")
                
                # Extract transcription data
                text = result.get('text', '')
                language = result.get('language', 'en')
                duration = result.get('duration', 0)
                
                # Create segments from words if available
                segments = []
                if 'words' in result:
                    current_segment = {
                        'text': '',
                        'start': 0,
                        'end': 0,
                        'words': []
                    }
                    
                    for word in result['words']:
                        word_text = word.get('word', '')
                        word_start = word.get('start', 0)
                        word_end = word.get('end', 0)
                        
                        # Add word to current segment
                        current_segment['words'].append({
                            'word': word_text,
                            'start': word_start,
                            'end': word_end
                        })
                        current_segment['text'] += word_text
                        current_segment['end'] = word_end
                        
                        # Create new segment every 10 words or 30 seconds
                        if (len(current_segment['words']) >= 10 or 
                            (word_end - current_segment['start']) >= 30):
                            segments.append(current_segment)
                            current_segment = {
                                'text': '',
                                'start': word_end,
                                'end': word_end,
                                'words': []
                            }
                    
                    # Add the last segment if it has content
                    if current_segment['text'].strip():
                        segments.append(current_segment)
                else:
                    # Fallback: create a single segment
                    segments = [{
                        'text': text,
                        'start': 0,
                        'end': duration,
                        'words': []
                    }]
                
                transcription_data = {
                    'transcription': text,
                    'segments': segments,
                    'language': language,
                    'word_count': len(text.split()),
                    'duration': duration,
                    'method': 'openai_whisper_api'
                }
                
                logger.info(f"✅ Transcription completed: {len(text)} characters, {len(segments)} segments")
                return transcription_data
                
            else:
                logger.error(f"❌ OpenAI API request failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None

def test_simple_transcription():
    """Test the simple transcription service"""
    try:
        logger.info("🧪 Testing Simple Transcription Service...")
        
        # Get API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("❌ OPENAI_API_KEY not found")
            return False
        
        # Initialize service
        service = SimpleTranscriptionService(openai_api_key)
        logger.info("✅ Simple Transcription Service initialized")
        
        # Test with a small audio file (if available)
        # For now, just test initialization
        logger.info("✅ Simple Transcription Service test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple transcription test failed: {e}")
        return False

if __name__ == "__main__":
    test_simple_transcription()
