#!/usr/bin/env python3
"""
Simple Transcription Service for OpenAI Whisper API
Used by Loom Video Processor
"""

import os
import logging
import openai
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class SimpleTranscriptionService:
    """Simple transcription service using OpenAI Whisper API"""
    
    def __init__(self, openai_api_key: str):
        """Initialize the transcription service"""
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        logger.info("✅ Simple Transcription Service initialized")
    
    def transcribe_video(self, video_path: str) -> Optional[Dict]:
        """
        Transcribe video using OpenAI Whisper API
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict with transcription data or None if failed
        """
        try:
            logger.info(f"🎤 Transcribing video: {video_path}")
            
            # Check file size
            file_size = os.path.getsize(video_path)
            logger.info(f"📊 File size: {file_size / (1024*1024):.1f} MB")
            
            # Open the video file
            with open(video_path, 'rb') as video_file:
                # Call OpenAI Whisper API
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=video_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            # Extract transcription data
            transcription_text = response.text
            segments = response.segments if hasattr(response, 'segments') else []
            
            # Convert segments to our format
            formatted_segments = []
            for segment in segments:
                formatted_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })
            
            # Create result dictionary
            result = {
                'transcription': transcription_text,
                'segments': formatted_segments,
                'language': getattr(response, 'language', 'en'),
                'duration': getattr(response, 'duration', 0),
                'word_count': len(transcription_text.split()),
                'method': 'openai_whisper_api'
            }
            
            logger.info(f"✅ Transcription successful: {len(transcription_text)} chars, {len(formatted_segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            
            # Log additional error details for debugging
            if hasattr(e, 'response'):
                logger.error(f"❌ API Response: {e.response}")
            if hasattr(e, 'status_code'):
                logger.error(f"❌ Status Code: {e.status_code}")
            if hasattr(e, 'body'):
                logger.error(f"❌ Response Body: {e.body}")
            
            return None
