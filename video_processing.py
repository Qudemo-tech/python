"""
Video Processing Module
Handles video processing, transcription, and storage functionalities
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional
from fastapi import HTTPException, Request
from pydantic import BaseModel

# Import processors
from loom_processor import LoomVideoProcessor
from gemini_transcription import GeminiTranscriptionProcessor

# Configure logging
logger = logging.getLogger(__name__)

# Global variables
gemini_processor = None
loom_processor = None
VIDEO_URL_MAPPING = {}

class ProcessVideoRequest(BaseModel, extra='allow'):
    video_url: str
    company_name: str
    bucket_name: Optional[str] = None
    source: Optional[str] = None
    meeting_link: Optional[str] = None
    is_loom: bool = True

def initialize_processors():
    """Initialize video processing processors"""
    global gemini_processor, loom_processor
    
    try:
        # Get API keys
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not gemini_api_key:
            logger.error("❌ GEMINI_API_KEY not found")
            return False
        if not pinecone_api_key:
            logger.error("❌ PINECONE_API_KEY not found")
            return False
        if not openai_api_key:
            logger.error("❌ OPENAI_API_KEY not found")
            return False
        
        # Initialize Gemini processor
        logger.info("🔧 Initializing Gemini Transcription Processor...")
        gemini_processor = GeminiTranscriptionProcessor(
            gemini_api_key=gemini_api_key,
            pinecone_api_key=pinecone_api_key,
            openai_api_key=openai_api_key
        )
        logger.info("✅ Gemini processor initialized")
        
        # Initialize Loom processor
        logger.info("🔧 Initializing Loom Video Processor...")
        loom_processor = LoomVideoProcessor(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key
        )
        logger.info("✅ Loom processor initialized")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize processors: {e}")
        return False

def process_video(video_url: str, company_name: str, bucket_name: Optional[str] = None, 
                 source: Optional[str] = None, meeting_link: Optional[str] = None):
    """Process a video URL and store in Pinecone with memory management"""
    try:
        logger.info(f"🎬 Processing video: {video_url}")
        logger.info(f"🏢 Company: {company_name}")
        
        # Check memory before processing
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"💾 Memory before processing: {memory_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"⚠️ Could not check memory: {e}")
            memory_mb = 0
        
        # Determine if it's a Loom video
        is_loom = "loom.com" in video_url.lower()
        
        if is_loom:
            if not loom_processor:
                raise Exception("Loom processor not initialized")
            
            # Choose processing mode based on memory
            if memory_mb > 1500:  # High memory usage
                logger.warning(f"⚠️ High memory usage ({memory_mb:.1f}MB), using lightweight mode")
                logger.info("🎬 Processing with Loom processor (LIGHTWEIGHT MODE)...")
                result = loom_processor.process_video_lightweight(
                    video_url=video_url,
                    company_name=company_name
                )
            else:
                logger.info("🎬 Processing with Loom processor (STANDARD MODE)...")
                result = loom_processor.process_video(
                    video_url=video_url,
                    company_name=company_name
                )
        else:
            if not gemini_processor:
                raise Exception("Gemini processor not initialized")
            
            logger.info("🎬 Processing with Gemini processor...")
            result = gemini_processor.process_video(
                video_url=video_url,
                company_name=company_name
            )
        
        logger.info(f"✅ Video processing completed successfully")
        return {
            "success": True,
            "message": "Video processed successfully",
            "company_name": company_name,
            "video_url": video_url,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"❌ Video processing failed: {e}")
        
        # Provide more specific error messages
        error_message = str(e)
        if "API request failed" in error_message:
            error_message = "Transcription service temporarily unavailable. Please try again in a few minutes."
        elif "Failed to extract transcription" in error_message:
            error_message = "Unable to extract video transcription. The video may not have captions or may be private."
        elif "timeout" in error_message.lower():
            error_message = "Request timed out. Please try again."
        
        return {
            "success": False,
            "error": error_message,
            "company_name": company_name,
            "video_url": video_url,
            "details": {
                "original_error": str(e),
                "suggestion": "Try again in a few minutes or use a different video"
            }
        }

def add_video_url_mapping(local_filename: str, original_url: str):
    """Add video URL mapping"""
    global VIDEO_URL_MAPPING
    filename = os.path.basename(local_filename)
    VIDEO_URL_MAPPING[filename] = original_url
    logger.info(f"📝 Added video mapping: {filename} -> {original_url}")
    
    # Also upsert to Supabase
    try:
        from supabase import create_client, Client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if supabase_url and supabase_key:
            supabase: Client = create_client(supabase_url, supabase_key)
            
            # For Loom videos, store the original share URL as the video_url for playback
            video_url_for_playback = original_url
            
            # Upsert by video_name
            supabase.table('videos').upsert({
                'video_name': filename,
                'video_url': video_url_for_playback
            }, on_conflict=['video_name']).execute()
            logger.info(f"📝 Upserted video mapping to Supabase: {filename} -> {video_url_for_playback}")
    except Exception as e:
        logger.error(f"❌ Failed to upsert video mapping to Supabase: {e}")

def get_original_video_url(local_filename: str) -> Optional[str]:
    """Get original video URL from local filename"""
    global VIDEO_URL_MAPPING
    if '[' in local_filename:
        filename = local_filename.split('[')[0].strip()
    else:
        filename = local_filename

    original_url = VIDEO_URL_MAPPING.get(filename)
    if not original_url:
        # Try to refresh mapping from Supabase
        logger.info(f"🔄 Refreshing video mappings from Supabase for: {filename}")
        VIDEO_URL_MAPPING.update(fetch_video_urls_from_supabase())
        original_url = VIDEO_URL_MAPPING.get(filename)
    if original_url:
        logger.info(f"🔗 Found video mapping: {filename} -> {original_url}")
    else:
        logger.warning(f"⚠️ No video mapping found for: {filename}")
    return original_url

def fetch_video_urls_from_supabase() -> Dict[str, str]:
    """Fetch video URL mappings from Supabase"""
    try:
        from supabase import create_client, Client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("⚠️ Supabase credentials not configured")
            return {}
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        response = supabase.table('videos').select('video_name,video_url').execute()
        mappings = {}
        
        for row in response.data:
            mappings[row['video_name']] = row['video_url']
        
        logger.info(f"📝 Fetched {len(mappings)} video mappings from Supabase")
        return mappings
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch video mappings from Supabase: {e}")
        return {}

def initialize_existing_mappings():
    """Initialize video URL mappings from Supabase"""
    global VIDEO_URL_MAPPING
    VIDEO_URL_MAPPING.update(fetch_video_urls_from_supabase())
    logger.info(f"📝 Initialized {len(VIDEO_URL_MAPPING)} video mappings")

def get_video_mappings() -> Dict[str, str]:
    """Get current video mappings"""
    return VIDEO_URL_MAPPING.copy()

def get_processors_status() -> Dict[str, str]:
    """Get status of video processors"""
    return {
        "gemini": "✅ Available" if gemini_processor else "❌ Not available",
        "loom": "✅ Available" if loom_processor else "❌ Not available"
    }
