"""
Video Processing Module
Handles video processing, transcription, and storage functionalities
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional, List
from fastapi import HTTPException, Request
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec

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
        
        logger.info("Initializing Gemini Transcription Processor...")
        
        try:
            gemini_processor = GeminiTranscriptionProcessor(
                gemini_api_key=os.getenv('GEMINI_API_KEY'),
                pinecone_api_key=os.getenv('PINECONE_API_KEY'),
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            logger.info("Gemini processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini processor: {e}")
            gemini_processor = None
        
        logger.info("Initializing Loom Video Processor...")
        
        try:
            loom_processor = LoomVideoProcessor(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                pinecone_api_key=os.getenv('PINECONE_API_KEY')
            )
            logger.info("Loom processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Loom processor: {e}")
            loom_processor = None
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize processors: {e}")
        return False

def process_video(video_url: str, company_name: str, qudemo_id: str = None, bucket_name: Optional[str] = None, 
                 source: Optional[str] = None, meeting_link: Optional[str] = None):
    """Process a video URL and store in Pinecone with semantic chunking for specific qudemo"""
    try:
        logger.info(f"🎬 Processing video: {video_url}")
        logger.info(f"🏢 Company: {company_name}")
        logger.info(f"🎯 Qudemo ID: {qudemo_id}")
        
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
            
            # Process with semantic chunking
            return process_video_with_semantic_chunking(video_url, company_name, qudemo_id)
        else:
            # Use Gemini processor for other video types
            if not gemini_processor:
                raise Exception("Gemini processor not initialized")
            
            return process_video_with_semantic_chunking(video_url, company_name, qudemo_id)
        
        # Post-processing memory cleanup
        if is_loom and loom_processor:
            logger.info("🧹 Post-processing memory cleanup...")
            loom_processor.cleanup_memory(preserve_whisper=False)
        
        # Check if the result is valid (not None)
        if result is None:
            logger.error("❌ Video processing failed - result is None")
            return {
                "success": False,
                "error": "Video processing failed - no result returned",
                "company_name": company_name,
                "qudemo_id": qudemo_id,
                "video_url": video_url,
                "details": {
                    "suggestion": "The video may not have audio or may be corrupted"
                }
            }
        
        logger.info(f"✅ Video processing completed successfully")
        return {
            "success": True,
            "message": "Video processed successfully",
            "company_name": company_name,
            "qudemo_id": qudemo_id,
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
            "qudemo_id": qudemo_id,
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
        
        logger.info(f"Fetched {len(mappings)} video mappings from Supabase")
        return mappings
        
    except Exception as e:
        logger.error(f"Failed to fetch video mappings from Supabase: {e}")
        return {}

def initialize_existing_mappings():
    """Initialize video URL mappings from Supabase"""
    global VIDEO_URL_MAPPING
    VIDEO_URL_MAPPING.update(fetch_video_urls_from_supabase())
    logger.info(f"Initialized {len(VIDEO_URL_MAPPING)} video mappings")

def get_video_mappings() -> Dict[str, str]:
    """Get current video mappings"""
    return VIDEO_URL_MAPPING.copy()

def get_processors_status() -> Dict[str, str]:
    """Get status of video processors"""
    return {
        "gemini": "Available" if gemini_processor else "Not available",
        "loom": "Available" if loom_processor else "Not available"
    }

def process_video_with_semantic_chunking(video_url: str, company_name: str, qudemo_id: str = None) -> Dict:
    """Process video with semantic chunking for enhanced retrieval with qudemo isolation"""
    try:
        logger.info(f"Processing video with semantic chunking: {video_url}")
        logger.info(f"Company: {company_name}, Qudemo ID: {qudemo_id}")
        
        # Determine processor based on video type
        is_loom = "loom.com" in video_url.lower()
        
        if is_loom:
            if not loom_processor:
                raise Exception("Loom processor not initialized")
            
            logger.info(f"Using Loom processor for: {video_url}")
            # Use the public process_video method to get enhanced segments
            transcription_data = loom_processor.process_video(video_url, company_name, qudemo_id)
            if not transcription_data:
                raise Exception("Failed to transcribe video with Loom processor")
            
            transcription = transcription_data.get('transcription', '')
            segments = transcription_data.get('segments', [])
            
            logger.info(f"Loom transcription successful: {len(transcription)} chars, {len(segments)} segments")
            
            # Create semantic chunks from transcription
            return _create_semantic_chunks_from_transcription(
                transcription=transcription,
                segments=segments,
                company_name=company_name,
                qudemo_id=qudemo_id,
                video_url=video_url
            )
        else:
            if not gemini_processor:
                raise Exception("Gemini processor not initialized")
            
            logger.info(f"Using Gemini processor for: {video_url}")
            # Use the correct method for Gemini processor
            transcription_data = gemini_processor.extract_transcription_with_gemini(video_url)
            if not transcription_data:
                raise Exception("Failed to transcribe video with Gemini processor")
            
            transcription = transcription_data.get('transcription', '')
            # For Gemini, segments might be in a different format or not available
            segments = transcription_data.get('segments', [])
            
            # If no segments, create a basic segment structure
            if not segments and transcription:
                segments = [{'text': transcription, 'start': 0, 'end': 0}]
            
            logger.info(f"Gemini transcription successful: {len(transcription)} chars, {len(segments)} segments")
            
            # Create semantic chunks from transcription
            return _create_semantic_chunks_from_transcription(
                transcription=transcription,
                segments=segments,
                company_name=company_name,
                qudemo_id=qudemo_id,
                video_url=video_url
            )
            
    except Exception as e:
        logger.error(f"Error in semantic chunking video processing: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "company_name": company_name,
            "qudemo_id": qudemo_id,
            "video_url": video_url
        }

def _create_semantic_chunks_from_transcription(transcription: str, segments: list, 
                                             company_name: str, qudemo_id: str = None, video_url: str = None) -> Dict:
    """Create semantic chunks from video transcription and store in Pinecone with qudemo isolation"""
    try:
        logger.info(f"🔧 Creating semantic chunks from transcription for {company_name} qudemo {qudemo_id}")
        
        # Initialize knowledge integrator for semantic chunking
        from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator
        
        integrator = EnhancedKnowledgeIntegrator(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            pinecone_index=os.getenv('PINECONE_INDEX')
        )
        
        # Prepare source information for video data
        source_info = {
            'source': 'video',
            'url': video_url,
            'title': f'Video Transcription - {company_name}',
            'platform': 'loom' if 'loom.com' in video_url else 'other',
            'transcription_length': len(transcription),
            'segment_count': len(segments),
            'processed_at': datetime.now().isoformat()
        }
        
        # Create multiple time-based chunks from transcription
        chunks = []
        
        # Always create timestamped chunks, regardless of Whisper segments
        logger.info(f"🔧 Creating timestamped chunks for video processing")
        
        # Determine chunk duration and number of chunks
        chunk_duration = 30  # 30 seconds per chunk
        transcription_length = len(transcription)
        
        # Estimate total video duration based on transcription length
        # For Loom videos, use a more realistic estimate
        if 'loom.com' in video_url:
            # Loom videos: estimate 2-3 words per second (more realistic for screen recordings)
            words_per_second = 2.5
            estimated_duration = max(60, len(transcription.split()) / words_per_second)
        else:
            # YouTube videos: estimate 2.5 words per second
            estimated_duration = max(60, transcription_length / 2.5)
        
        # Ensure we create at least 3 chunks for meaningful segmentation
        min_chunks = 3
        num_chunks = max(min_chunks, int(estimated_duration / chunk_duration))
        
        logger.info(f"🔧 Estimated duration: {estimated_duration:.1f}s, creating {num_chunks} chunks")
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, estimated_duration)
            
            # Extract text for this time segment
            text_start = int((start_time / estimated_duration) * transcription_length)
            text_end = int((end_time / estimated_duration) * transcription_length)
            chunk_text = transcription[text_start:text_end].strip()
            
            # Ensure each chunk has meaningful content
            if chunk_text and len(chunk_text) > 10:  # At least 10 characters
                chunk_data = {
                    'text': chunk_text,
                    'full_context': chunk_text,
                    'source': source_info.get('source', 'video'),
                    'title': source_info.get('title', f'Video Transcription - {company_name}'),
                    'url': source_info.get('url', video_url),
                    'processed_at': source_info.get('processed_at', ''),
                    'start_timestamp': start_time,
                    'end_timestamp': end_time,
                    'chunk_index': i,
                    'total_chunks': num_chunks
                }
                chunks.append(chunk_data)
                logger.info(f"🔧 Created chunk {i+1}/{num_chunks}: {start_time}s → {end_time}s ({len(chunk_text)} chars)")
            else:
                logger.info(f"🔧 Skipping chunk {i+1}: insufficient content ({len(chunk_text)} chars)")
        
        # If we still don't have enough chunks, create at least one
        if len(chunks) == 0:
            logger.warning("🔧 No chunks created, creating fallback chunk")
            chunk_data = {
                'text': transcription,
                'full_context': transcription,
                'source': source_info.get('source', 'video'),
                'title': source_info.get('title', f'Video Transcription - {company_name}'),
                'url': source_info.get('url', video_url),
                'processed_at': source_info.get('processed_at', ''),
                'start_timestamp': 0,
                'end_timestamp': estimated_duration,
                'chunk_index': 0,
                'total_chunks': 1
            }
            chunks.append(chunk_data)
        
        logger.info(f"🔧 Created {len(chunks)} chunks with timestamps")
        
        stored_result = integrator.store_semantic_chunks(
            chunks=chunks,
            company_name=company_name,
            qudemo_id=qudemo_id
        )
        
        if stored_result.get('success', False):
            chunks_stored = stored_result.get('chunks_stored', 0)
            logger.info(f"✅ Successfully stored {chunks_stored} semantic chunks for video")
        else:
            chunks_stored = 0
            logger.error(f"❌ Failed to store semantic chunks: {stored_result.get('error', 'Unknown error')}")
        
        return {
            "success": True,
            "message": "Video processed with semantic chunking",
            "company_name": company_name,
            "qudemo_id": qudemo_id,
            "video_url": video_url,
            "result": {
                "chunks_stored": chunks_stored,
                "transcription_length": len(transcription),
                "segment_count": len(segments),
                "processing_method": "semantic_chunking"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating semantic chunks: {e}")
        return {
            "success": False,
            "error": str(e),
            "company_name": company_name,
            "qudemo_id": qudemo_id,
            "video_url": video_url
        }
