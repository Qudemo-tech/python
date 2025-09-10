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

async def process_video(video_url: str, company_name: str, qudemo_id: str = None, bucket_name: Optional[str] = None, 
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
            
            # For Loom videos, use the Loom processor directly to avoid duplicate processing
            logger.info(f"🎯 Using Loom processor directly for: {video_url}")
            result = loom_processor.process_video(video_url, company_name, qudemo_id)
            
            if result and result.get('success'):
                logger.info(f"✅ Loom video processed successfully: {result.get('chunks_created', 0)} chunks created")
                return {
                    "success": True,
                    "message": "Loom video processed successfully",
                    "company_name": company_name,
                    "qudemo_id": qudemo_id,
                    "video_url": video_url,
                    "result": result
                }
            else:
                logger.error(f"❌ Loom video processing failed: {result}")
                return {
                    "success": False,
                    "error": result.get('error', 'Unknown error') if result else 'No result returned',
                    "company_name": company_name,
                    "qudemo_id": qudemo_id,
                    "video_url": video_url
                }
        else:
            # Use Gemini processor for other video types
            if not gemini_processor:
                raise Exception("Gemini processor not initialized")
            
            logger.info(f"🎯 Using Gemini processor for: {video_url}")
            
            # Use Gemini processor directly to avoid unified chunking issues
            result = await gemini_processor.process_video_with_qudemo(video_url, company_name, qudemo_id)
            
            if result and result.get('success'):
                logger.info(f"✅ Gemini video processed successfully")
                return {
                    "success": True,
                    "message": "Video processed successfully",
                    "company_name": company_name,
                    "qudemo_id": qudemo_id,
                    "video_url": video_url,
                    "result": result
                }
            else:
                logger.error(f"❌ Gemini video processing failed: {result}")
                return {
                    "success": False,
                    "error": result.get('error', 'Unknown error') if result else 'No result returned',
                    "company_name": company_name,
                    "qudemo_id": qudemo_id,
                    "video_url": video_url
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

async def process_video_with_semantic_chunking(video_url: str, company_name: str, qudemo_id: str = None) -> Dict:
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
            
            # Check if video was already processed by looking for existing data
            try:
                from enhanced_knowledge_integration import get_enhanced_knowledge_integration
                integrator = get_enhanced_knowledge_integration()
                
                # Check if we already have data for this video
                existing_data = integrator.get_knowledge_summary(company_name, qudemo_id)
                if existing_data and existing_data.get('success') and existing_data.get('data', {}).get('video_sources'):
                    logger.info(f"✅ Video already processed for {company_name} qudemo {qudemo_id}, skipping duplicate processing")
                    return {
                        "success": True,
                        "message": "Video already processed successfully",
                        "company_name": company_name,
                        "qudemo_id": qudemo_id,
                        "video_url": video_url,
                        "details": "Video was already processed by Loom processor"
                    }
            except Exception as e:
                logger.warning(f"Could not check existing data: {e}")
            
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
            # Use the correct async method for Gemini processor
            transcription_data = await gemini_processor.process_video_with_qudemo(video_url, company_name, qudemo_id)
            
            if not transcription_data:
                raise Exception("Failed to transcribe video with Gemini processor")
            
            # Check if Gemini processor already successfully processed and stored chunks
            chunks_stored = transcription_data.get('chunks_stored', 0) or transcription_data.get('chunks_created', 0)
            if transcription_data.get('success') and chunks_stored > 0:
                logger.info(f"✅ Gemini processor already successfully processed video: {chunks_stored} chunks stored")
                return {
                    'success': True,
                    'message': f'Successfully processed video with {chunks_stored} chunks',
                    'chunks_created': chunks_stored,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'video_url': video_url
                }
            
            # If Gemini processor didn't store chunks, try to process transcription data
            transcription = transcription_data.get('transcription', '')
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
    """Create semantic chunks from video transcription using unified chunking strategy"""
    try:
        logger.info(f"🔧 Creating semantic chunks from transcription for {company_name} qudemo {qudemo_id}")
        
        # Initialize unified chunking processor
        from unified_chunking_utils import UnifiedChunkingProcessor
        chunking_processor = UnifiedChunkingProcessor()
        
        # Create chunks using unified strategy
        chunks = chunking_processor.create_timestamped_chunks(
            transcription, video_url, company_name, qudemo_id
        )
        
        # Apply semantic boundary detection
        chunks = chunking_processor.detect_semantic_boundaries(chunks)
        
        logger.info(f"🔧 Created {len(chunks)} unified chunks")
        
        # Initialize knowledge integrator for storage
        from enhanced_knowledge_integration import get_enhanced_knowledge_integration
        
        integrator = get_enhanced_knowledge_integration()
        
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
        
        # Store chunks in Pinecone
        if chunks:
            logger.info(f"🔧 Storing {len(chunks)} unified chunks in Pinecone")
            
            # Store chunks using knowledge integrator
            try:
                result = integrator.store_knowledge_data(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    chunks=chunks,
                    source_info=source_info
                )
                
                if result.get('success'):
                    logger.info(f"✅ Successfully stored {len(chunks)} chunks in Pinecone")
                    return {
                        'success': True,
                        'message': f'Successfully processed video with {len(chunks)} chunks',
                        'chunks_created': len(chunks),
                        'company_name': company_name,
                        'qudemo_id': qudemo_id,
                        'video_url': video_url
                    }
                else:
                    logger.error(f"❌ Failed to store chunks: {result.get('error', 'Unknown error')}")
                    return {
                        'success': False,
                        'error': f"Failed to store chunks: {result.get('error', 'Unknown error')}",
                        'company_name': company_name,
                        'qudemo_id': qudemo_id,
                        'video_url': video_url
                    }
            except Exception as e:
                logger.error(f"❌ Error storing chunks: {e}")
                return {
                    'success': False,
                    'error': f"Error storing chunks: {str(e)}",
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'video_url': video_url
                }
        else:
            logger.warning("🔧 No chunks created from transcription")
            return {
                'success': False,
                'error': 'No chunks created from transcription',
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'video_url': video_url
            }
            
    except Exception as e:
        logger.error(f"❌ Error in semantic chunking: {e}")
        return {
            'success': False,
            'error': f"Error in semantic chunking: {str(e)}",
            'company_name': company_name,
            'qudemo_id': qudemo_id,
            'video_url': video_url
        }
