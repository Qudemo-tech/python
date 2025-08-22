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

def process_video(video_url: str, company_name: str, bucket_name: Optional[str] = None, 
                 source: Optional[str] = None, meeting_link: Optional[str] = None, qudemo_id: Optional[str] = None):
    """Process a video URL and store in Pinecone with semantic chunking"""
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
            result = process_video_with_semantic_chunking(video_url, company_name, qudemo_id)
        else:
            # Use Gemini processor for other video types
            if not gemini_processor:
                raise Exception("Gemini processor not initialized")
            
            result = process_video_with_semantic_chunking(video_url, company_name, qudemo_id)
        
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
            # Download and transcribe video
            transcription_data = loom_processor._download_and_transcribe(video_url)
            if not transcription_data:
                raise Exception("Failed to transcribe video with Loom processor")
            
            transcription = transcription_data.get('transcription', '')
            segments = transcription_data.get('segments', [])
            
            logger.info(f"Loom transcription successful: {len(transcription)} chars, {len(segments)} segments")
            
            # Create semantic chunks from transcription with qudemo metadata
            return _create_semantic_chunks_from_transcription(
                transcription=transcription,
                segments=segments,
                company_name=company_name,
                video_url=video_url,
                qudemo_id=qudemo_id,
                video_type='loom'
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
            
            logger.info(f"Gemini transcription successful: {len(transcription)} chars, {len(segments)} segments")
            
            # Create semantic chunks from transcription with qudemo metadata
            return _create_semantic_chunks_from_transcription(
                transcription=transcription,
                segments=segments,
                company_name=company_name,
                video_url=video_url,
                qudemo_id=qudemo_id,
                video_type='youtube'
            )
        
    except Exception as e:
        logger.error(f"❌ Video processing with semantic chunking failed: {e}")
        raise e

def _create_semantic_chunks_from_transcription(transcription: str, segments: List[Dict], 
                                              company_name: str, video_url: str, 
                                              qudemo_id: str = None, video_type: str = 'unknown') -> Dict:
    """Create semantic chunks from transcription with qudemo-specific metadata"""
    try:
        logger.info(f"Creating semantic chunks for {company_name} with qudemo_id: {qudemo_id}")
        
        if not transcription:
            raise Exception("No transcription provided")
        
        # Initialize Pinecone
        from openai import OpenAI
        
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not pinecone_api_key or not openai_api_key:
            raise Exception("Pinecone or OpenAI API keys not configured")
        
        # Initialize Pinecone with new API
        pc = Pinecone(api_key=pinecone_api_key)
        client = OpenAI(api_key=openai_api_key)
        
        # Create or get index
        index_name = f"{company_name.lower().replace(' ', '-')}-videos"
        
        try:
            # Check if index exists
            if index_name not in pc.list_indexes().names():
                # Create index if it doesn't exist
                try:
                    pc.create_index(
                        name=index_name,
                        dimension=1536,  # OpenAI ada-002 embedding dimension
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                    logger.info(f"Created new Pinecone index: {index_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create Pinecone index: {create_error}")
                    raise Exception("Failed to initialize vector database")
            
            # Get the index
            index = pc.Index(index_name)
            logger.info(f"Using existing Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to get Pinecone index {index_name}: {e}")
            raise Exception("Failed to initialize vector database")
        
        # Split transcription into semantic chunks
        chunks = _split_transcription_into_chunks(transcription, segments, video_type)
        
        # Create embeddings and store in Pinecone
        vectors_to_upsert = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding
                embedding = client.embeddings.create(
                    input=chunk['text'],
                    model="text-embedding-ada-002"
                ).data[0].embedding
                
                # Create metadata with qudemo-specific information
                metadata = {
                    'text': chunk['text'],
                    'company_name': company_name,
                    'video_url': video_url,
                    'video_type': video_type,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'timestamp_start': chunk.get('start_time', 0),
                    'timestamp_end': chunk.get('end_time', 0),
                    'source_type': 'video_transcript',
                    'created_at': datetime.now().isoformat()
                }
                
                # Add qudemo_id if provided
                if qudemo_id:
                    metadata['qudemo_id'] = qudemo_id
                
                # Create unique ID for the vector
                vector_id = f"{company_name}-{video_url}-{i}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
                
            except Exception as chunk_error:
                logger.error(f"Failed to process chunk {i}: {chunk_error}")
                continue
        
        if vectors_to_upsert:
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    index.upsert(vectors=batch)
                    logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")
                except Exception as upsert_error:
                    logger.error(f"Failed to upsert batch {i//batch_size + 1}: {upsert_error}")
        
        logger.info(f"✅ Successfully processed {len(vectors_to_upsert)} semantic chunks for {company_name}")
        
        return {
            'success': True,
            'chunks_created': len(vectors_to_upsert),
            'company_name': company_name,
            'video_url': video_url,
            'qudemo_id': qudemo_id,
            'video_type': video_type,
            'index_name': index_name
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create semantic chunks: {e}")
        raise e

def _split_transcription_into_chunks(transcription: str, segments: List[Dict], video_type: str) -> List[Dict]:
    """Split transcription into semantic chunks with timestamps"""
    try:
        chunks = []
        
        if segments and len(segments) > 0:
            # Use segments if available (more accurate timestamps)
            current_chunk = {
                'text': '',
                'start_time': segments[0].get('start', 0),
                'end_time': 0
            }
            
            for segment in segments:
                segment_text = segment.get('text', '').strip()
                segment_start = segment.get('start', 0)
                segment_end = segment.get('end', 0)
                
                # If current chunk is getting too long (>500 chars), start a new one
                if len(current_chunk['text']) + len(segment_text) > 500:
                    # Finalize current chunk
                    current_chunk['end_time'] = segment_start
                    if current_chunk['text'].strip():
                        chunks.append(current_chunk)
                    
                    # Start new chunk
                    current_chunk = {
                        'text': segment_text,
                        'start_time': segment_start,
                        'end_time': segment_end
                    }
                else:
                    # Add to current chunk
                    if current_chunk['text']:
                        current_chunk['text'] += ' ' + segment_text
                    else:
                        current_chunk['text'] = segment_text
                    current_chunk['end_time'] = segment_end
            
            # Add final chunk
            if current_chunk['text'].strip():
                chunks.append(current_chunk)
        
        else:
            # Fallback: split by sentences if no segments available
            import re
            
            sentences = re.split(r'(?<=[.!?])\s+', transcription)
            current_chunk = {
                'text': '',
                'start_time': 0,
                'end_time': 0
            }
            
            # Estimate time per sentence based on video type
            if video_type == 'youtube':
                seconds_per_sentence = 8
            elif video_type == 'loom':
                seconds_per_sentence = 6
            else:
                seconds_per_sentence = 7
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_start = i * seconds_per_sentence
                sentence_end = (i + 1) * seconds_per_sentence
                
                # If current chunk is getting too long, start a new one
                if len(current_chunk['text']) + len(sentence) > 500:
                    # Finalize current chunk
                    current_chunk['end_time'] = sentence_start
                    if current_chunk['text'].strip():
                        chunks.append(current_chunk)
                    
                    # Start new chunk
                    current_chunk = {
                        'text': sentence,
                        'start_time': sentence_start,
                        'end_time': sentence_end
                    }
                else:
                    # Add to current chunk
                    if current_chunk['text']:
                        current_chunk['text'] += ' ' + sentence
                    else:
                        current_chunk['text'] = sentence
                        current_chunk['start_time'] = sentence_start
                    current_chunk['end_time'] = sentence_end
            
            # Add final chunk
            if current_chunk['text'].strip():
                chunks.append(current_chunk)
        
        logger.info(f"Created {len(chunks)} semantic chunks from transcription")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Failed to split transcription into chunks: {e}")
        # Return single chunk as fallback
        return [{
            'text': transcription,
            'start_time': 0,
            'end_time': 0
        }]
