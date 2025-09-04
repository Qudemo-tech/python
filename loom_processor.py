#!/usr/bin/env python3
"""
Loom Video Processor
Handles Loom video processing with transcription and vector storage
Optimized for 8GB RAM
"""

import os
import logging
import time
import json
import gc
import psutil
from typing import Dict, Optional, List
import requests
import tempfile
import whisper
from pinecone import Pinecone, ServerlessSpec
import openai
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoomVideoProcessor:
    def __init__(self, openai_api_key: str, pinecone_api_key: str):
        """Initialize Loom Video Processor"""
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        
        # Configure OpenAI for embeddings
        openai.api_key = openai_api_key
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.default_index_name = os.getenv("PINECONE_INDEX", "qudemo-index")
        
        # Initialize Whisper model (lazy loading)
        self._whisper_model = None
        
        # Memory management for 8GB RAM
        self.memory_threshold = 4000  # MB
        self.warning_memory_threshold = 3000   # MB
        
        logger.info("Initializing Loom Video Processor (8GB RAM Optimized)...")
    
    def check_memory_usage(self) -> float:
        """Check current memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.memory_threshold:
                logger.warning(f"⚠️ High memory usage: {memory_mb:.1f} MB")
            else:
                logger.info(f"✅ Memory usage: {memory_mb:.1f} MB (Safe)")
            
            return memory_mb
        except Exception as e:
            logger.error(f"Failed to check memory: {e}")
            return 0.0
    
    def cleanup_memory(self):
        """Clean up memory"""
        try:
            logger.info("🧹 Performing memory cleanup...")
            gc.collect()
            
            # Unload Whisper model to free memory
            if self._whisper_model:
                logger.info("🗑️ Unloading Whisper model to free memory")
                del self._whisper_model
                self._whisper_model = None
                gc.collect()
            
            memory_after = self.check_memory_usage()
            logger.info(f"🧹 Memory cleanup completed: {memory_after:.1f}MB")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            if self._whisper_model:
                del self._whisper_model
                self._whisper_model = None
    
    def get_whisper_model(self):
        """Get or load Whisper model"""
        if self._whisper_model is None:
            # Check memory before loading
            memory_mb = self.check_memory_usage()
            if memory_mb > self.warning_memory_threshold:
                logger.warning(f"⚠️ Memory usage high ({memory_mb:.1f}MB) before loading Whisper")
                self.cleanup_memory()
            
            # Check memory again after cleanup
            memory_mb = self.check_memory_usage()
            if memory_mb > 2500:  # Hard limit
                logger.error(f"🚨 Memory too high ({memory_mb:.1f}MB), cannot load Whisper safely")
                raise Exception(f"Memory limit exceeded: {memory_mb:.1f}MB (max: 2500MB)")
            
            logger.info("📥 Loading Whisper model (small)...")
            try:
                self._whisper_model = whisper.load_model("small")
                logger.info("✅ Whisper model (small) loaded successfully")
                
                # Check memory after loading
                memory_mb = self.check_memory_usage()
                logger.info(f"📊 Memory after Whisper load: {memory_mb:.1f} MB")
                
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper model: {e}")
                raise
        else:
            logger.info(f"♻️ Using existing Whisper model")
        
        return self._whisper_model
    
    def extract_loom_video_info(self, loom_url: str) -> Optional[Dict]:
        """Extract video information from Loom URL"""
        try:
            logger.info(f"Extracting Loom video info from: {loom_url}")
            
            # Parse Loom URL to get video ID
            parsed_url = urlparse(loom_url)
            path_parts = parsed_url.path.strip('/').split('/')
            
            if len(path_parts) >= 2:
                video_id = path_parts[-1]
                
                # Create minimal video data structure
                video_data = {
                    'url': loom_url,
                    'title': f'Loom Video - {video_id}',
                    'duration': 0
                }
                
                # Extract relevant info
                video_info = {
                    'video_id': video_id,
                    'title': video_data.get('title', 'Unknown'),
                    'duration': video_data.get('duration', 0),
                    'video_url': video_data.get('url'),
                    'thumbnail_url': None
                }
                
                logger.info(f"Loom video info extracted: {video_info['title']}")
                return video_info
                
        except Exception as e:
            logger.error(f"Failed to extract Loom video info: {e}")
            return None
    
    def download_loom_video_with_quality_fallback(self, video_url: str, output_path: str) -> bool:
        """Download Loom video with quality fallback"""
        try:
            logger.info(f"Downloading Loom video with quality fallback: {video_url}")
            
            # Check memory before download
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory before download: {memory_mb:.1f} MB")
            
            # Quality levels to try
            quality_formats = [
                "hls-raw-1500+hls-raw-audio-audio",  # 720p + Audio
                "hls-raw-3200+hls-raw-audio-audio",  # 1080p + Audio
                "hls-raw-1500",                      # 720p video only
                "hls-raw-3200",                      # 1080p video only
                "best"                               # Any available format
            ]
            
            quality_names = ["720p+audio", "1080p+audio", "720p", "1080p", "best"]
            
            for i, (format_spec, quality_name) in enumerate(zip(quality_formats, quality_names)):
                try:
                    logger.info(f"Attempting download with {quality_name} quality (attempt {i+1}/{len(quality_formats)})")
                    
                    # Check memory before each attempt
                    memory_mb = self.check_memory_usage()
                    if memory_mb > self.memory_threshold:
                        logger.warning(f"High memory before {quality_name} download ({memory_mb:.1f}MB), skipping")
                        continue
                    
                    # Use yt-dlp with specific quality
                    import subprocess
                    import sys
                    import os
                    
                    # Ensure parent dir exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Remove existing file if it exists
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                            logger.info("Removed existing file before download")
                        except Exception:
                            pass
                    
                    # Build yt-dlp command
                    cmd = [
                        sys.executable, '-m', 'yt_dlp',
                        '--no-warnings',
                        '--retries', '2', '--fragment-retries', '2',
                        '--restrict-filenames',
                        '--merge-output-format', 'mp4',
                        '--force-overwrites',
                        '--format', format_spec,
                        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        '--referer', 'https://www.loom.com/',
                        '--add-header', 'Origin: https://www.loom.com',
                        '--add-header', 'Sec-Fetch-Mode: navigate',
                        '--output', output_path,
                        video_url
                    ]
                    
                    logger.info(f"yt-dlp command: {' '.join(cmd[:8])}... --format {format_spec} ...")
                    
                    # Run yt-dlp with timeout
                    timeout = 180 if i == 0 else 120
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                    
                    # Handle yt-dlp output
                    if result.returncode == 0 and not os.path.exists(output_path):
                        candidate_mp4 = output_path if output_path.endswith('.mp4') else f"{output_path}.mp4"
                        if os.path.exists(candidate_mp4):
                            try:
                                os.replace(candidate_mp4, output_path)
                                logger.info("Renamed downloaded file to expected path")
                            except Exception:
                                pass
                    
                    # Check if download was successful
                    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                        logger.info(f"Successfully downloaded with {quality_name} quality: {file_size_mb:.1f} MB")
                        
                        # Check memory after successful download
                        memory_mb = self.check_memory_usage()
                        logger.info(f"Memory after {quality_name} download: {memory_mb:.1f} MB")
                        
                        return True
                    else:
                        error_msg = result.stderr or result.stdout
                        logger.warning(f"Download failed (code {result.returncode}): {error_msg[:200]}...")
                        
                        # Clean up failed download
                        if os.path.exists(output_path):
                            try:
                                os.remove(output_path)
                            except Exception:
                                pass
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"Download timed out after {timeout}s")
                except Exception as e:
                    logger.warning(f"Download failed with exception: {e}")
                
                # Small delay between attempts
                if i < len(quality_formats) - 1:
                    time.sleep(2)
            
            logger.error("All quality levels failed for video download")
            return False
            
        except Exception as e:
            logger.error(f"Failed to download Loom video with quality fallback: {e}")
            return False
    
    def transcribe_video(self, video_path: str) -> Optional[Dict]:
        """Transcribe video using Whisper"""
        try:
            logger.info(f"Transcribing video: {video_path}")
            
            # Check memory before transcription
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory before transcription: {memory_mb:.1f} MB")
            
            if memory_mb > self.memory_threshold:
                logger.warning(f"High memory before transcription ({memory_mb:.1f}MB), performing cleanup")
                self.cleanup_memory()
            
            # Load Whisper model
            model = self.get_whisper_model()
            
            # Check file size and accessibility
            import os
            if not os.path.exists(video_path):
                raise Exception(f"Video file does not exist: {video_path}")
            
            if not os.access(video_path, os.R_OK):
                raise Exception(f"Video file is not readable: {video_path}")
            
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024
            logger.info(f"Video file size: {file_size_mb:.1f} MB")
            logger.info(f"Video file path: {video_path}")
            logger.info(f"Video file exists: {os.path.exists(video_path)}")
            logger.info(f"Video file readable: {os.access(video_path, os.R_OK)}")
            
            if file_size_mb > 50:
                logger.warning(f"Large video file ({file_size_mb:.1f}MB), transcription may be slow")
            
            # Transcribe video
            logger.info("Starting Whisper transcription...")
            
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            
            try:
                # Ensure model is available
                if model is None:
                    logger.warning("Model is None, reloading...")
                    model = self.get_whisper_model()
                
                logger.info(f"Model status before transcription: {type(model).__name__ if model else 'None'}")
                logger.info("Starting Whisper transcription...")
                
                result = model.transcribe(
                    video_path,
                    word_timestamps=True,
                    verbose=False,
                    fp16=False,
                    condition_on_previous_text=False,
                    temperature=0.0
                )
                logger.info("Transcription completed successfully")
            except Exception as e:
                logger.error(f"Standard transcription failed: {e}")
                # Try lightweight transcription as fallback
                logger.info("Attempting lightweight transcription...")
                try:
                    if model is None:
                        logger.warning("Model was cleared, reloading...")
                        model = self.get_whisper_model()
                    
                    result = model.transcribe(
                        video_path,
                        word_timestamps=True,
                        verbose=False,
                        fp16=False
                    )
                    logger.info("Lightweight transcription completed")
                except Exception as e2:
                    logger.error(f"Lightweight transcription also failed: {e2}")
                    raise Exception(f"Both transcription methods failed. Standard error: {e}, lightweight error: {e2}")
            
            # Check memory after transcription
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory after transcription: {memory_mb:.1f} MB")
            
            # Process segments
            enhanced_segments = []
            
            # If we have multiple segments, use them
            if len(result.get('segments', [])) > 1:
                for segment in result.get('segments', []):
                    start = float(segment.get('start', 0.0))
                    end = float(segment.get('end', start))
                    
                    # If end time is missing or same as start, estimate it
                    if end <= start:
                        text = segment.get('text', '').strip()
                        estimated_duration = max(len(text.split()) * 0.5, 1.0)
                        end = start + estimated_duration
                    
                    enhanced_segments.append({
                        'text': segment.get('text', '').strip(),
                        'start': start,
                        'end': end
                    })
            else:
                # For single segment, create time-based sub-segments
                logger.info("Single segment detected, creating time-based sub-segments")
                text = result.get('text', '').strip()
                word_count = len(text.split())
                
                # Estimate duration for Loom videos
                words_per_second = 2.5
                estimated_duration = max(60, word_count / words_per_second)
                
                # Create sub-segments every 20 seconds
                segment_duration = 20
                num_sub_segments = max(3, int(estimated_duration / segment_duration))
                
                logger.info(f"Creating {num_sub_segments} sub-segments for {estimated_duration:.1f}s video")
                
                for i in range(num_sub_segments):
                    start_time = i * segment_duration
                    end_time = min((i + 1) * segment_duration, estimated_duration)
                    
                    # Extract text for this sub-segment
                    text_start = int((start_time / estimated_duration) * len(text))
                    text_end = int((end_time / estimated_duration) * len(text))
                    sub_text = text[text_start:text_end].strip()
                    
                    if sub_text and len(sub_text) > 5:
                        enhanced_segments.append({
                            'text': sub_text,
                            'start': start_time,
                            'end': end_time
                        })
                        logger.info(f"Created sub-segment {i+1}: {start_time}s → {end_time}s ({len(sub_text)} chars)")
                
                # If no sub-segments created, use the original segment
                if len(enhanced_segments) == 0:
                    enhanced_segments.append({
                        'text': text,
                        'start': 0,
                        'end': estimated_duration
                    })
                    logger.info(f"Using original segment: 0s → {estimated_duration:.1f}s")
            
            transcription_data = {
                'transcription': result['text'],
                'segments': enhanced_segments,
                'language': result.get('language', 'en'),
                'word_count': len(result['text'].split())
            }
            
            logger.info(f"Transcription completed: {transcription_data['word_count']} words")
            logger.info(f"Language: {transcription_data.get('language', 'Unknown')}")
            logger.info(f"Enhanced segments created: {len(enhanced_segments)}")
            
            # Cleanup memory after transcription
            self.cleanup_memory()
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.cleanup_memory()
            return None
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks using OpenAI"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} chunks...")
            
            embeddings = []
            batch_size = 100  # OpenAI batch size limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    response = openai.embeddings.create(
                        input=batch,
                        model="text-embedding-3-small"
                    )
                    batch_embeddings = [e.embedding for e in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f"Created embeddings for batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")
                    # Create zero embeddings for failed batch
                    zero_embedding = [0.0] * 1536  # OpenAI embedding dimension
                    embeddings.extend([zero_embedding] * len(batch))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            return []
    
    def store_in_pinecone(self, company_name: str, video_url: str, video_info: Dict, 
                         transcription_data: Dict, chunks: List[Dict], embeddings: List[List[float]], 
                         qudemo_id: str = None) -> bool:
        """Store transcription chunks and embeddings in Pinecone"""
        try:
            logger.info(f"Storing in Pinecone for company: {company_name} qudemo: {qudemo_id}")
            
            # Create or get single shared index
            index_name = self.default_index_name
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                try:
                    logger.info(f"Creating new Pinecone index: {index_name}")
                    self.pc.create_index(
                        name=index_name,
                        dimension=1536,  # OpenAI embedding dimension
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                    # Wait for index to be ready
                    time.sleep(10)
                except Exception as ce:
                    msg = str(ce)
                    if 'max serverless indexes' in msg.lower() or 'forbidden' in msg.lower():
                        if existing_indexes:
                            fallback = existing_indexes[0]
                            logger.warning(f"Index quota reached; falling back to existing index: {fallback}")
                            index_name = fallback
                        else:
                            logger.error("No existing Pinecone indexes available to fallback to.")
                            raise
                    else:
                        raise
            
            # Get index and namespace per company and qudemo
            index = self.pc.Index(index_name)
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}" if qudemo_id else company_name.lower().replace(' ', '-')
            logger.info(f"Storing data in namespace: '{namespace}' in index: '{index_name}'")
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{company_name}_{qudemo_id}_{video_url}_{i}" if qudemo_id else f"{company_name}_{video_url}_{i}"
                
                # Extract and validate timestamps
                chunk_start = float(chunk.get('start_timestamp', 0.0)) if isinstance(chunk, dict) else 0.0
                chunk_end = float(chunk.get('end_timestamp', 0.0)) if isinstance(chunk, dict) else 0.0
                
                vector_data = {
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'company': company_name,
                        'qudemo_id': qudemo_id,
                        'video_url': video_url,
                        'chunk_index': i,
                        'text': chunk['text'] if isinstance(chunk, dict) else str(chunk),
                        'start': chunk_start,
                        'end': chunk_end,
                        'title': video_info.get('title', 'Unknown'),
                        'duration': video_info.get('duration', 'Unknown'),
                        'language': transcription_data.get('language', 'Unknown'),
                        'word_count': transcription_data.get('word_count', 'Unknown'),
                        'source_type': 'video'
                    }
                }
                
                # Debug timestamp storage
                if chunk_start > 0.0 or chunk_end > 0.0:
                    logger.info(f"Storing chunk {i+1}: start={chunk_start:.2f}s, end={chunk_end:.2f}s")
                
                vectors.append(vector_data)
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"Upserted batch {i//batch_size + 1}")
            
            logger.info(f"Successfully stored {len(vectors)} vectors in Pinecone for {company_name} qudemo {qudemo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Pinecone storage failed: {e}")
            return False
    
    def process_video(self, video_url: str, company_name: str, qudemo_id: str = None) -> Optional[Dict]:
        """Main Loom video processing pipeline"""
        try:
            logger.info(f"🎬 Processing Loom video: {video_url}")
            logger.info(f"🏢 Company: {company_name}, Qudemo ID: {qudemo_id}")
            
            # Memory check before starting
            memory_mb = self.check_memory_usage()
            if memory_mb > self.warning_memory_threshold:
                logger.warning(f"⚠️ High memory before processing: {memory_mb:.1f}MB")
                self.cleanup_memory()
                memory_mb = self.check_memory_usage()
            
            if memory_mb > self.memory_threshold:
                logger.error(f"🚨 Memory too high for processing: {memory_mb:.1f}MB")
                return {
                    "success": False,
                    "error": f"Memory usage too high ({memory_mb:.1f}MB) for video processing",
                    "code": "MEMORY_LIMIT_EXCEEDED"
                }
            
            logger.info(f"✅ Memory check passed: {memory_mb:.1f}MB")
            
            # Step 1: Extract video info
            video_info = self.extract_loom_video_info(video_url)
            if not video_info:
                raise Exception("Failed to extract video info")
            
            # Step 2: Download video
            # Create a more reliable temporary file path
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            temp_video_path = os.path.join(temp_dir, f"loom_video_{int(time.time())}.mp4")
            
            # Use quality fallback download
            logger.info("Using quality fallback download")
            download_success = self.download_loom_video_with_quality_fallback(video_url, temp_video_path)
            
            if not download_success:
                raise Exception("Failed to download video with quality fallback")
            
            # Check memory before transcription
            memory_mb = self.check_memory_usage()
            if memory_mb > self.memory_threshold:
                logger.warning(f"⚠️ High memory before transcription ({memory_mb:.1f}MB), performing cleanup")
                self.cleanup_memory()
            
            # Check memory again after cleanup
            memory_mb = self.check_memory_usage()
            if memory_mb > 2500:  # Hard limit
                logger.error(f"🚨 Memory still too high ({memory_mb:.1f}MB) after cleanup, skipping video")
                return {
                    "success": False,
                    "message": f"Memory usage too high ({memory_mb:.1f}MB), video too large to process safely",
                    "error": "memory_limit_exceeded"
                }
            
            # Step 3: Transcribe video
            transcription_data = self.transcribe_video(temp_video_path)
            if not transcription_data:
                raise Exception("Failed to transcribe video")
            
            # Check memory after transcription
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory after transcription: {memory_mb:.1f} MB")
            
            # Step 4: Create chunks from segments
            transcription = transcription_data.get('transcription', '')
            if not transcription:
                raise Exception("Empty transcription")
            segments = transcription_data.get('segments', [])
            
            # Create chunks from enhanced segments
            chunks = []
            for i, segment in enumerate(segments):
                chunk_data = {
                    'text': segment.get('text', ''),
                    'full_context': segment.get('text', ''),
                    'source': 'video',
                    'title': f'Video Transcription - {company_name}',
                    'url': video_url,
                    'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'start_timestamp': segment.get('start', 0),
                    'end_timestamp': segment.get('end', 0),
                    'chunk_index': i,
                    'total_chunks': len(segments)
                }
                chunks.append(chunk_data)
            
            # Log chunk information
            logger.info(f"Created {len(chunks)} timestamped chunks from enhanced segments")
            
            # Check memory before embeddings
            memory_mb = self.check_memory_usage()
            if memory_mb > self.memory_threshold:
                logger.warning(f"⚠️ High memory before embeddings ({memory_mb:.1f}MB), performing cleanup")
                self.cleanup_memory()
            
            # Step 5: Create embeddings
            embeddings = self.create_embeddings([c['text'] for c in chunks])
            if not embeddings or len(embeddings) != len(chunks):
                raise Exception("Failed to create embeddings")
            
            # Check memory before storage
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory before storage: {memory_mb:.1f} MB")
            
            # Step 6: Store in Pinecone
            storage_success = self.store_in_pinecone(
                company_name, video_url, video_info, transcription_data, chunks, embeddings, qudemo_id
            )
            
            if not storage_success:
                raise Exception("Failed to store in Pinecone")
            
            # Final memory cleanup
            self.cleanup_memory()
            
            # Return success result
            result = {
                'success': True,
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'title': video_info.get('title', 'Unknown'),
                'chunks_created': len(chunks),
                'vectors_stored': len(embeddings),
                'word_count': transcription_data.get('word_count', 'Unknown'),
                'language': transcription_data.get('language', 'Unknown'),
                'method': 'loom_transcription',
                'memory_usage_mb': self.check_memory_usage(),
                'production_mode': True
            }
            
            logger.info(f"✅ Loom video processing completed successfully for {company_name} qudemo {qudemo_id}")
            
            # Clean up temporary file
            try:
                os.unlink(temp_video_path)
                logger.info("🧹 Cleaned up temporary video file")
            except:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Loom video processing failed: {e}")
            # Cleanup on error
            self.cleanup_memory()
            
            # Clean up temporary file
            try:
                os.unlink(temp_video_path)
                logger.info("🧹 Cleaned up temporary video file (error)")
            except:
                pass
            
            return {
                "success": False,
                "error": str(e),
                "code": "PROCESSING_FAILED"
            }
