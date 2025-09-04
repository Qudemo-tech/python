#!/usr/bin/env python3
"""
Production-Ready Loom Video Processor
Handles Loom video processing without external FFmpeg dependency
Optimized for cloud deployment (Render, Heroku, etc.)
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
        """Initialize Production Loom Video Processor"""
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        
        # Configure OpenAI for embeddings
        openai.api_key = openai_api_key
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.default_index_name = os.getenv("PINECONE_INDEX", "qudemo-index")
        
        # Initialize Whisper model (lazy loading)
        self._whisper_model = None
        
        # Memory management for cloud deployment
        self.memory_threshold = 3000  # MB (more conservative for cloud)
        self.warning_memory_threshold = 2000   # MB
        
        logger.info("Initializing Production Loom Video Processor (Cloud Optimized)...")
    
    def check_memory_usage(self) -> float:
        """Check current memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            status = "Safe" if memory_mb < self.warning_memory_threshold else "Warning" if memory_mb < self.memory_threshold else "Critical"
            logger.info(f"✅ Memory usage: {memory_mb:.1f} MB ({status})")
            return memory_mb
        except Exception as e:
            logger.warning(f"Could not check memory usage: {e}")
            return 0.0
    
    def cleanup_memory(self):
        """Clean up memory"""
        try:
            logger.info("🧹 Performing memory cleanup...")
            gc.collect()
            memory_after = self.check_memory_usage()
            logger.info(f"🧹 Memory cleanup completed: {memory_after:.1f}MB")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def get_whisper_model(self):
        """Get or load Whisper model with memory management"""
        if self._whisper_model is None:
            memory_before = self.check_memory_usage()
            if memory_before > self.memory_threshold:
                logger.warning(f"High memory usage before loading Whisper: {memory_before:.1f} MB")
                self.cleanup_memory()
            
            logger.info("📥 Loading Whisper model (small)...")
            self._whisper_model = whisper.load_model("small")
            logger.info("✅ Whisper model (small) loaded successfully")
            
            memory_after = self.check_memory_usage()
            logger.info(f"📊 Memory after Whisper load: {memory_after:.1f} MB")
        
        return self._whisper_model
    
    def extract_loom_video_info(self, video_url: str) -> Dict[str, str]:
        """Extract video information from Loom URL"""
        try:
            logger.info(f"Extracting Loom video info from: {video_url}")
            
            # Parse the URL to extract video ID
            parsed_url = urlparse(video_url)
            if 'loom.com' not in parsed_url.netloc:
                raise Exception("Invalid Loom URL")
            
            # Extract video ID from URL
            video_id = None
            if '/share/' in video_url:
                video_id = video_url.split('/share/')[1].split('?')[0]
            
            if not video_id:
                raise Exception("Could not extract video ID from URL")
            
            # Create a simple title
            title = f"Loom Video - {video_id}"
            
            logger.info(f"Loom video info extracted: {title}")
            
            return {
                'title': title,
                'video_id': video_id,
                'url': video_url
            }
            
        except Exception as e:
            logger.error(f"Failed to extract Loom video info: {e}")
            raise
    
    def download_loom_video(self, video_url: str, output_path: str) -> bool:
        """Download Loom video using yt-dlp with production-compatible settings"""
        try:
            logger.info(f"Downloading Loom video: {video_url}")
            
            # Check memory before download
            memory_before = self.check_memory_usage()
            logger.info(f"Memory before download: {memory_before:.1f} MB")
            
            # Use yt-dlp with production-compatible settings
            import subprocess
            import sys
            
            # yt-dlp command with fallback formats
            cmd = [
                sys.executable, '-m', 'yt_dlp',
                '--no-warnings',
                '--retries', '2',
                '--fragment-retries', '2',
                '--output', output_path,
                '--format', 'best[height<=720]/best',  # Prefer 720p or lower
                video_url
            ]
            
            logger.info(f"yt-dlp command: {' '.join(cmd)}")
            
            # Run yt-dlp
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                if os.path.exists(output_path):
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    logger.info(f"Successfully downloaded: {file_size_mb:.1f} MB")
                    
                    memory_after = self.check_memory_usage()
                    logger.info(f"Memory after download: {memory_after:.1f} MB")
                    return True
                else:
                    logger.error("Download completed but file not found")
                    return False
            else:
                logger.error(f"Download failed (code {result.returncode}): {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Download timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def transcribe_video(self, video_path: str) -> Dict[str, any]:
        """Production-compatible video transcription"""
        try:
            logger.info(f"Transcribing video: {video_path}")
            
            # Check memory before transcription
            memory_before = self.check_memory_usage()
            logger.info(f"Memory before transcription: {memory_before:.1f} MB")
            
            if memory_before > self.memory_threshold:
                logger.warning(f"High memory usage before transcription: {memory_before:.1f} MB")
                self.cleanup_memory()
            
            # Load Whisper model
            model = self.get_whisper_model()
            
            # Validate video file
            if not os.path.exists(video_path):
                raise Exception(f"Video file does not exist: {video_path}")
            
            if not os.access(video_path, os.R_OK):
                raise Exception(f"Video file is not readable: {video_path}")
            
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"Video file size: {file_size_mb:.1f} MB")
            
            if file_size_mb < 0.1:
                raise Exception(f"Video file too small: {file_size_mb:.1f} MB")
            
            # Production-compatible transcription
            logger.info("Starting production-compatible Whisper transcription...")
            
            # Use minimal settings for maximum compatibility
            result = model.transcribe(
                video_path,
                word_timestamps=True,
                fp16=False,  # Disable fp16 for cloud compatibility
                temperature=0.0,  # Deterministic output
                verbose=False,  # Reduce logging
                condition_on_previous_text=False  # Disable for stability
            )
            
            logger.info("Transcription completed successfully")
            
            # Check memory after transcription
            memory_after = self.check_memory_usage()
            logger.info(f"Memory after transcription: {memory_after:.1f} MB")
            
            # Extract results
            segments = result.get('segments', [])
            text = result.get('text', '').strip()
            language = result.get('language', 'en')
            
            logger.info(f"Transcription completed: {len(text.split())} words")
            logger.info(f"Language: {language}")
            
            # Enhance segments with better timing
            enhanced_segments = self.validate_and_enhance_timestamps(segments)
            logger.info(f"Enhanced segments created: {len(enhanced_segments)}")
            
            return {
                'text': text,
                'segments': enhanced_segments,
                'language': language,
                'word_count': len(text.split())
            }
            
        except Exception as e:
            logger.error(f"Production transcription failed: {e}")
            raise
    
    def validate_and_enhance_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """Validate and enhance timestamp segments"""
        enhanced_segments = []
        
        for i, segment in enumerate(segments):
            try:
                start = float(segment.get('start', 0))
                end = float(segment.get('end', 0))
                text = segment.get('text', '').strip()
                
                # Validate timestamps
                if start < 0:
                    start = 0
                if end <= start:
                    end = start + 1.0
                
                # Enhance with additional metadata
                enhanced_segment = {
                    'start': start,
                    'end': end,
                    'text': text,
                    'segment_id': i,
                    'duration': end - start,
                    'word_count': len(text.split())
                }
                
                enhanced_segments.append(enhanced_segment)
                
            except Exception as e:
                logger.warning(f"Failed to enhance segment {i}: {e}")
                continue
        
        return enhanced_segments
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} chunks...")
            
            embeddings = []
            batch_size = 100  # OpenAI batch limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    response = openai.embeddings.create(
                        input=batch,
                        model="text-embedding-3-large"
                    )
                    batch_embeddings = [e.embedding for e in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f"Created embeddings for batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")
                    # Create zero embeddings for failed batch
                    zero_embedding = [0.0] * 3072  # OpenAI embedding dimension
                    embeddings.extend([zero_embedding] * len(batch))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    def store_in_pinecone(self, chunks: List[Dict], embeddings: List[List[float]], 
                         company_name: str, qudemo_id: str) -> bool:
        """Store chunks in Pinecone with Standard Plan configuration"""
        try:
            logger.info(f"Storing in Pinecone for company: {company_name} qudemo: {qudemo_id}")
            
            # Use dedicated video index for Standard Plan
            index_name = "qudemo-video-index"
            namespace = f"{company_name}-{qudemo_id}"
            
            # Check if index exists, create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                try:
                    logger.info(f"Creating new Pinecone video index: {index_name}")
                    self.pc.create_index(
                        name=index_name,
                        dimension=3072,  # OpenAI embedding dimension
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                    logger.info(f"✅ Created Pinecone index: {index_name}")
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
            
            # Get the index
            index = self.pc.Index(index_name)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{company_name}-{qudemo_id}-video-{i}"
                
                vector = {
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'company_name': company_name,
                        'qudemo_id': qudemo_id,
                        'content_type': 'video',
                        'source': 'loom',
                        'chunk_index': i,
                        'start_time': chunk.get('start', 0),
                        'end_time': chunk.get('end', 0),
                        'text': chunk.get('text', ''),
                        'word_count': chunk.get('word_count', 0)
                    }
                }
                vectors.append(vector)
            
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
    
    def process_video(self, video_url: str, company_name: str, qudemo_id: str) -> Dict[str, any]:
        """Process Loom video end-to-end"""
        try:
            logger.info(f"🎬 Processing Loom video: {video_url}")
            logger.info(f"🏢 Company: {company_name}, Qudemo ID: {qudemo_id}")
            
            # Check memory at start
            memory_start = self.check_memory_usage()
            if memory_start > self.memory_threshold:
                logger.warning(f"High memory usage at start: {memory_start:.1f} MB")
                self.cleanup_memory()
            
            # Extract video info
            video_info = self.extract_loom_video_info(video_url)
            
            # Download video
            temp_video_path = os.path.join(tempfile.gettempdir(), f"loom_video_{int(time.time())}.mp4")
            
            if not self.download_loom_video(video_url, temp_video_path):
                raise Exception("Failed to download video")
            
            try:
                # Transcribe video
                transcription_result = self.transcribe_video(temp_video_path)
                
                # Create chunks from segments
                chunks = []
                for segment in transcription_result['segments']:
                    chunk = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'],
                        'word_count': segment['word_count']
                    }
                    chunks.append(chunk)
                
                logger.info(f"Created {len(chunks)} timestamped chunks from enhanced segments")
                
                # Create embeddings
                texts = [chunk['text'] for chunk in chunks]
                embeddings = self.create_embeddings(texts)
                
                # Store in Pinecone
                if self.store_in_pinecone(chunks, embeddings, company_name, qudemo_id):
                    logger.info(f"✅ Loom video processing completed successfully for {company_name} qudemo {qudemo_id}")
                    
                    return {
                        'success': True,
                        'chunks_created': len(chunks),
                        'word_count': transcription_result['word_count'],
                        'language': transcription_result['language'],
                        'company_name': company_name,
                        'qudemo_id': qudemo_id
                    }
                else:
                    raise Exception("Failed to store in Pinecone")
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    try:
                        os.unlink(temp_video_path)
                        logger.info("🧹 Cleaned up temporary video file")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temporary file: {e}")
                
                # Final memory cleanup
                self.cleanup_memory()
            
        except Exception as e:
            logger.error(f"❌ Loom video processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_created': 0
            }
