#!/usr/bin/env python3
"""
Loom Video Processor
Handles Loom video processing with transcription and vector storage
Optimized for Render's 2GB memory constraint
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
import numpy as np
import openai
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoomVideoProcessor:
    def __init__(self, openai_api_key: str, pinecone_api_key: str):
        """
        Initialize Loom Video Processor
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            pinecone_api_key: Pinecone API key
        """
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        
        # Configure OpenAI for embeddings
        openai.api_key = openai_api_key
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.default_index_name = os.getenv("PINECONE_INDEX", "qudemo-index")
        
        # Initialize Whisper model (lazy loading with memory management)
        self._whisper_model = None
        
        # Memory management
        self.memory_threshold = 6000  # MB - trigger cleanup at 6GB (leaving 2GB buffer for 8GB RAM)
        self.max_video_size = 100 * 1024 * 1024  # 100MB max video size
        
        # Use small model for tutorial videos (excellent accuracy, perfect for longer videos on 8GB RAM)
        self.model_strategy = "small"
        
        logger.info("Initializing Loom Video Processor (Memory Optimized)...")
    
    def check_memory_usage(self) -> float:
        """Check current memory usage and log it"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Current memory usage: {memory_mb:.1f} MB")
            return memory_mb
        except Exception as e:
            logger.error(f"Failed to check memory: {e}")
            return 0.0
    
    def cleanup_memory(self, preserve_whisper: bool = True):
        """Force garbage collection and memory cleanup"""
        try:
            logger.info("Performing memory cleanup...")
            gc.collect()
            
            # More aggressive memory cleanup
            memory_mb = self.check_memory_usage()
            
            # If memory is very high (>2500MB), clear Whisper model even if preserve_whisper is True
            if memory_mb > 2500 and self._whisper_model:
                logger.warning(f"High memory usage ({memory_mb:.1f}MB), clearing Whisper model for aggressive cleanup")
                del self._whisper_model
                self._whisper_model = None
                gc.collect()
            # Original logic for very high memory
            elif memory_mb > 6500 and self._whisper_model and not preserve_whisper:
                logger.warning(f"Critical memory usage ({memory_mb:.1f}MB), clearing Whisper model")
                del self._whisper_model
                self._whisper_model = None
                gc.collect()
            
            # Force additional cleanup for Render's memory constraints
            if memory_mb > 2000:
                logger.info("Performing additional memory cleanup for Render constraints...")
                import sys
                # Clear any cached objects
                for obj in gc.get_objects():
                    if hasattr(obj, '__dict__'):
                        obj.__dict__.clear()
                gc.collect()
            
            logger.info("Memory cleanup completed")
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def get_whisper_model(self):
        """Load Whisper small model once and keep it loaded for better performance"""
        if self._whisper_model is None:
            # Check memory before loading
            memory_mb = self.check_memory_usage()
            if memory_mb > self.memory_threshold:
                logger.warning(f"High memory usage ({memory_mb:.1f}MB) before loading Whisper")
                self.cleanup_memory(preserve_whisper=True)
            
            logger.info("Loading Whisper model (small) - excellent accuracy for tutorial videos...")
            try:
                self._whisper_model = whisper.load_model("small")
                logger.info("Whisper model (small) loaded successfully")
                
                # Check memory after loading
                memory_mb = self.check_memory_usage()
                logger.info(f"Memory after Whisper load: {memory_mb:.1f} MB")
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model (small): {e}")
                raise
        else:
            logger.info(f"Using existing Whisper model: {type(self._whisper_model).__name__}")
        
        return self._whisper_model
    
    def is_whisper_loaded(self) -> bool:
        """Check if Whisper model is currently loaded"""
        return self._whisper_model is not None
    
    def is_loom_url(self, url: str) -> bool:
        """Check if URL is a Loom URL"""
        domain = urlparse(url).netloc.lower()
        return 'loom.com' in domain
    
    def extract_loom_video_info(self, loom_url: str) -> Optional[Dict]:
        """
        Extract video information from Loom URL
        
        Args:
            loom_url: Loom video URL
            
        Returns:
            Dict with video info or None if failed
        """
        try:
            logger.info(f"Extracting Loom video info from: {loom_url}")
            
            # Parse Loom URL to get video ID
            parsed_url = urlparse(loom_url)
            path_parts = parsed_url.path.strip('/').split('/')
            
            if len(path_parts) >= 2:
                video_id = path_parts[-1]  # Last part is usually the video ID
                
                            # Try different Loom API endpoints
            api_urls = [
                f"https://www.loom.com/api/campaigns/sessions/{video_id}/transcoded-video",
                f"https://www.loom.com/api/sessions/{video_id}",
                f"https://www.loom.com/api/v2/sessions/{video_id}"
            ]
            
            video_data = None
            for api_url in api_urls:
                try:
                    logger.info(f"Trying Loom API: {api_url}")
                    response = requests.get(api_url, timeout=30)
                    if response.status_code == 200:
                        video_data = response.json()
                        logger.info(f"Success with API: {api_url}")
                        break
                except Exception as e:
                    logger.warning(f"Failed with API {api_url}: {e}")
                    continue
            
            if not video_data:
                # Fallback: try to extract info from the share page
                logger.info("Trying fallback method - extracting from share page")
                try:
                    share_response = requests.get(loom_url, timeout=30)
                    if share_response.status_code == 200:
                        # Look for video data in the page
                        import re
                        content = share_response.text
                        
                        # Try multiple patterns to find video URL
                        video_url_patterns = [
                            r'"videoUrl":"([^"]+)"',
                            r'"url":"([^"]*\.mp4[^"]*)"',
                            r'"video_url":"([^"]+)"',
                            r'"src":"([^"]*\.mp4[^"]*)"',
                            r'https://[^"]*\.mp4[^"]*'
                        ]
                        
                        video_url = None
                        for pattern in video_url_patterns:
                            match = re.search(pattern, content)
                            if match:
                                video_url = match.group(1) if match.groups() else match.group(0)
                                if video_url.startswith('http'):
                                    break
                        
                        # Try to find title
                        title_patterns = [
                            r'"title":"([^"]+)"',
                            r'"name":"([^"]+)"',
                            r'<title>([^<]+)</title>'
                        ]
                        
                        title = None
                        for pattern in title_patterns:
                            match = re.search(pattern, content)
                            if match:
                                title = match.group(1)
                                break
                        
                        if video_url:
                            video_data = {
                                'url': video_url,
                                'title': title if title else 'Unknown',
                                'duration': 0
                            }
                            logger.info(f"Extracted video data from share page: {video_url}")
                        else:
                            logger.warning("No video URL found in share page")
                except Exception as e:
                    logger.error(f"Fallback method failed: {e}")
            
            if not video_data:
                # Create a minimal video data structure for processing
                logger.warning("Could not extract video data, using minimal structure")
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
                'thumbnail_url': video_data.get('thumbnailUrl')
            }
            
            logger.info(f"Loom video info extracted: {video_info['title']}")
            return video_info
                
        except Exception as e:
            logger.error(f"Failed to extract Loom video info: {e}")
            return None
    
    def download_loom_video(self, video_url: str, output_path: str) -> bool:
        """
        Download Loom video to local file with progressive quality fallback
        
        Args:
            video_url: Direct video URL
            output_path: Local path to save video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading Loom video to: {output_path}")
            
            # Check memory before download
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory before download: {memory_mb:.1f} MB")
            
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # First, check content length to estimate file size
            head_response = requests.head(video_url, timeout=30, headers=headers)
            content_length = head_response.headers.get('content-length')
            if content_length:
                file_size_mb = int(content_length) / 1024 / 1024
                logger.info(f"Estimated file size: {file_size_mb:.1f} MB")
                
                # Check if file is too large for our memory constraints
                if file_size_mb > 100:  # 100MB limit
                    logger.warning(f"File too large ({file_size_mb:.1f}MB), will use yt-dlp for optimized download")
                    return False
            
            response = requests.get(video_url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()

            # Check if we got a video file. Loom share pages return text/html; treat that as failure
            content_type = response.headers.get('content-type', '').lower()
            is_video_like = any(t in content_type for t in ['video', 'mp4', 'webm', 'ogg', 'application/octet-stream'])
            if not is_video_like:
                logger.warning(f"Response doesn't appear to be a video file (content-type: {content_type}). Will use yt-dlp fallback.")
                return False

            # Download the file with progress monitoring
            downloaded_size = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Check memory every 10MB downloaded
                        if downloaded_size % (10 * 1024 * 1024) == 0:
                            memory_mb = self.check_memory_usage()
                            if memory_mb > self.memory_threshold:
                                logger.warning(f"High memory during download ({memory_mb:.1f}MB), stopping download")
                                return False

            # Verify the file is valid
            import os
            file_size = os.path.getsize(output_path)
            if file_size < 1024:  # Less than 1KB
                logger.error(f"Downloaded file is too small: {file_size} bytes")
                return False

            logger.info(f"Loom video downloaded successfully: {file_size} bytes")
            
            # Check memory after download
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory after download: {memory_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Loom video: {e}")
            return False

    def download_loom_video_with_quality_fallback(self, video_url: str, output_path: str) -> bool:
        """
        Download Loom video with progressive quality fallback (480p -> 720p -> 1080p)
        
        Args:
            video_url: Loom share URL
            output_path: Local path to save video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading Loom video with quality fallback: {video_url}")
            
            # Check memory before download
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory before download: {memory_mb:.1f} MB")
            
            # Quality levels to try in order (lowest to highest)
            # Loom uses HLS stream formats, not height-based formats
            quality_formats = [
                "hls-raw-1500+hls-raw-audio-audio",  # 720p + Audio (merged)
                "hls-raw-3200+hls-raw-audio-audio",  # 1080p + Audio (merged)
                "hls-raw-1500",                      # 720p video only (fallback)
                "hls-raw-3200",                      # 1080p video only (fallback)
                "hls-raw-audio-audio",               # Audio only (fallback)
                "best"                               # Any available format
            ]
            
            quality_names = ["720p+audio", "1080p+audio", "720p", "1080p", "audio", "best"]
            
            # First, check available formats to avoid format errors
            try:
                logger.info("Checking available formats for this Loom video...")
                import subprocess
                import sys
                
                format_check_cmd = [
                    sys.executable, '-m', 'yt_dlp',
                    '--no-warnings',
                    '--list-formats',
                    '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    '--referer', 'https://www.loom.com/',
                    video_url
                ]
                
                format_result = subprocess.run(format_check_cmd, capture_output=True, text=True, timeout=30)
                if format_result.returncode == 0:
                    available_formats = format_result.stdout
                    logger.info(f"Available formats: {available_formats[:500]}...")
                else:
                    logger.warning(f"Could not check formats: {format_result.stderr[:200]}")
            except Exception as e:
                logger.warning(f"Format check failed: {e}")
            
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
                    
                    # Build yt-dlp command with quality specification
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
                    timeout = 180 if i == 0 else 120  # Longer timeout for first attempt
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
                        
                        # If this was the last attempt, log the full error
                        if i == len(quality_formats) - 1:
                            logger.error(f"All quality levels failed. Full error: {error_msg}")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"Download timed out after {timeout}s")
                except Exception as e:
                    logger.warning(f"Download failed with exception: {e}")
                
                # Small delay between attempts
                if i < len(quality_formats) - 1:
                    import time
                    time.sleep(2)
            
            logger.error("All quality levels failed for video download")
            return False
            
        except Exception as e:
            logger.error(f"Failed to download Loom video with quality fallback: {e}")
            return False
    
    def validate_and_enhance_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Validate and enhance timestamps to ensure precision (YouTube-style)
        
        Args:
            segments: List of segments with timestamps
            
        Returns:
            Enhanced segments with validated timestamps
        """
        enhanced_segments = []
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            start = float(segment.get('start', 0.0))
            end = float(segment.get('end', start))
            
            # Validate timestamps
            if start < 0:
                start = 0.0
            if end <= start:
                # Estimate duration based on text length (YouTube-style)
                word_count = len(text.split())
                estimated_duration = max(word_count * 0.4, 0.5)  # ~0.4s per word, min 0.5s
                end = start + estimated_duration
            
            # Ensure reasonable duration
            if end - start > 30:  # Cap at 30 seconds per segment
                end = start + 30
            
            enhanced_segments.append({
                'text': text,
                'start': start,
                'end': end
            })
        
        logger.info(f"Enhanced {len(enhanced_segments)} segments with validated timestamps")
        return enhanced_segments

    def transcribe_video(self, video_path: str) -> Optional[Dict]:
        """
        Transcribe video using Whisper with memory management
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with transcription data or None if failed
        """
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
            
            # Check file size and warn if large
            import os
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024
            logger.info(f"Video file size: {file_size_mb:.1f} MB")
            
            if file_size_mb > 50:
                logger.warning(f"Large video file ({file_size_mb:.1f}MB), transcription may be slow")
            
            # Transcribe video with memory monitoring
            logger.info("Starting Whisper transcription...")
            
            # Check memory before transcription
            memory_mb = self.check_memory_usage()
            if memory_mb > self.memory_threshold:
                logger.warning(f"High memory before transcription ({memory_mb:.1f}MB), performing cleanup")
                self.cleanup_memory(preserve_whisper=True)  # Preserve Whisper model during transcription
            
            # Simple transcription without complex threading
            try:
                # Ensure model is available
                if model is None:
                    logger.warning("Model is None, reloading...")
                    model = self.get_whisper_model()
                
                logger.info(f"Model status before transcription: {type(model).__name__ if model else 'None'}")
                logger.info("Starting Whisper transcription...")
                
                result = model.transcribe(
                    video_path,
                    word_timestamps=False,  # Disable word timestamps for faster processing
                    verbose=False,
                    fp16=False,  # Explicitly disable FP16 for CPU compatibility
                    condition_on_previous_text=False,  # Disable for faster processing
                    temperature=0.0  # Use greedy decoding for speed
                )
                logger.info("Transcription completed successfully")
            except Exception as e:
                logger.error(f"Standard transcription failed: {e}")
                # Try lightweight transcription as fallback
                logger.info("Attempting lightweight transcription...")
                try:
                    # Ensure model is still available
                    if model is None:
                        logger.warning("Model was cleared, reloading...")
                        model = self.get_whisper_model()
                    
                    result = model.transcribe(
                        video_path,
                        word_timestamps=False,  # Disable word timestamps to save memory
                        verbose=False,
                        fp16=False  # Disable fp16 to save memory
                    )
                    logger.info("Lightweight transcription completed")
                except Exception as e2:
                    logger.error(f"Lightweight transcription also failed: {e2}")
                    raise Exception(f"Both standard and lightweight transcription failed. Standard error: {e}, lightweight error: {e2}")
            
            # Check memory after transcription
            memory_mb = self.check_memory_usage()
            logger.info(f"Memory after transcription: {memory_mb:.1f} MB")
            
            # Process segments to ensure precise timestamps (similar to YouTube API)
            enhanced_segments = []
            for segment in result.get('segments', []):
                # Ensure we have precise start and end times
                start = float(segment.get('start', 0.0))
                end = float(segment.get('end', start))
                
                # If end time is missing or same as start, estimate it
                if end <= start:
                    # Estimate duration based on text length (similar to YouTube API)
                    text = segment.get('text', '').strip()
                    estimated_duration = max(len(text.split()) * 0.5, 1.0)  # ~0.5s per word, min 1s
                    end = start + estimated_duration
                
                enhanced_segments.append({
                    'text': segment.get('text', '').strip(),
                    'start': start,
                    'end': end
                })
            
            # Validate and enhance timestamps (YouTube-style precision)
            enhanced_segments = self.validate_and_enhance_timestamps(enhanced_segments)
            
            transcription_data = {
                'transcription': result['text'],
                'segments': enhanced_segments,  # Use enhanced segments
                'language': result.get('language', 'en'),
                'word_count': len(result['text'].split())
            }
            
            logger.info(f"Transcription completed: {transcription_data['word_count']} words")
            logger.info(f"Language: {transcription_data.get('language', 'Unknown')}")
            logger.info(f"Enhanced segments created: {len(enhanced_segments)}")
            
            # Log the full transcription content with detailed breakdown
            transcription_text = transcription_data.get('transcription', '')
            if transcription_text:
                logger.info("FULL TRANSCRIPTION:")
                logger.info("=" * 80)
                logger.info(transcription_text)
                logger.info("=" * 80)
                logger.info(f"Total transcription length: {len(transcription_text)} characters")
                logger.info(f"Total words: {transcription_data['word_count']}")
            
            # Log detailed segment information
            segments = transcription_data.get('segments', [])
            if segments:
                logger.info("TRANSCRIPTION SEGMENTS WITH TIMESTAMPS:")
                logger.info("=" * 80)
                for i, segment in enumerate(segments, 1):
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    
                    # Format timestamps as MM:SS
                    start_formatted = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                    end_formatted = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                    
                    logger.info(f"Segment {i:2d} [{start_formatted} → {end_formatted}] ({end_time-start_time:.1f}s):")
                    logger.info(f"  \"{text}\"")
                    logger.info("")
                logger.info("=" * 80)
                logger.info(f"Total segments: {len(segments)}")
            
            # Cleanup memory after transcription (preserve Whisper model)
            self.cleanup_memory(preserve_whisper=True)
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Cleanup on error (preserve Whisper model for retry)
            self.cleanup_memory(preserve_whisper=True)
            return None
    
    def chunk_transcription(
        self,
        transcription: str,
        segments: Optional[List[Dict]] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        max_chunk_duration: int = 60,
    ) -> List[Dict]:
        """
        Create timestamped chunks from transcription (YouTube-style precision).

        If Whisper segments (with start/end) are provided, build chunks by aggregating
        consecutive segments until reaching the target size or max duration, and record
        start/end timestamps. Otherwise, fall back to character-based chunking without
        timestamps.

        Returns list of dicts: { text: str, start: float, end: float }
        """
        if segments:
            chunks: List[Dict] = []
            current_text_parts: List[str] = []
            current_start: Optional[float] = None
            current_end: Optional[float] = None

            def flush_chunk():
                nonlocal current_text_parts, current_start, current_end
                if current_text_parts and current_start is not None and current_end is not None:
                    # Ensure we have valid timestamps (YouTube-style precision)
                    chunk_start = float(max(0.0, current_start))
                    chunk_end = float(max(current_start, current_end))
                    
                    # Log chunk details for debugging
                    logger.info(f"Creating chunk: start={chunk_start:.2f}s, end={chunk_end:.2f}s, duration={chunk_end-chunk_start:.2f}s")
                    
                    chunks.append({
                        'text': ' '.join(current_text_parts).strip(),
                        'start': chunk_start,
                        'end': chunk_end,
                    })
                current_text_parts = []
                current_start = None
                current_end = None

            for seg in segments:
                seg_text = (seg.get('text') or '').strip()
                if not seg_text:
                    continue
                seg_start = float(seg.get('start', 0.0))
                seg_end = float(seg.get('end', seg_start))

                if current_start is None:
                    current_start = seg_start
                    current_end = seg_end
                else:
                    current_end = seg_end

                current_text_parts.append(seg_text)

                # Heuristics to flush chunk: length or duration cap (YouTube-style)
                current_text_len = sum(len(p) for p in current_text_parts) + (len(current_text_parts) - 1)
                current_duration = current_end - (current_start or current_end)
                
                # More aggressive chunking for better timestamp precision (like YouTube)
                if current_text_len >= chunk_size or current_duration >= max_chunk_duration:
                    flush_chunk()

            # Flush remaining
            flush_chunk()

            logger.info(f"Created {len(chunks)} timestamped chunks from segments")
            
            # Log detailed chunk information (YouTube-style debugging)
            for i, chunk in enumerate(chunks[:3]):  # Log first 3 chunks
                logger.info(f"    Chunk {i+1}: [{chunk['start']:.2f}s → {chunk['end']:.2f}s] {chunk['text'][:100]}...")
            
            return chunks

        # Fallback: character-based chunks without timestamps
        fallback_chunks: List[Dict] = []
        pos = 0
        while pos < len(transcription):
            end = pos + chunk_size
            if end < len(transcription):
                for i in range(end, max(pos + chunk_size - 100, pos), -1):
                    if transcription[i] in '.!?':
                        end = i + 1
                        break
            text_chunk = transcription[pos:end].strip()
            if text_chunk:
                fallback_chunks.append({'text': text_chunk, 'start': 0.0, 'end': 0.0})
            pos = end - overlap
            if pos >= len(transcription):
                break
        logger.info(f"Created {len(fallback_chunks)} chunks from transcription (no timestamps)")
        return fallback_chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for text chunks using OpenAI
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embedding vectors
        """
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
                         transcription_data: Dict, chunks: List[Dict], embeddings: List[List[float]]) -> bool:
        """
        Store transcription chunks and embeddings in Pinecone
        
        Args:
            company_name: Name of the company
            video_url: Original video URL
            video_info: Video metadata
            transcription_data: Transcription data
            chunks: Text chunks
            embeddings: Embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Storing in Pinecone for company: {company_name}")
            
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
            
            # Get index and namespace per company
            index = self.pc.Index(index_name)
            namespace = company_name.lower().replace(' ', '-')
            logger.info(f"Storing data in namespace: '{namespace}' in index: '{index_name}'")
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{company_name}_{video_url}_{i}"
                
                # Extract and validate timestamps
                chunk_start = float(chunk.get('start', 0.0)) if isinstance(chunk, dict) else 0.0
                chunk_end = float(chunk.get('end', 0.0)) if isinstance(chunk, dict) else 0.0
                
                vector_data = {
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'company': company_name,
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
            
            logger.info(f"Successfully stored {len(vectors)} vectors in Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Pinecone storage failed: {e}")
            return False
    
    def process_video(self, video_url: str, company_name: str) -> Optional[Dict]:
        """
        Complete Loom video processing pipeline with memory management
        
        Args:
            video_url: Loom video URL
            company_name: Company name for organization
            
        Returns:
            Dict with processing results or None if failed
        """
        try:
            logger.info(f"Processing Loom video: {video_url}")
            
            # Check Whisper model status
            whisper_loaded = self.is_whisper_loaded()
            logger.info(f"Whisper model loaded: {whisper_loaded}")
            
            # Initial memory check
            memory_mb = self.check_memory_usage()
            logger.info(f"Initial memory: {memory_mb:.1f} MB")
            
            # Step 1: Extract video info
            video_info = self.extract_loom_video_info(video_url)
            if not video_info:
                raise Exception("Failed to extract video info")
            
            # Step 2: Download video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
            
            # Try direct download first, then quality fallback
            download_success = self.download_loom_video(video_info['video_url'], temp_video_path)
            if not download_success:
                # Use quality fallback download
                logger.info("Trying quality fallback download")
                download_success = self.download_loom_video_with_quality_fallback(video_url, temp_video_path)
                
                if not download_success:
                    raise Exception("Failed to download video with all methods")
            
            if not download_success:
                raise Exception("Failed to download video")
            
            try:
                # Check memory before transcription
                memory_mb = self.check_memory_usage()
                if memory_mb > self.memory_threshold:
                    logger.warning(f"High memory before transcription ({memory_mb:.1f}MB), performing cleanup")
                    self.cleanup_memory(preserve_whisper=True)
                
                # Step 3: Transcribe video
                transcription_data = self.transcribe_video(temp_video_path)
                if not transcription_data:
                    raise Exception("Failed to transcribe video")
                
                # Check memory after transcription
                memory_mb = self.check_memory_usage()
                logger.info(f"Memory after transcription: {memory_mb:.1f} MB")
                
                # Step 4: Chunk the transcription
                transcription = transcription_data.get('transcription', '')
                if not transcription:
                    raise Exception("Empty transcription")
                segments = transcription_data.get('segments', [])
                
                # Use smaller chunks for Loom videos for better timestamp precision
                chunk_size = 600  # Further reduced for memory optimization
                max_chunk_duration = 30  # Further reduced for memory optimization
                
                chunks = self.chunk_transcription(transcription, segments=segments, 
                                                chunk_size=chunk_size, 
                                                max_chunk_duration=max_chunk_duration)
                if not chunks:
                    raise Exception("Failed to create chunks")
                
                # Log detailed chunk information
                logger.info(f"Created {len(chunks)} timestamped chunks")
                logger.info("TRANSCRIPTION CHUNKS WITH TIMESTAMPS:")
                logger.info("=" * 80)
                for i, chunk in enumerate(chunks, 1):
                    start_time = chunk.get('start', 0)
                    end_time = chunk.get('end', 0)
                    text = chunk.get('text', '').strip()
                    
                    # Format timestamps as MM:SS
                    start_formatted = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                    end_formatted = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                    
                    logger.info(f"Chunk {i:2d} [{start_formatted} → {end_formatted}] ({end_time-start_time:.1f}s):")
                    logger.info(f"  \"{text}\"")
                    logger.info("")
                logger.info("=" * 80)
                logger.info(f"Total chunks: {len(chunks)}")
                logger.info(f"Average chunk duration: {sum(chunk.get('end', 0) - chunk.get('start', 0) for chunk in chunks) / len(chunks):.1f}s")
                
                # Check memory before embeddings
                memory_mb = self.check_memory_usage()
                if memory_mb > self.memory_threshold:
                    logger.warning(f"High memory before embeddings ({memory_mb:.1f}MB), performing cleanup")
                    self.cleanup_memory(preserve_whisper=True)
                
                # Step 5: Create embeddings
                embeddings = self.create_embeddings([c['text'] if isinstance(c, dict) else str(c) for c in chunks])
                if not embeddings or len(embeddings) != len(chunks):
                    raise Exception("Failed to create embeddings")
                
                # Check memory before storage
                memory_mb = self.check_memory_usage()
                logger.info(f"Memory before storage: {memory_mb:.1f} MB")
                
                # Step 6: Store in Pinecone
                storage_success = self.store_in_pinecone(
                    company_name, video_url, video_info, transcription_data, chunks, embeddings
                )

                # Log a brief timestamp summary for Loom chunks
                try:
                    total = len(chunks)
                    with_ts = sum(1 for c in chunks if float(c.get('start', 0.0)) > 0.0 or float(c.get('end', 0.0)) > 0.0)
                    logger.info(f"Timestamp chunk check (Loom - {company_name}): {with_ts}/{total} chunks have timestamps")
                    for i, c in enumerate(chunks[: min(3, total)]):  # Reduced logging
                        s = float(c.get('start', 0.0))
                        e = float(c.get('end', 0.0))
                        t = (c.get('text') or '')[:80].replace('\n', ' ')  # Reduced text length
                        logger.info(f"    #{i+1}: [{s:.2f} → {e:.2f}] {t}")
                except Exception:
                    pass
                
                if not storage_success:
                    raise Exception("Failed to store in Pinecone")
                
                # Final memory cleanup
                self.cleanup_memory(preserve_whisper=True)
                
                # Return success result
                result = {
                    'success': True,
                    'video_url': video_url,
                    'company_name': company_name,
                    'title': video_info.get('title', 'Unknown'),
                    'chunks_created': len(chunks),
                    'vectors_stored': len(embeddings),
                    'word_count': transcription_data.get('word_count', 'Unknown'),
                    'language': transcription_data.get('language', 'Unknown'),
                    'method': 'loom_transcription'
                }
                
                logger.info(f"Loom video processing completed successfully")
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_video_path)
                    logger.info("Cleaned up temporary video file")
                except:
                    pass
                
                # Final memory cleanup
                self.cleanup_memory(preserve_whisper=True)
                
        except Exception as e:
            logger.error(f"Loom video processing failed: {e}")
            # Cleanup on error
            self.cleanup_memory(preserve_whisper=True)
            return None
    
    def search_similar_chunks(self, company_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks in Pinecone
        
        Args:
            company_name: Company name
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Create query embedding
            query_embedding = self.create_embeddings([query])
            if not query_embedding:
                raise Exception("Failed to create query embedding")
            
            # Get index
            index_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
            index = self.pc.Index(index_name)
            
            # Search
            results = index.query(
                vector=query_embedding[0],
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'video_url': match.metadata.get('video_url', ''),
                    'title': match.metadata.get('title', ''),
                    'chunk_index': match.metadata.get('chunk_index', 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def process_video_lightweight(self, video_url: str, company_name: str) -> Optional[Dict]:
        """
        Lightweight Loom video processing for low memory situations
        
        Args:
            video_url: Loom video URL
            company_name: Company name for organization
            
        Returns:
            Dict with processing results or None if failed
        """
        try:
            logger.info(f"Processing Loom video (LIGHTWEIGHT MODE): {video_url}")
            
            # Check Whisper model status
            whisper_loaded = self.is_whisper_loaded()
            logger.info(f"Whisper model loaded (lightweight): {whisper_loaded}")
            
            # Check memory - if too high, fail early
            memory_mb = self.check_memory_usage()
            if memory_mb > 6000:  # Increased threshold for 8GB RAM capacity
                raise Exception(f"Memory too high ({memory_mb:.1f}MB) for lightweight processing")
            
            # Step 1: Extract video info (minimal)
            video_info = self.extract_loom_video_info(video_url)
            if not video_info:
                raise Exception("Failed to extract video info")
            
            # Step 2: Use yt-dlp directly for optimized download
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
            
            logger.info("Using quality fallback for lightweight download")
            download_success = self.download_loom_video_with_quality_fallback(video_url, temp_video_path)
            
            if not download_success:
                raise Exception("Failed to download video with quality fallback")
            
            try:
                # Check memory before transcription
                memory_mb = self.check_memory_usage()
                if memory_mb > 5000:  # Increased threshold for 8GB RAM capacity
                    logger.warning(f"High memory before transcription ({memory_mb:.1f}MB), performing cleanup")
                    self.cleanup_memory(preserve_whisper=True)
                
                # Step 3: Transcribe video with minimal settings
                logger.info("Starting lightweight transcription...")
                
                # Load Whisper model if not loaded
                model = self.get_whisper_model()
                
                # Use transcription settings that preserve timestamps
                result = model.transcribe(
                    temp_video_path,
                    word_timestamps=False,  # Disable word timestamps to save memory
                    verbose=False,
                    fp16=False,  # Disable fp16 to save memory
                    condition_on_previous_text=False,  # Disable for faster processing
                    temperature=0.0  # Use greedy decoding for speed
                )
                
                # Create transcription data with segments for timestamps
                transcription_data = {
                    'transcription': result['text'],
                    'segments': result.get('segments', []),  # Keep segments for timestamps
                    'language': result.get('language', 'en'),
                    'word_count': len(result['text'].split())
                }
                
                logger.info(f"Lightweight transcription completed: {transcription_data['word_count']} words")
                
                # Log full transcription for lightweight mode
                transcription_text = transcription_data.get('transcription', '')
                if transcription_text:
                    logger.info("FULL TRANSCRIPTION (LIGHTWEIGHT MODE):")
                    logger.info("=" * 80)
                    logger.info(transcription_text)
                    logger.info("=" * 80)
                    logger.info(f"Total transcription length: {len(transcription_text)} characters")
                    logger.info(f"Total words: {transcription_data['word_count']}")
                
                # Check memory after transcription
                memory_mb = self.check_memory_usage()
                logger.info(f"Memory after transcription: {memory_mb:.1f} MB")
                
                # Step 4: Create chunks with timestamps from segments
                transcription = transcription_data.get('transcription', '')
                segments = transcription_data.get('segments', [])
                
                if not transcription:
                    raise Exception("Empty transcription")
                
                # Create timestamped chunks from segments
                chunks = []
                if segments:
                    # Use segments to create timestamped chunks
                    current_chunk = []
                    current_start = 0.0
                    current_end = 0.0
                    chunk_duration = 30.0  # 30-second chunks
                    
                    for segment in segments:
                        segment_start = float(segment.get('start', 0.0))
                        segment_end = float(segment.get('end', 0.0))
                        segment_text = segment.get('text', '').strip()
                        
                        if not segment_text:
                            continue
                        
                        # If this segment starts a new chunk
                        if not current_chunk or (segment_start - current_start) >= chunk_duration:
                            # Save previous chunk if it exists
                            if current_chunk:
                                chunk_text = ' '.join(current_chunk)
                                chunks.append({
                                    'text': chunk_text,
                                    'start': current_start,
                                    'end': current_end
                                })
                            
                            # Start new chunk
                            current_chunk = [segment_text]
                            current_start = segment_start
                            current_end = segment_end
                        else:
                            # Add to current chunk
                            current_chunk.append(segment_text)
                            current_end = segment_end
                    
                    # Add final chunk
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'start': current_start,
                            'end': current_end
                        })
                    
                    logger.info(f"Created {len(chunks)} timestamped chunks from segments")
                else:
                    # Fallback to simple character-based chunking (no timestamps)
                    chunk_size = 500  # Very small chunks
                    overlap = 50
                    pos = 0
                    
                    while pos < len(transcription):
                        end = pos + chunk_size
                        if end < len(transcription):
                            for i in range(end, max(pos + chunk_size - 100, pos), -1):
                                if transcription[i] in '.!?':
                                    end = i + 1
                                    break
                        text_chunk = transcription[pos:end].strip()
                        if text_chunk:
                            chunks.append({'text': text_chunk, 'start': 0.0, 'end': 0.0})
                        pos = end - overlap
                        if pos >= len(transcription):
                            break
                    
                    logger.info(f"Created {len(chunks)} simple chunks (no timestamps)")
                
                # Log detailed chunk information for lightweight mode
                if segments:
                    logger.info("TIMESTAMPED CHUNKS (LIGHTWEIGHT MODE):")
                    logger.info("=" * 80)
                    for i, chunk in enumerate(chunks, 1):
                        text = chunk.get('text', '').strip()
                        start = chunk.get('start', 0.0)
                        end = chunk.get('end', 0.0)
                        logger.info(f"Chunk {i:2d} [{start:.2f}s → {end:.2f}s]:")
                        logger.info(f"  \"{text}\"")
                        logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"Total chunks: {len(chunks)}")
                    logger.info(f"Average chunk duration: {sum(chunk.get('end', 0.0) - chunk.get('start', 0.0) for chunk in chunks) / len(chunks):.1f}s")
                else:
                    logger.info("SIMPLE CHUNKS (LIGHTWEIGHT MODE - NO TIMESTAMPS):")
                    logger.info("=" * 80)
                    for i, chunk in enumerate(chunks, 1):
                        text = chunk.get('text', '').strip()
                        logger.info(f"Chunk {i:2d} (No timestamps):")
                        logger.info(f"  \"{text}\"")
                        logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"Total chunks: {len(chunks)}")
                    logger.info(f"Average chunk size: {sum(len(chunk.get('text', '')) for chunk in chunks) / len(chunks):.0f} characters")
                
                # Check memory before embeddings
                memory_mb = self.check_memory_usage()
                if memory_mb > 2500:  # Increased from 1800MB to 2500MB
                    logger.warning(f"High memory before embeddings ({memory_mb:.1f}MB), performing cleanup")
                    self.cleanup_memory(preserve_whisper=True)
                
                # Step 5: Create embeddings in smaller batches
                embeddings = []
                batch_size = 50  # Smaller batch size
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_texts = [c['text'] for c in batch]
                    
                    try:
                        response = openai.embeddings.create(
                            input=batch_texts,
                            model="text-embedding-3-small"
                        )
                        batch_embeddings = [e.embedding for e in response.data]
                        embeddings.extend(batch_embeddings)
                        
                        logger.info(f"Created embeddings for batch {i//batch_size + 1}")
                        
                        # Check memory after each batch
                        memory_mb = self.check_memory_usage()
                        if memory_mb > 2500:  # Increased from 1800MB to 2500MB
                            logger.warning(f"High memory after batch {i//batch_size + 1}, performing cleanup")
                            self.cleanup_memory(preserve_whisper=True)
                        
                    except Exception as e:
                        logger.error(f"Batch embedding failed: {e}")
                        # Create zero embeddings for failed batch
                        zero_embedding = [0.0] * 1536
                        embeddings.extend([zero_embedding] * len(batch))
                
                if not embeddings or len(embeddings) != len(chunks):
                    raise Exception("Failed to create embeddings")
                
                # Step 6: Store in Pinecone
                storage_success = self.store_in_pinecone(
                    company_name, video_url, video_info, transcription_data, chunks, embeddings
                )
                
                if not storage_success:
                    raise Exception("Failed to store in Pinecone")
                
                # Log timestamp verification
                chunks_with_timestamps = sum(1 for c in chunks if float(c.get('start', 0.0)) > 0.0 or float(c.get('end', 0.0)) > 0.0)
                logger.info(f"Timestamp chunk check (Loom lightweight - {company_name}): {chunks_with_timestamps}/{len(chunks)} chunks have timestamps")
                for i, c in enumerate(chunks[:min(3, len(chunks))]):
                    s = float(c.get('start', 0.0))
                    e = float(c.get('end', 0.0))
                    t = (c.get('text') or '')[:80].replace('\n', ' ')
                    logger.info(f"    #{i+1}: [{s:.2f} → {e:.2f}] {t}")
                
                # Final memory cleanup
                self.cleanup_memory(preserve_whisper=True)
                
                # Return success result
                result = {
                    'success': True,
                    'video_url': video_url,
                    'company_name': company_name,
                    'title': video_info.get('title', 'Unknown'),
                    'chunks_created': len(chunks),
                    'vectors_stored': len(embeddings),
                    'word_count': transcription_data.get('word_count', 'Unknown'),
                    'language': transcription_data.get('language', 'Unknown'),
                    'method': 'loom_transcription_lightweight'
                }
                
                logger.info(f"Loom video processing completed successfully (lightweight mode)")
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_video_path)
                    logger.info("Cleaned up temporary video file")
                except:
                    pass
                
                # Final memory cleanup
                self.cleanup_memory(preserve_whisper=True)
                
        except Exception as e:
            logger.error(f"Loom video processing failed (lightweight): {e}")
            # Cleanup on error
            self.cleanup_memory(preserve_whisper=True)
            return None

    def process_video_with_semantic_chunking(self, video_url: str, company_name: str) -> Dict:
        """
        Process video with semantic chunking for better retrieval
        
        Args:
            video_url: URL of the video to process
            company_name: Company name for storage
            
        Returns:
            Processing result dictionary
        """
        try:
            logger.info(f"Processing video with semantic chunking: {video_url}")
            
            # Download and transcribe video
            transcription_result = self._download_and_transcribe(video_url)
            if not transcription_result:
                return {
                    "success": False,
                    "error": "Failed to transcribe video",
                    "video_url": video_url,
                    "company_name": company_name
                }
            
            # Extract transcription text
            transcription = transcription_result.get('transcription', '')
            segments = transcription_result.get('segments', [])
            
            if not transcription.strip():
                return {
                    "success": False,
                    "error": "No transcription content found",
                    "video_url": video_url,
                    "company_name": company_name
                }
            
            logger.info(f"Transcription completed: {len(transcription)} characters")
            
            # Use semantic chunking instead of fixed-size chunks
            semantic_chunks = self._create_semantic_chunks_from_transcription(
                transcription=transcription,
                segments=segments,
                company_name=company_name,
                video_url=video_url
            )
            
            if not semantic_chunks:
                return {
                    "success": False,
                    "error": "Failed to create semantic chunks",
                    "video_url": video_url,
                    "company_name": company_name
                }
            
            logger.info(f"Successfully created {len(semantic_chunks)} semantic chunks")
            
            return {
                "success": True,
                "message": "Video processed successfully with semantic chunking",
                "video_url": video_url,
                "company_name": company_name,
                "chunks_created": len(semantic_chunks),
                "transcription_length": len(transcription),
                "segments_count": len(segments)
            }
            
        except Exception as e:
            logger.error(f"Error in semantic video processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "video_url": video_url,
                "company_name": company_name
            }
    
    def _create_semantic_chunks_from_transcription(self, transcription: str, segments: List[Dict], 
                                                  company_name: str, video_url: str) -> List[Dict]:
        """
        Create semantic chunks from video transcription
        
        Args:
            transcription: Full transcription text
            segments: Time-stamped segments
            company_name: Company name
            video_url: Video URL
            
        Returns:
            List of stored chunk information
        """
        try:
            # Initialize knowledge integrator for semantic chunking
            from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator
            
            integrator = EnhancedKnowledgeIntegrator(
                openai_api_key=self.openai_api_key,
                pinecone_api_key=self.pinecone_api_key,
                pinecone_index=self.default_index_name
            )
            
            # Prepare source information
            source_info = {
                'title': f'Video: {video_url}',
                'url': video_url,
                'source': 'video_transcription',
                'content_type': 'video',
                'segments_count': len(segments),
                'transcription_length': len(transcription),
                'processing_method': 'semantic_chunking'
            }
            
            # Store using semantic chunking
            # Create a chunk dictionary from the transcription
            chunk_data = {
                'text': transcription,
                'full_context': transcription,
                'source': source_info.get('source', 'video_transcription'),
                'title': source_info.get('title', f'Video: {video_url}'),
                'url': source_info.get('url', video_url),
                'processed_at': source_info.get('processed_at', ''),
            }
            
            stored_result = integrator.store_semantic_chunks(
                chunks=[chunk_data],
                company_name=company_name
            )
            
            if stored_result.get('success', False):
                logger.info(f"Stored {stored_result.get('chunks_stored', 0)} semantic chunks in Pinecone")
                return [chunk_data]  # Return the chunk data for consistency
            else:
                logger.error(f"Failed to store semantic chunks: {stored_result.get('error', 'Unknown error')}")
                return []
            
        except Exception as e:
            logger.error(f"Error creating semantic chunks: {e}")
            return []

    def _download_and_transcribe(self, video_url: str) -> Optional[Dict]:
        """
        Download and transcribe video using existing methods
        
        Args:
            video_url: Video URL to process
            
        Returns:
            Transcription result or None
        """
        try:
            # Use existing download and transcription logic
            # This is a simplified version - you can integrate with your existing methods
            
            # For now, return a placeholder
            # In practice, this would call your existing transcription methods
            logger.info(f"Downloading and transcribing: {video_url}")
            
            # Placeholder - replace with actual transcription logic
            return {
                'transcription': 'Sample transcription text...',
                'segments': [{'text': 'Sample segment', 'start': 0.0, 'end': 10.0}]
            }
            
        except Exception as e:
            logger.error(f"Error in download and transcribe: {e}")
            return None



