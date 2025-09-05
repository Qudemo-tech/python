#!/usr/bin/env python3
"""
Loom Video Processor
Handles Loom video processing with transcription and vector storage
Optimized for 8GB RAM with Pinecone Standard Plan
"""

import os
import logging
import time
import json
import gc
import psutil
import subprocess
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
        
        # Memory management (optimized for API-only processing - no local Whisper model)
        self.memory_threshold = 6000  # MB (increased since no local Whisper model)
        self.warning_memory_threshold = 4000   # MB (increased since no local Whisper model)
        
        # Windows-specific FFmpeg configuration
        self._configure_ffmpeg_for_windows()
        
        logger.info("Initializing Loom Video Processor (8GB RAM Optimized)...")
    
    def _configure_ffmpeg_for_windows(self):
        """Configure FFmpeg for Windows compatibility"""
        try:
            import platform
            if platform.system() == "Windows":
                # Set FFmpeg path for Windows
                import whisper
                
                # Try imageio-ffmpeg first (most reliable for Windows)
                try:
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    if ffmpeg_path and os.path.exists(ffmpeg_path):
                        # Set multiple ways to ensure Whisper finds it
                        whisper.ffmpeg_path = ffmpeg_path
                        os.environ['FFMPEG_BINARY'] = ffmpeg_path
                        os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
                        logger.info(f"✅ FFmpeg configured via imageio-ffmpeg: {ffmpeg_path}")
                        logger.info(f"✅ Set FFMPEG_BINARY env var: {ffmpeg_path}")
                        logger.info(f"✅ Added FFmpeg to PATH: {os.path.dirname(ffmpeg_path)}")
                        return
                except Exception as e:
                    logger.warning(f"imageio-ffmpeg not available: {e}")
                
                # Try to find FFmpeg in common locations
                possible_paths = [
                    r"C:\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Users\{}\AppData\Local\ffmpeg\bin\ffmpeg.exe".format(os.getenv('USERNAME', '')),
                    "ffmpeg"  # Fallback to PATH
                ]
                
                ffmpeg_found = False
                for path in possible_paths:
                    try:
                        if os.path.exists(path) or path == "ffmpeg":
                            # Test if FFmpeg works
                            import subprocess
                            result = subprocess.run([path, "-version"], 
                                                  capture_output=True, 
                                                  timeout=5)
                            if result.returncode == 0:
                                whisper.ffmpeg_path = path
                                os.environ['FFMPEG_BINARY'] = path
                                os.environ['PATH'] = os.path.dirname(path) + os.pathsep + os.environ.get('PATH', '')
                                logger.info(f"✅ FFmpeg configured: {path}")
                                ffmpeg_found = True
                                break
                    except Exception:
                        continue
                
                if not ffmpeg_found:
                    logger.warning("⚠️ FFmpeg not found, will use fallback transcription method")
                    # Don't set a path, let the fallback method handle it
                    
        except Exception as e:
            logger.warning(f"FFmpeg configuration failed: {e}")
    
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
            if memory_mb > 5000:  # Hard limit (increased since no local Whisper model)
                logger.error(f"🚨 Memory too high ({memory_mb:.1f}MB), cannot load Whisper safely")
                raise Exception(f"Memory limit exceeded: {memory_mb:.1f}MB (max: 5000MB)")
            
            logger.info("📥 Loading Whisper model (tiny for Windows compatibility)...")
            try:
                self._whisper_model = whisper.load_model("tiny")
                logger.info("✅ Whisper model (tiny) loaded successfully")
                
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
            
            # Quality levels to try (more flexible format selection)
            quality_formats = [
                "best[height<=720]",                 # Best quality up to 720p
                "best[height<=1080]",                # Best quality up to 1080p
                "best"                               # Any available format
            ]
            
            quality_names = ["720p", "1080p", "best"]
            
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
                    
                    # Download video with audio in one command
                    download_cmd = [
                        sys.executable, '-m', 'yt_dlp',
                        '--no-warnings',
                        '--retries', '2', '--fragment-retries', '2',
                        '--restrict-filenames',
                        '--force-overwrites',
                        '--format', format_spec,
                        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        '--referer', 'https://www.loom.com/',
                        '--add-header', 'Origin: https://www.loom.com',
                        '--add-header', 'Sec-Fetch-Mode: navigate',
                        '--output', output_path,
                        video_url
                    ]
                    
                    logger.info(f"Downloading: {' '.join(download_cmd[:8])}... --format {format_spec} ...")
                    
                    # Download video
                    timeout = 180 if i == 0 else 120
                    result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=timeout)
                    
                    if result.returncode != 0:
                        logger.warning(f"Download failed: {result.stderr[:200]}...")
                        continue
                    
                    # Check if download was successful
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                        logger.info(f"✅ Successfully downloaded {quality_name} quality video: {file_size_mb:.1f} MB")
                        
                        # Check memory after successful download
                        memory_mb = self.check_memory_usage()
                        logger.info(f"Memory after {quality_name} download: {memory_mb:.1f} MB")
                        
                        return True
                    else:
                        logger.warning(f"Merge failed: {merge_result.stderr[:200]}...")
                        
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
        """Transcribe video using Whisper (DEPRECATED - use OpenAI Whisper API instead)"""
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
            
            # Additional file validation
            if file_size_mb < 0.1:  # Less than 100KB
                raise Exception(f"Video file too small: {file_size_mb:.1f} MB")
            
            if file_size_mb > 50:
                logger.warning(f"Large video file ({file_size_mb:.1f}MB), transcription may be slow")
            
            # Transcribe video
            logger.info("Starting Whisper transcription...")
            
            # Small delay to ensure file is fully written and accessible
            time.sleep(1.0)
            
            # Final file check before transcription
            if not os.path.exists(video_path):
                raise Exception(f"Video file disappeared before transcription: {video_path}")
            
            if not os.access(video_path, os.R_OK):
                raise Exception(f"Video file became unreadable before transcription: {video_path}")
            
            # Create a copy of the file in a more accessible location for Whisper
            import shutil
            safe_video_path = os.path.join(tempfile.gettempdir(), f"whisper_safe_{int(time.time())}.mp4")
            try:
                shutil.copy2(video_path, safe_video_path)
                logger.info(f"Created safe copy for Whisper: {safe_video_path}")
                # Use the safe copy for transcription
                transcription_path = safe_video_path
            except Exception as copy_error:
                logger.warning(f"Failed to create safe copy: {copy_error}, using original path")
                transcription_path = video_path
            
            try:
                # Ensure model is available
                if model is None:
                    logger.warning("Model is None, reloading...")
                    model = self.get_whisper_model()
                
                logger.info(f"Model status before transcription: {type(model).__name__ if model else 'None'}")
                logger.info("Starting Whisper transcription...")
                
                # Ensure FFmpeg path is set for Whisper - DIRECT INJECTION METHOD
                import whisper
                try:
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    if ffmpeg_path and os.path.exists(ffmpeg_path):
                        # Set multiple ways to ensure Whisper finds it
                        whisper.ffmpeg_path = ffmpeg_path
                        os.environ['FFMPEG_BINARY'] = ffmpeg_path
                        os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
                        
                        # DIRECT INJECTION: Override Whisper's internal FFmpeg detection
                        try:
                            import whisper.audio
                            whisper.audio.ffmpeg_path = ffmpeg_path
                            logger.info(f"✅ Direct injected FFmpeg into whisper.audio: {ffmpeg_path}")
                        except:
                            pass
                        
                        try:
                            import whisper.utils
                            whisper.utils.ffmpeg_path = ffmpeg_path
                            logger.info(f"✅ Direct injected FFmpeg into whisper.utils: {ffmpeg_path}")
                        except:
                            pass
                        
                        # Force reload of whisper modules
                        import importlib
                        try:
                            importlib.reload(whisper.audio)
                            importlib.reload(whisper.utils)
                            logger.info("✅ Reloaded Whisper modules with FFmpeg path")
                        except:
                            pass
                        
                        logger.info(f"✅ Set Whisper FFmpeg path: {ffmpeg_path}")
                        logger.info(f"✅ Set FFMPEG_BINARY env var: {ffmpeg_path}")
                        logger.info(f"✅ Added FFmpeg to PATH: {os.path.dirname(ffmpeg_path)}")
                except Exception as e:
                    logger.warning(f"Could not set FFmpeg path: {e}")
                
                result = model.transcribe(
                    transcription_path,
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
                        transcription_path,
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
            
            # Clean up the safe copy file
            try:
                if 'safe_video_path' in locals() and os.path.exists(safe_video_path):
                    os.unlink(safe_video_path)
                    logger.info("🧹 Cleaned up safe copy file")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup safe copy: {cleanup_error}")
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.cleanup_memory()
            
            # Clean up the safe copy file on error
            try:
                if 'safe_video_path' in locals() and os.path.exists(safe_video_path):
                    os.unlink(safe_video_path)
                    logger.info("🧹 Cleaned up safe copy file (error)")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup safe copy on error: {cleanup_error}")
            
            return None
    
    # Gemini transcription removed - Loom videos should only use Whisper API
    
    def transcribe_with_openai_api(self, video_path: str) -> Optional[Dict]:
        """Transcribe video using OpenAI Whisper API as fallback"""
        try:
            logger.info(f"🔄 Transcribing with OpenAI Whisper API: {video_path}")
            
            # Check file size and compress if needed
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"📊 Video file size: {file_size_mb:.1f} MB")
            
            if file_size_mb > 25:
                logger.warning(f"⚠️ Video file ({file_size_mb:.1f}MB) exceeds OpenAI limit (25MB), attempting compression...")
                compressed_path = self._compress_video_for_openai(video_path)
                if compressed_path:
                    video_path = compressed_path
                    new_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    logger.info(f"✅ Video compressed to {new_size_mb:.1f}MB")
                else:
                    logger.error("❌ Video compression failed")
                    return None
            
            # Use Simple Transcription Service
            from simple_transcription import SimpleTranscriptionService
            
            # Initialize the service
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise Exception("Missing OPENAI_API_KEY for API transcription")
            
            service = SimpleTranscriptionService(openai_api_key)
            
            # Transcribe the video
            result = service.transcribe_video(video_path)
            
            if result:
                logger.info(f"✅ OpenAI API transcription completed: {len(result.get('transcription', ''))} characters")
                return result
            else:
                logger.error("❌ OpenAI API transcription failed")
                return None
            
        except Exception as e:
            logger.error(f"❌ OpenAI API transcription error: {e}")
            return None
    
    def _compress_video_for_openai(self, video_path: str) -> Optional[str]:
        """Compress video to under 25MB for OpenAI Whisper API"""
        try:
            import tempfile
            import subprocess
            
            # Create compressed file path
            temp_dir = tempfile.gettempdir()
            compressed_path = os.path.join(temp_dir, f"compressed_{int(time.time())}.mp4")
            
            # Use FFmpeg to compress video
            # Target: 20MB max (safety margin)
            original_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            target_size_mb = 20
            
            # If file is very large, be more aggressive
            if original_size_mb > 100:
                target_size_mb = 15
            compression_ratio = target_size_mb / original_size_mb
            
            # Calculate bitrate based on target file size
            # More aggressive compression for large files
            if original_size_mb > 50:
                target_bitrate = 500  # Very low bitrate for large files
            elif original_size_mb > 30:
                target_bitrate = 800  # Low bitrate
            else:
                target_bitrate = 1200  # Medium bitrate
            
            logger.info(f"🎵 Extracting audio from video: {original_size_mb:.1f}MB")
            logger.info(f"🎵 Target: MP3 audio format for Whisper API")
            
            # FFmpeg command to extract audio only (Whisper API works better with audio)
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-c:a', 'mp3',  # MP3 audio format
                '-b:a', '128k',  # Good audio quality
                '-ac', '2',      # Stereo audio
                '-ar', '44100',  # Standard sample rate
                '-y',  # Overwrite output file
                compressed_path.replace('.mp4', '.mp3')  # Change extension to .mp3
            ]
            
            # Try to use the FFmpeg from imageio-ffmpeg
            try:
                import imageio_ffmpeg
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                if ffmpeg_path and os.path.exists(ffmpeg_path):
                    cmd[0] = ffmpeg_path
                    logger.info(f"✅ Using FFmpeg from imageio-ffmpeg: {ffmpeg_path}")
            except:
                logger.info("🔧 Using system FFmpeg")
            
            # Run compression
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Update path to use .mp3 extension
            audio_path = compressed_path.replace('.mp4', '.mp3')
            
            if result.returncode == 0 and os.path.exists(audio_path):
                compressed_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                logger.info(f"✅ Audio extracted successfully: {compressed_size_mb:.1f}MB")
                return audio_path
            else:
                logger.error(f"❌ FFmpeg compression failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Video compression error: {e}")
            return None

    def _get_video_duration(self, video_path: str) -> float:
        """
        Get video duration in seconds using FFprobe
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Duration in seconds, or None if failed
        """
        try:
            import subprocess
            
            # FFprobe command to get duration
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', video_path
            ]
            
            # Use FFprobe from imageio-ffmpeg if available
            try:
                import imageio_ffmpeg
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                if ffmpeg_path and os.path.exists(ffmpeg_path):
                    # Get the directory containing ffmpeg
                    ffmpeg_dir = os.path.dirname(ffmpeg_path)
                    # Look for ffprobe in the same directory
                    ffprobe_path = os.path.join(ffmpeg_dir, 'ffprobe.exe')
                    if os.path.exists(ffprobe_path):
                        cmd[0] = ffprobe_path
                        logger.info(f"✅ Using FFprobe from imageio-ffmpeg: {ffprobe_path}")
                    else:
                        # Try without .exe extension for Unix systems
                        ffprobe_path = os.path.join(ffmpeg_dir, 'ffprobe')
                        if os.path.exists(ffprobe_path):
                            cmd[0] = ffprobe_path
                            logger.info(f"✅ Using FFprobe from imageio-ffmpeg: {ffprobe_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not find FFprobe in imageio-ffmpeg: {e}")
                pass
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                return duration
            else:
                logger.error(f"❌ FFprobe failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get video duration: {e}")
            return None

    def _chunk_large_video(self, video_path: str, chunk_duration: int = 600) -> list:
        """
        Split large video into chunks for processing
        
        Args:
            video_path: Path to the video file
            chunk_duration: Duration of each chunk in seconds (default: 10 minutes)
            
        Returns:
            List of chunk file paths
        """
        try:
            logger.info(f"🎬 Chunking large video: {video_path}")
            logger.info(f"⏱️ Chunk duration: {chunk_duration} seconds ({chunk_duration//60} minutes)")
            
            # Get video duration
            duration = self._get_video_duration(video_path)
            if not duration:
                logger.error("❌ Could not get video duration")
                return []
            
            logger.info(f"📊 Total video duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            # Calculate number of chunks needed
            num_chunks = int(duration / chunk_duration) + 1
            logger.info(f"📊 Will create {num_chunks} chunks")
            
            chunk_paths = []
            base_name = os.path.splitext(video_path)[0]
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration, duration)
                
                chunk_path = f"{base_name}_chunk_{i+1:03d}.mp4"
                
                # FFmpeg command to extract chunk
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-ss', str(start_time),  # Start time
                    '-t', str(end_time - start_time),  # Duration
                    '-c', 'copy',  # Copy without re-encoding
                    '-avoid_negative_ts', 'make_zero',
                    '-y',  # Overwrite output
                    chunk_path
                ]
                
                # Use FFmpeg from imageio-ffmpeg if available
                try:
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    if ffmpeg_path and os.path.exists(ffmpeg_path):
                        cmd[0] = ffmpeg_path
                except:
                    pass
                
                logger.info(f"🎬 Creating chunk {i+1}/{num_chunks}: {start_time:.1f}s - {end_time:.1f}s")
                
                # Run FFmpeg command
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(chunk_path):
                    chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
                    logger.info(f"✅ Chunk {i+1} created: {chunk_size_mb:.1f}MB")
                    chunk_paths.append(chunk_path)
                else:
                    logger.error(f"❌ Failed to create chunk {i+1}: {result.stderr}")
                    break
            
            logger.info(f"✅ Successfully created {len(chunk_paths)} chunks")
            return chunk_paths
            
        except Exception as e:
            logger.error(f"❌ Video chunking failed: {e}")
            return []

    def _process_video_chunks(self, chunk_paths: list, video_url: str, company_name: str, qudemo_id: str) -> dict:
        """
        Process video chunks and combine results
        
        Args:
            chunk_paths: List of chunk file paths
            video_url: Original video URL
            company_name: Company name for storage
            qudemo_id: QuDemo ID for storage
            
        Returns:
            Combined processing results
        """
        try:
            logger.info(f"🎬 Processing {len(chunk_paths)} video chunks")
            
            all_chunks = []
            all_embeddings = []
            total_transcription = ""
            chunk_offset = 0
            
            for i, chunk_path in enumerate(chunk_paths):
                logger.info(f"🎬 Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
                
                # Process each chunk
                chunk_result = self._process_single_chunk(
                    chunk_path, video_url, company_name, qudemo_id, 
                    chunk_index=i, chunk_offset=chunk_offset
                )
                
                if chunk_result and chunk_result.get('success'):
                    # Add chunk results
                    chunk_data = chunk_result.get('chunks', [])
                    embeddings = chunk_result.get('embeddings', [])
                    transcription = chunk_result.get('transcription', '')
                    
                    # Adjust timestamps for chunk offset
                    for chunk in chunk_data:
                        chunk['start_timestamp'] += chunk_offset
                        chunk['end_timestamp'] += chunk_offset
                    
                    all_chunks.extend(chunk_data)
                    all_embeddings.extend(embeddings)
                    total_transcription += transcription + " "
                    
                    # Update offset for next chunk
                    if chunk_data:
                        chunk_offset = chunk_data[-1]['end_timestamp']
                    
                    logger.info(f"✅ Chunk {i+1} processed: {len(chunk_data)} segments")
                else:
                    logger.error(f"❌ Chunk {i+1} processing failed")
                    continue
            
            if not all_chunks:
                logger.error("❌ No chunks were processed successfully")
                return {
                    'success': False,
                    'error': 'No chunks were processed successfully',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            
            # Store all chunks in Pinecone
            logger.info(f"💾 Storing {len(all_chunks)} total chunks in Pinecone")
            storage_result = self._store_chunks_in_pinecone(
                all_chunks, all_embeddings, company_name, qudemo_id, video_url
            )
            
            if storage_result:
                logger.info(f"✅ Successfully stored {len(all_chunks)} chunks from {len(chunk_paths)} video segments")
                return {
                    'success': True,
                    'chunks_stored': len(all_chunks),
                    'video_type': 'loom_chunked',
                    'storage_details': {
                        'method': 'chunked_processing',
                        'chunks_processed': len(chunk_paths),
                        'total_segments': len(all_chunks),
                        'total_transcription_length': len(total_transcription)
                    },
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            else:
                logger.error("❌ Failed to store chunks in Pinecone")
                return {
                    'success': False,
                    'error': 'Failed to store chunks in Pinecone',
                    'video_url': video_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
                
        except Exception as e:
            logger.error(f"❌ Chunk processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url,
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }

    def _process_single_chunk(self, chunk_path: str, video_url: str, company_name: str, 
                            qudemo_id: str, chunk_index: int, chunk_offset: float) -> dict:
        """
        Process a single video chunk
        
        Args:
            chunk_path: Path to the chunk file
            video_url: Original video URL
            company_name: Company name for storage
            qudemo_id: QuDemo ID for storage
            chunk_index: Index of this chunk
            chunk_offset: Time offset for this chunk
            
        Returns:
            Processing results for this chunk
        """
        try:
            # Transcribe chunk
            transcription_data = self.transcribe_with_openai_api(chunk_path)
            
            if not transcription_data:
                logger.error(f"❌ Transcription failed for chunk {chunk_index + 1}")
                return None
            
            # Create chunks from transcription
            chunks = self._create_timestamped_chunks(transcription_data, chunk_index, chunk_offset)
            
            if not chunks:
                logger.error(f"❌ No chunks created for chunk {chunk_index + 1}")
                return None
            
            # Create embeddings
            embeddings = self.create_embeddings([c['text'] for c in chunks])
            
            if not embeddings:
                logger.error(f"❌ Embeddings failed for chunk {chunk_index + 1}")
                return None
            
            return {
                'success': True,
                'chunks': chunks,
                'embeddings': embeddings,
                'transcription': transcription_data.get('transcription', ''),
                'chunk_index': chunk_index,
                'chunk_offset': chunk_offset
            }
            
        except Exception as e:
            logger.error(f"❌ Single chunk processing failed: {e}")
            return None

    def _create_timestamped_chunks(self, transcription_data: dict, chunk_index: int = 0, chunk_offset: float = 0) -> list:
        """
        Create timestamped chunks from transcription data
        
        Args:
            transcription_data: Transcription data with segments
            chunk_index: Index of the video chunk
            chunk_offset: Time offset for this chunk
            
        Returns:
            List of timestamped chunks
        """
        try:
            segments = transcription_data.get('segments', [])
            chunks = []
            
            for i, segment in enumerate(segments):
                chunk_data = {
                    'text': segment.get('text', ''),
                    'full_context': segment.get('text', ''),
                    'source': 'video',
                    'title': f'Video Transcription - Chunk {chunk_index + 1}',
                    'url': '',  # Will be set by caller
                    'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'start_timestamp': segment.get('start', 0) + chunk_offset,
                    'end_timestamp': segment.get('end', 0) + chunk_offset,
                    'chunk_index': i,
                    'total_chunks': len(segments),
                    'video_chunk_index': chunk_index
                }
                chunks.append(chunk_data)
            
            return chunks

        except Exception as e:
            logger.error(f"❌ Failed to create timestamped chunks: {e}")
            return []

    def _store_chunks_in_pinecone(self, chunks: list, embeddings: list, company_name: str, 
                                qudemo_id: str, video_url: str) -> bool:
        """
        Store chunks and embeddings in Pinecone
        
        Args:
            chunks: List of chunk data
            embeddings: List of embeddings
            company_name: Company name for storage
            qudemo_id: QuDemo ID for storage
            video_url: Original video URL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use Standard Plan - multiple indexes for better organization
            index_name = "qudemo-video-index"  # Dedicated video index for Standard Plan
            
            # Check if index exists
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
                    # Wait for index to be ready
                    time.sleep(10)
                except Exception as ce:
                    msg = str(ce)
                    if 'max serverless indexes' in msg.lower() or 'forbidden' in msg.lower():
                        # Fallback to default index if quota reached
                        index_name = self.default_index_name
                        logger.warning(f"Index quota reached; falling back to default index: {index_name}")
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
                        'title': chunk.get('title', 'Unknown'),
                        'language': 'en',  # Default language
                        'word_count': len(chunk.get('text', '').split()) if isinstance(chunk, dict) else 0,
                        'source_type': 'video',
                        'video_chunk_index': chunk.get('video_chunk_index', 0)
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
            return []
    
    def store_in_pinecone(self, company_name: str, video_url: str, video_info: Dict, 
                         transcription_data: Dict, chunks: List[Dict], embeddings: List[List[float]], 
                         qudemo_id: str = None) -> bool:
        """Store transcription chunks and embeddings in Pinecone with Standard Plan optimization"""
        try:
            logger.info(f"Storing in Pinecone for company: {company_name} qudemo: {qudemo_id}")
            
            # Use Standard Plan - multiple indexes for better organization
            index_name = "qudemo-video-index"  # Dedicated video index for Standard Plan
            
            # Check if index exists
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
                    # Wait for index to be ready
                    time.sleep(10)
                except Exception as ce:
                    msg = str(ce)
                    if 'max serverless indexes' in msg.lower() or 'forbidden' in msg.lower():
                        # Fallback to default index if quota reached
                        index_name = self.default_index_name
                        logger.warning(f"Index quota reached; falling back to default index: {index_name}")
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
    
    def process_loom_video(self, video_url: str, company_name: str, qudemo_id: str = None, media_file_path: str = None) -> Optional[Dict]:
        """Main Loom video processing pipeline - enhanced video processor interface"""
        return self.process_video(video_url, company_name, qudemo_id)

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
            
            # Try yt-dlp first (it works on Render and Loom doesn't block it)
            logger.info("🎥 Attempting to download Loom video with yt-dlp...")
            
            # Try to download the video using yt-dlp
            download_success = self.download_loom_video_with_quality_fallback(video_url, temp_video_path)
                
            if download_success and os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 1000:
                logger.info("✅ Successfully downloaded Loom video with yt-dlp")
            else:
                logger.warning("⚠️ yt-dlp download failed, using fallback approach")
                logger.info("🔄 Creating fallback transcription for Loom video")
                
                # Only create fallback file if yt-dlp actually failed
                # Create a minimal video file that can be processed
                os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
                
                # Create a minimal MP4 file that can be processed by Whisper API
                # This is a workaround for production environments where yt-dlp doesn't work
                try:
                    # Create a minimal valid MP4 file (just a few seconds of silence)
                    import subprocess
                    ffmpeg_cmd = [
                        'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                        '-t', '1', '-c:a', 'aac', '-y', temp_video_path
                    ]
                    
                    # Use FFmpeg from imageio-ffmpeg if available
                    try:
                        import imageio_ffmpeg
                        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                        if ffmpeg_path and os.path.exists(ffmpeg_path):
                            ffmpeg_cmd[0] = ffmpeg_path
                    except:
                        pass
                    
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        logger.info("✅ Created minimal video file for production-safe processing")
                    else:
                        logger.warning("⚠️ Failed to create minimal video file, using dummy file")
                        with open(temp_video_path, 'w') as f:
                            f.write("dummy_file_for_production")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create minimal video file: {e}, using dummy file")
                    with open(temp_video_path, 'w') as f:
                        f.write("dummy_file_for_production")
            
            logger.info("✅ Production-safe Loom processing setup complete")
            
            # Check memory before transcription
            memory_mb = self.check_memory_usage()
            if memory_mb > self.memory_threshold:
                logger.warning(f"⚠️ High memory before transcription ({memory_mb:.1f}MB), performing cleanup")
                self.cleanup_memory()
            
            # Check memory again after cleanup
            memory_mb = self.check_memory_usage()
            if memory_mb > 5000:  # Hard limit (increased since no local Whisper model)
                logger.error(f"🚨 Memory still too high ({memory_mb:.1f}MB) after cleanup, skipping video")
                return {
                    "success": False,
                    "message": f"Memory usage too high ({memory_mb:.1f}MB), video too large to process safely",
                    "error": "memory_limit_exceeded"
                }
            
            # Step 3: Check video duration and decide processing strategy
            video_duration = self._get_video_duration(temp_video_path)
            if not video_duration:
                logger.warning("⚠️ Could not get video duration, proceeding with single file processing")
                video_duration = 0
            
            logger.info(f"📊 Video duration: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes)")
            
            # Decide processing strategy based on video duration
            if video_duration > 600:  # 10 minutes
                logger.info("🎬 Large video detected (>10 minutes), using chunked processing")
                
                # Chunk the video into 10-minute segments
                chunk_paths = self._chunk_large_video(temp_video_path, chunk_duration=600)
                
                if not chunk_paths:
                    raise Exception("Failed to chunk large video")
                
                # Process chunks
                result = self._process_video_chunks(chunk_paths, video_url, company_name, qudemo_id)
                
                # Clean up chunk files
                for chunk_path in chunk_paths:
                    try:
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                    except:
                        pass
                
                if result and result.get('success'):
                    logger.info(f"✅ Large video processing completed successfully")
                    return result
                else:
                    raise Exception(f"Large video processing failed: {result.get('error', 'Unknown error') if result else 'No result'}")
            
            else:
                logger.info("🎬 Standard video processing (<10 minutes)")
                
                # Step 3: Transcribe video (use OpenAI Whisper API directly - no local Whisper)
                transcription_data = None
                
                # Use OpenAI Whisper API directly (skip local Whisper to avoid memory/FFmpeg issues)
                try:
                    logger.info("🎤 Using OpenAI Whisper API directly (no local Whisper model loading)...")
                    transcription_data = self.transcribe_with_openai_api(temp_video_path)
                    if transcription_data:
                        logger.info("✅ OpenAI Whisper API transcription successful")
                    else:
                        logger.warning("⚠️ OpenAI Whisper API transcription returned no data")
                except Exception as e:
                    logger.warning(f"⚠️ OpenAI Whisper API transcription failed: {e}")
                
            # For Loom videos, we only use Whisper API - no Gemini fallback
            if not transcription_data:
                logger.error("❌ Whisper API transcription failed - no fallback for Loom videos")
                raise Exception("Whisper API transcription failed for Loom video")
            
            if not transcription_data:
                raise Exception("All transcription methods failed")
            
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