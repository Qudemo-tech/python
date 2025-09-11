#!/usr/bin/env python3
"""
GCS-based Loom Video Processor
Replaces Pinecone with Google Cloud Storage for better scalability
Uses OpenAI Whisper API for transcription (no local whisper model)
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
import openai
from urllib.parse import urlparse
from google_cloud_storage_service import GoogleCloudStorageService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoomVideoProcessorGCS:
    def __init__(self, openai_api_key: str, gcs_bucket_name: str = None):
        """Initialize GCS-based Loom Video Processor"""
        self.openai_api_key = openai_api_key
        
        # Initialize GCS service
        self.gcs_service = GoogleCloudStorageService(
            bucket_name=gcs_bucket_name,
            service_account_path='service-account-key.json'
        )
        
        # Initialize OpenAI client for Whisper API
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Memory management (optimized for API-only processing)
        self.memory_threshold = 6000  # MB
        self.warning_memory_threshold = 4000   # MB
        
        # Windows-specific FFmpeg configuration
        self._configure_ffmpeg_for_windows()
        
        logger.info("Initializing GCS-based Loom Video Processor (8GB RAM Optimized)...")
        logger.info("✅ Using OpenAI Whisper API (no local whisper model)")

    def _configure_ffmpeg_for_windows(self):
        """Configure FFmpeg for Windows compatibility"""
        try:
            # Try imageio-ffmpeg first (most reliable for Windows)
            try:
                import imageio_ffmpeg
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                if ffmpeg_path and os.path.exists(ffmpeg_path):
                    # Set FFmpeg path for video processing
                    os.environ['FFMPEG_BINARY'] = ffmpeg_path
                    os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
                    logger.info(f"✅ FFmpeg configured via imageio-ffmpeg: {ffmpeg_path}")
                    logger.info(f"✅ Set FFMPEG_BINARY env var: {ffmpeg_path}")
                    logger.info(f"✅ Added FFmpeg to PATH: {os.path.dirname(ffmpeg_path)}")
                    return
            except ImportError:
                logger.warning("⚠️ imageio-ffmpeg not available, trying other methods")
            
            # Try common FFmpeg locations
            possible_paths = [
                "ffmpeg.exe",
                "C:\\ffmpeg\\bin\\ffmpeg.exe",
                "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
                "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe"
            ]
            
            ffmpeg_found = False
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        # Test if FFmpeg works
                        import subprocess
                        result = subprocess.run([path, "-version"], 
                                              capture_output=True, 
                                              timeout=5)
                        if result.returncode == 0:
                            os.environ['FFMPEG_BINARY'] = path
                            os.environ['PATH'] = os.path.dirname(path) + os.pathsep + os.environ.get('PATH', '')
                            logger.info(f"✅ FFmpeg configured: {path}")
                            ffmpeg_found = True
                            break
                except Exception:
                    continue
            
            if not ffmpeg_found:
                logger.warning("⚠️ FFmpeg not found - video processing may fail")
                logger.info("💡 Install FFmpeg or imageio-ffmpeg for video processing")
                
        except Exception as e:
            logger.error(f"❌ FFmpeg configuration failed: {e}")

    def check_memory_usage(self) -> float:
        """Check current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return 0.0
    
    def cleanup_memory(self):
        """Clean up memory (API-only processing)"""
        try:
            logger.info("🧹 Performing memory cleanup...")
            gc.collect()
            
            memory_after = self.check_memory_usage()
            logger.info(f"🧹 Memory cleanup completed: {memory_after:.1f}MB")
                
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            # No local models to clean up
            pass
        except:
            pass

    def extract_loom_video_info(self, loom_url: str) -> Optional[Dict]:
        """Extract video information from Loom URL"""
        try:
            logger.info(f"Extracting Loom video info from: {loom_url}")
            
            # Parse the Loom URL to extract video ID
            parsed_url = urlparse(loom_url)
            if 'loom.com' not in parsed_url.netloc:
                raise ValueError("Invalid Loom URL")
            
            # Extract video ID from URL path
            video_id = parsed_url.path.strip('/').split('/')[-1]
            if not video_id:
                raise ValueError("Could not extract video ID from Loom URL")
            
            # Create video info
            video_info = {
                'id': video_id,
                'url': loom_url,
                'title': f"Loom Video {video_id}",
                'duration': 'Unknown',
                'platform': 'loom'
            }
            
            logger.info(f"✅ Extracted Loom video info: {video_info}")
            return video_info
            
        except Exception as e:
            logger.error(f"❌ Failed to extract Loom video info: {e}")
            return None

    def download_loom_video(self, loom_url: str, output_path: str) -> bool:
        """Download Loom video using yt-dlp"""
        try:
            logger.info(f"📥 Downloading Loom video: {loom_url}")
            
            # Use yt-dlp to download the video
            cmd = [
                'yt-dlp',
                '--output', output_path,
                '--format', 'best[height<=720]',  # Limit to 720p for smaller files
                '--no-playlist',
                loom_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    logger.info(f"✅ Loom video downloaded successfully: {file_size:.1f}MB")
                    return True
                else:
                    logger.error("❌ Video file not found after download")
                    return False
            else:
                logger.error(f"❌ Download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Download timeout (5 minutes)")
            return False
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False

    def transcribe_with_whisper_api(self, video_path: str) -> Optional[Dict]:
        """Transcribe video using OpenAI Whisper API"""
        try:
            logger.info(f"🔄 Transcribing with OpenAI Whisper API: {video_path}")
            
            # Check if file exists and is not too large
            if not os.path.exists(video_path):
                logger.error(f"❌ Video file not found: {video_path}")
                return None
            
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            if file_size > 25:
                logger.warning(f"⚠️ File too large for Whisper API: {file_size:.1f}MB")
                # Compress the video first
                compressed_path = self.compress_video_for_whisper_api(video_path)
                if compressed_path:
                    video_path = compressed_path
                else:
                    logger.error("❌ Failed to compress video for Whisper API")
                    return None
            
            # Transcribe using OpenAI Whisper API
            with open(video_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            # Process the transcript
            if hasattr(transcript, 'segments') and transcript.segments:
                segments = []
                for segment in transcript.segments:
                    segments.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip()
                    })
                
                result = {
                    'text': transcript.text,
                    'language': getattr(transcript, 'language', 'en'),
                    'segments': segments,
                    'duration': getattr(transcript, 'duration', 0)
                }
                
                logger.info(f"✅ Whisper API transcription successful: {len(segments)} segments")
                return result
            else:
                logger.warning("⚠️ Whisper API returned no segments")
                return None
                
        except Exception as e:
            logger.error(f"❌ Whisper API transcription failed: {e}")
            return None

    def compress_video_for_whisper_api(self, video_path: str) -> Optional[str]:
        """Compress video to under 25MB for OpenAI Whisper API"""
        try:
            logger.info(f"🗜️ Compressing video for Whisper API: {video_path}")
            
            # Create compressed output path
            base_name = os.path.splitext(video_path)[0]
            compressed_path = f"{base_name}_compressed.mp3"
            
            # Check if FFmpeg is available
            ffmpeg_cmd = os.environ.get('FFMPEG_BINARY', 'ffmpeg')
            
            # FFmpeg command to extract audio only (Whisper API works better with audio)
            cmd = [
                ffmpeg_cmd,
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'mp3',
                '-ab', '64k',  # Low bitrate for smaller file
                '-ar', '16000',  # Sample rate for Whisper
                '-ac', '1',  # Mono audio
                '-y',  # Overwrite output file
                compressed_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(compressed_path):
                file_size = os.path.getsize(compressed_path) / (1024 * 1024)  # MB
                logger.info(f"✅ Video compressed successfully: {file_size:.1f}MB")
                return compressed_path
            else:
                logger.error(f"❌ Video compression failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Video compression failed: {e}")
            return None

    def process_loom_video(self, loom_url: str, company_name: str, qudemo_id: str) -> Optional[Dict]:
        """Process Loom video and store in GCS"""
        try:
            logger.info(f"🎬 Processing Loom video: {loom_url}")
            
            # Check memory before processing
            memory_mb = self.check_memory_usage()
            if memory_mb > self.warning_memory_threshold:
                logger.warning(f"⚠️ High memory usage: {memory_mb:.1f}MB")
                self.cleanup_memory()
            
            # Step 1: Extract video info
            video_info = self.extract_loom_video_info(loom_url)
            if not video_info:
                logger.error("❌ Failed to extract video info")
                return None
            
            # Step 2: Download video
            temp_dir = tempfile.mkdtemp()
            video_filename = f"loom_{video_info['id']}.mp4"
            video_path = os.path.join(temp_dir, video_filename)
            
            if not self.download_loom_video(loom_url, video_path):
                logger.error("❌ Failed to download Loom video")
                return None
            
            # Step 3: Transcribe using OpenAI Whisper API
            logger.info("🎤 Using OpenAI Whisper API for transcription...")
            transcription_result = self.transcribe_with_whisper_api(video_path)
            
            if not transcription_result:
                logger.error("❌ Whisper API transcription failed")
                return None
            
            # Step 4: Store in GCS
            storage_result = self.gcs_service.store_transcript(
                company_name=company_name,
                qudemo_id=qudemo_id,
                transcript_data=transcription_result,
                video_info=video_info
            )
            
            if storage_result:
                logger.info("✅ Loom video processed and stored successfully in GCS")
                
                # Clean up temporary files
                try:
                    os.remove(video_path)
                    os.rmdir(temp_dir)
                except:
                    pass
                
                return {
                    'success': True,
                    'video_url': loom_url,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'segments': len(transcription_result.get('segments', [])),
                    'transcription_method': 'openai_whisper_api',
                    'storage_method': 'gcs'
                }
            else:
                logger.error("❌ Failed to store transcript in GCS")
                return None
                
        except Exception as e:
            logger.error(f"❌ Loom video processing failed: {e}")
            return None
        finally:
            # Clean up memory
            self.cleanup_memory()

    def process_video(self, video_url: str, company_name: str, qudemo_id: str) -> Optional[Dict]:
        """Main video processing method"""
        return self.process_loom_video(video_url, company_name, qudemo_id)