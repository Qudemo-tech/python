#!/usr/bin/env python3
"""
Loom Video Processor - Replace YouTube with Loom API
"""
import os
import tempfile
import logging
import whisper
import requests
from typing import List, Dict, Optional
import psutil
from dotenv import load_dotenv
import re
import json

# Load environment variables
load_dotenv()

# Import Render configuration if available
try:
    from render_deployment_config import get_render_optimized_settings
    RENDER_CONFIG = get_render_optimized_settings()
except ImportError:
    # Fallback configuration for development
    RENDER_CONFIG = {
        'max_memory_mb': 400,
        'ytdlp_format': 'best[height<=720]',
        'ytdlp_max_filesize': '200M',
        'whisper_model': 'tiny',
        'embedding_batch_size': 5,
        'prefer_ytdlp': True,
    }

logger = logging.getLogger(__name__)

class LoomVideoProcessor:
    def __init__(self, max_memory_mb: int = None):
        """
        Initialize Loom video processor
        
        Args:
            max_memory_mb: Maximum memory usage in MB (uses Render config if None)
        """
        self.max_memory_mb = max_memory_mb or RENDER_CONFIG['max_memory_mb']
        self.loom_api_key = os.getenv('LOOM_API_KEY')
        self.loom_api_base = "https://www.loom.com/api/v1"
        
        # Use Render-optimized settings
        self.ytdlp_format = RENDER_CONFIG['ytdlp_format']
        self.ytdlp_max_filesize = RENDER_CONFIG['ytdlp_max_filesize']
        self.whisper_model = RENDER_CONFIG['whisper_model']
        self.prefer_ytdlp = RENDER_CONFIG['prefer_ytdlp']
        
        logger.info(f"🎬 LoomVideoProcessor initialized with:")
        logger.info(f"   Memory limit: {self.max_memory_mb}MB")
        logger.info(f"   yt-dlp format: {self.ytdlp_format}")
        logger.info(f"   yt-dlp max filesize: {self.ytdlp_max_filesize}")
        logger.info(f"   Whisper model: {self.whisper_model}")
        logger.info(f"   Prefer yt-dlp: {self.prefer_ytdlp}")
    
    def log_memory(self, stage: str):
        """Log current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"💾 {stage} - Memory: {memory_mb:.1f}MB")
        return memory_mb
    
    def extract_loom_id(self, loom_url: str) -> Optional[str]:
        """Extract Loom video ID from various URL formats"""
        # Handle different Loom URL formats
        patterns = [
            r'loom\.com/share/([a-zA-Z0-9-]+)',  # loom.com/share/VIDEO_ID
            r'loom\.com/embed/([a-zA-Z0-9-]+)',  # loom.com/embed/VIDEO_ID
            r'loom\.com/recordings/([a-zA-Z0-9-]+)',  # loom.com/recordings/VIDEO_ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, loom_url)
            if match:
                return match.group(1)
        
        # If it's already just an ID
        if re.match(r'^[a-zA-Z0-9-]+$', loom_url):
            return loom_url
            
        return None
    
    def get_loom_video_info(self, loom_id: str) -> Dict:
        """Get video information from Loom API"""
        if not self.loom_api_key:
            raise Exception("LOOM_API_KEY environment variable is required")
        
        headers = {
            'Authorization': f'Bearer {self.loom_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Try different API endpoints
        api_endpoints = [
            f"{self.loom_api_base}/recordings/{loom_id}",
            f"https://www.loom.com/api/v1/recordings/{loom_id}",
            f"https://api.loom.com/v1/recordings/{loom_id}"
        ]
        
        for endpoint in api_endpoints:
            try:
                logger.info(f"🔍 Trying Loom API endpoint: {endpoint}")
                response = requests.get(endpoint, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    video_info = response.json()
                    logger.info(f"📹 Loom video info: {video_info.get('title', 'Untitled')}")
                    return video_info
                elif response.status_code == 401:
                    logger.error(f"❌ Loom API authentication failed for {endpoint}")
                    continue
                else:
                    logger.warning(f"⚠️ Loom API returned status {response.status_code} for {endpoint}")
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Failed to access {endpoint}: {e}")
                continue
        
        # If all endpoints fail, try a fallback approach
        logger.info("🔄 Trying fallback approach - direct video download")
        return {"id": loom_id, "title": f"Loom Video {loom_id}", "fallback": True}
    
    def get_loom_video_url(self, loom_id: str) -> str:
        """Get direct video download URL from Loom"""
        if not self.loom_api_key:
            raise Exception("LOOM_API_KEY environment variable is required")
        
        headers = {
            'Authorization': f'Bearer {self.loom_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Try different download endpoints
        download_endpoints = [
            f"{self.loom_api_base}/recordings/{loom_id}/download",
            f"https://www.loom.com/api/v1/recordings/{loom_id}/download",
            f"https://api.loom.com/v1/recordings/{loom_id}/download"
        ]
        
        for endpoint in download_endpoints:
            try:
                logger.info(f"🔍 Trying download endpoint: {endpoint}")
                response = requests.get(endpoint, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    download_info = response.json()
                    video_url = download_info.get('url')
                    
                    if video_url:
                        logger.info(f"📥 Loom video download URL obtained")
                        return video_url
                    else:
                        logger.warning(f"⚠️ No download URL in response from {endpoint}")
                        continue
                elif response.status_code == 401:
                    logger.error(f"❌ Loom API authentication failed for {endpoint}")
                    continue
                else:
                    logger.warning(f"⚠️ Download endpoint returned status {response.status_code} for {endpoint}")
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Failed to access download endpoint {endpoint}: {e}")
                continue
        
        # Fallback: try to extract video URL from Loom share page
        logger.info("🔄 Trying to extract video URL from Loom share page")
        try:
            share_url = f"https://www.loom.com/share/{loom_id}"
            logger.info(f"🔍 Fetching Loom share page: {share_url}")
            
            # Get the share page content with browser-like headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            response = requests.get(share_url, headers=headers, timeout=30)
            if response.status_code == 200:
                content = response.text
                
                # Look for video URL patterns in the page
                import re
                
                # Pattern 1: Look for CDN video URLs
                cdn_patterns = [
                    r'https://cdn\.loom\.com/sessions/recordings/[^"\s]+\.mp4',
                    r'https://cdn\.loom\.com/sessions/recordings/[^"\s]+/with-play\.mp4',
                    r'"url":"(https://cdn\.loom\.com/sessions/recordings/[^"]+\.mp4)"',
                    r'"videoUrl":"(https://cdn\.loom\.com/sessions/recordings/[^"]+\.mp4)"'
                ]
                
                for pattern in cdn_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        video_url = matches[0].replace('\\u002F', '/')
                        logger.info(f"✅ Found video URL in page: {video_url}")
                        return video_url
                
                # Pattern 2: Look for any MP4 URLs
                mp4_pattern = r'https://[^"\s]+\.mp4'
                matches = re.findall(mp4_pattern, content)
                if matches:
                    for match in matches:
                        if 'loom.com' in match or 'cdn.loom.com' in match:
                            video_url = match.replace('\\u002F', '/')
                            logger.info(f"✅ Found MP4 URL in page: {video_url}")
                            return video_url
                
                logger.warning("⚠️ No video URL found in Loom share page")
            else:
                logger.warning(f"⚠️ Failed to fetch Loom share page: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract video URL from share page: {e}")
        
        # Final fallback: try multiple CDN URL patterns
        logger.info("🔄 Trying multiple CDN URL patterns")
        
        # Try different CDN URL patterns
        cdn_patterns = [
            f"https://cdn.loom.com/sessions/recordings/{loom_id}.mp4",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4?t=0",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4?t=0&s=0",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4?t=0&s=0&e=0",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4?t=0&s=0&e=0&f=mp4",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4?t=0&s=0&e=0&f=mp4&v=1",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4?t=0&s=0&e=0&f=mp4&v=1&q=high",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4?t=0&s=0&e=0&f=mp4&v=1&q=high&r=1",
            f"https://cdn.loom.com/sessions/recordings/{loom_id}/with-play.mp4?t=0&s=0&e=0&f=mp4&v=1&q=high&r=1&w=1920&h=1080",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Range': 'bytes=0-1024',  # Just check first 1KB
        }
        
        for i, cdn_url in enumerate(cdn_patterns):
            try:
                logger.info(f"🔍 Testing CDN pattern {i+1}: {cdn_url}")
                response = requests.head(cdn_url, headers=headers, timeout=10)
                
                if response.status_code == 200 or response.status_code == 206:
                    logger.info(f"✅ CDN URL works: {cdn_url}")
                    return cdn_url
                elif response.status_code == 403:
                    logger.warning(f"⚠️ CDN URL forbidden (403): {cdn_url}")
                    continue
                elif response.status_code == 404:
                    logger.warning(f"⚠️ CDN URL not found (404): {cdn_url}")
                    continue
                else:
                    logger.warning(f"⚠️ CDN URL returned {response.status_code}: {cdn_url}")
                    continue
                    
            except Exception as e:
                logger.warning(f"⚠️ CDN URL failed: {e}")
                continue
        
        # If all CDN patterns fail, try a different approach
        logger.info("🔄 All CDN patterns failed, trying alternative approach")
        
        # Try to construct a different URL pattern
        alternative_urls = [
            f"https://www.loom.com/share/{loom_id}",
            f"https://www.loom.com/embed/{loom_id}",
            f"https://www.loom.com/recordings/{loom_id}"
        ]
        
        for alt_url in alternative_urls:
            try:
                logger.info(f"🔍 Testing alternative URL: {alt_url}")
                response = requests.head(alt_url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Alternative URL works: {alt_url}")
                    return alt_url
            except:
                continue
        
        # If we get here, we couldn't find a direct video URL
        # Don't return HTML page URLs as they will cause FFmpeg to fail
        logger.warning("⚠️ Could not find direct video URL, will rely on yt-dlp fallback")
        raise Exception("Could not obtain Loom video download URL from any endpoint")
    
    def validate_video_file(self, file_path: str) -> bool:
        """Validate that the downloaded file is actually a video file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"❌ File does not exist: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # Less than 1KB
                logger.error(f"❌ File too small ({file_size} bytes): {file_path}")
                return False
            
            # Check file extension
            if not file_path.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv')):
                logger.warning(f"⚠️ File doesn't have video extension: {file_path}")
            
            # Try to read the first few bytes to check if it's a video file
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
                # Check for common video file signatures
                video_signatures = [
                    b'\x00\x00\x00\x20ftyp',  # MP4
                    b'\x00\x00\x00\x18ftyp',  # MP4
                    b'\x00\x00\x00\x1cftyp',  # MP4
                    b'\x1a\x45\xdf\xa3',      # WebM
                    b'RIFF',                  # AVI
                    b'\x00\x00\x00\x14ftyp',  # MP4
                ]
                
                for sig in video_signatures:
                    if header.startswith(sig):
                        logger.info(f"✅ Valid video file signature detected: {file_path}")
                        return True
                
                # If no signature matches, check if it's HTML (which would cause FFmpeg to fail)
                if b'<!DOCTYPE' in header or b'<html' in header.lower():
                    logger.error(f"❌ File appears to be HTML, not video: {file_path}")
                    return False
                
                # If we can't determine, log a warning but continue
                logger.warning(f"⚠️ Could not determine file type from header: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error validating file {file_path}: {e}")
            return False
    
    def download_loom_video(self, loom_url: str, output_filename: str) -> str:
        """Download Loom video using yt-dlp as primary method with API fallback"""
        self.log_memory("Before Loom download")
        
        # Extract Loom ID from URL
        loom_id = self.extract_loom_id(loom_url)
        if not loom_id:
            raise Exception(f"Invalid Loom URL format: {loom_url}")
        
        logger.info(f"🎬 Processing Loom video ID: {loom_id}")
        
        # Primary method: Use yt-dlp (which has excellent Loom support)
        logger.info("🔄 Using yt-dlp as primary download method for Loom video")
        try:
            import yt_dlp
            
            ydl_opts = {
                'format': self.ytdlp_format,  # Use Render-optimized format
                'outtmpl': output_filename,
                'quiet': True,
                'max_filesize': self.ytdlp_max_filesize,  # Use Render-optimized max filesize
                'extract_flat': False,
                'no_warnings': True,
                'ignoreerrors': False,
                'nocheckcertificate': True,  # Sometimes helps with SSL issues
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                }
            }
            
            logger.info(f"🔍 Downloading with yt-dlp: {loom_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First, try to extract info to see if it's accessible
                try:
                    info = ydl.extract_info(loom_url, download=False)
                    logger.info(f"✅ Video info extracted: {info.get('title', 'Unknown')} - Duration: {info.get('duration', 'Unknown')}s")
                except Exception as info_error:
                    logger.warning(f"⚠️ Could not extract video info: {info_error}")
                
                # Now download the video
                ydl.download([loom_url])
            
            if os.path.exists(output_filename):
                # Verify the file is actually a video
                file_size = os.path.getsize(output_filename)
                if file_size > 1024:  # At least 1KB
                    # Validate that it's actually a video file
                    if self.validate_video_file(output_filename):
                        logger.info(f"✅ Successfully downloaded using yt-dlp: {output_filename} ({file_size} bytes)")
                        self.log_memory("After yt-dlp download")
                        return output_filename
                    else:
                        logger.warning(f"⚠️ Downloaded file failed validation, removing: {output_filename}")
                        os.remove(output_filename)  # Remove invalid file
                else:
                    logger.warning(f"⚠️ Downloaded file too small ({file_size} bytes), may be invalid")
                    os.remove(output_filename)  # Remove invalid file
            else:
                logger.warning("⚠️ yt-dlp download failed - file not created")
                
        except Exception as yt_error:
            logger.warning(f"⚠️ yt-dlp download failed: {yt_error}")
        
        # Fallback method: Try API approach (if API key is available and not on Render)
        if self.loom_api_key and not self.prefer_ytdlp:
            try:
                logger.info("🔄 Trying API approach as fallback")
                # Get video info first
                video_info = self.get_loom_video_info(loom_id)
                
                # Get direct download URL
                video_url = self.get_loom_video_url(loom_id)
                
                # Download the video file
                logger.info(f"📥 Downloading Loom video via API: {video_url}")
                
                r = requests.get(video_url, stream=True, timeout=60)
                r.raise_for_status()
                
                # Check file size
                content_length = r.headers.get('content-length')
                if content_length:
                    file_size_mb = int(content_length) / (1024 * 1024)
                    if file_size_mb > 100:  # Allow up to 100MB for Loom videos
                        logger.warning(f"Large Loom video detected: {file_size_mb:.1f}MB")
                
                with open(output_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            # Check memory during download
                            if self.log_memory("Downloading Loom") > self.max_memory_mb:
                                raise Exception("Memory limit exceeded during Loom download")
                
                self.log_memory("After Loom download")
                
                # Validate the downloaded file
                if self.validate_video_file(output_filename):
                    return output_filename
                else:
                    logger.warning(f"⚠️ API downloaded file failed validation, removing: {output_filename}")
                    os.remove(output_filename)
                    raise Exception("Downloaded file is not a valid video")
                
            except Exception as e:
                logger.warning(f"⚠️ API approach failed: {e}")
        
        # Final fallback: Try direct download from Loom CDN (only if not on Render)
        if not self.prefer_ytdlp:
            logger.info("🔄 Using final fallback - direct CDN download")
            
            # Get the video URL first
            try:
                video_url = self.get_loom_video_url(loom_id)
                
                if not video_url:
                    raise Exception("Could not obtain video URL")
                
                logger.info(f"🔍 Downloading from URL: {video_url}")
                
                # Set up headers for video download
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }
                
                r = requests.get(video_url, headers=headers, stream=True, timeout=60)
                
                if r.status_code == 200:
                    logger.info(f"✅ Successfully downloading from: {video_url}")
                    
                    # Check file size
                    content_length = r.headers.get('content-length')
                    if content_length:
                        file_size_mb = int(content_length) / (1024 * 1024)
                        if file_size_mb > 100:
                            logger.warning(f"Large Loom video detected: {file_size_mb:.1f}MB")
                    
                    with open(output_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                # Check memory during download
                                if self.log_memory("Downloading Loom") > self.max_memory_mb:
                                    raise Exception("Memory limit exceeded during Loom download")
                    
                    self.log_memory("After Loom download")
                    
                    # Validate the downloaded file
                    if self.validate_video_file(output_filename):
                        return output_filename
                    else:
                        logger.warning(f"⚠️ Fallback downloaded file failed validation, removing: {output_filename}")
                        os.remove(output_filename)
                        raise Exception("Downloaded file is not a valid video")
                else:
                    logger.error(f"❌ Download failed with status {r.status_code}: {video_url}")
                    raise Exception(f"Download failed with status {r.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Final fallback failed: {e}")
        
        # If we get here, all methods failed
        raise Exception(f"Could not download Loom video from any available source. yt-dlp failed and fallbacks are disabled on Render.")
    
    def transcribe_video(self, video_path: str, company_name: str, original_video_url: str = None) -> List[Dict]:
        """Transcribe video using tiny Whisper model with aggressive memory management"""
        self.log_memory("Before transcription")
        
        # Force garbage collection before transcription
        import gc
        gc.collect()
        
        # Memory check before loading model
        current_memory = self.log_memory("Before loading Whisper model")
        if current_memory > self.max_memory_mb * 0.7:  # 70% threshold
            logger.warning(f"⚠️ High memory usage before model load: {current_memory:.1f}MB")
            # Force additional cleanup
            gc.collect()
        
        # Use tiny model for memory efficiency
        try:
            model = whisper.load_model(self.whisper_model)
            self.log_memory("After loading Whisper model")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise
        
        # Memory check before transcription
        current_memory = self.log_memory("Before Whisper transcription")
        if current_memory > self.max_memory_mb * 0.8:  # 80% threshold
            logger.warning(f"⚠️ High memory usage before transcription: {current_memory:.1f}MB")
        
        # Transcribe with memory monitoring
        try:
            result = model.transcribe(video_path, task="translate")
            
            # Force cleanup after transcription
            del model
            gc.collect()
            
            self.log_memory("After transcription")
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            # Cleanup on error
            if 'model' in locals():
                del model
            gc.collect()
            raise
        
        # Generate context with memory monitoring
        try:
            context_input = "\n".join([seg["text"] for seg in result["segments"][:5]])  # Reduced from 10 to 5
            context_prompt = f"Summarize the main topic or context of this Loom transcript:\n\n{context_input}"
            
            # Import OpenAI here to avoid circular imports
            import openai
            context_resp = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You summarize Loom video transcripts."},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=100,  # Reduced from 150 to 100
                temperature=0.5
            )
            context = context_resp.choices[0].message.content.strip()
            logger.info(f"📘 Context: {context}")
        except Exception as e:
            logger.error(f"❌ Context generation failed: {e}")
            context = "Loom video content"
        
        # Format time for chunks
        def format_time(t):
            h, m, s = int(t // 3600), int((t % 3600) // 60), t % 60
            return f"{h:02}:{m:02}:{s:06.3f}".replace('.', ',')
        
        # Create transcript chunks with memory monitoring
        chunks = []
        for i, seg in enumerate(result["segments"]):
            # Memory check every 10 segments
            if i % 10 == 0:
                self.log_memory(f"Processing chunk {i}")
            
            chunk = {
                "source": f"{os.path.basename(video_path)} [{format_time(seg['start'])} - {format_time(seg['end'])}]",
                "original_video_url": original_video_url,
                "text": seg["text"].strip(),
                "context": context,
                "type": "loom_video",
                "start": seg["start"],
                "end": seg["end"]
            }
            chunks.append(chunk)
        
        # Final cleanup
        del result
        gc.collect()
        self.log_memory("After chunk creation")
        
        return chunks
    
    def process_loom_video(self, loom_url: str, company_name: str) -> Dict:
        """Process Loom video: download, transcribe, and cleanup"""
        try:
            # Create temporary file with unique name
            temp_filename = f"loom_video_{os.getpid()}.mp4"
            
            # Download video
            downloaded_file = self.download_loom_video(loom_url, temp_filename)
            
            # Check video file size before processing
            if os.path.exists(downloaded_file):
                file_size_mb = os.path.getsize(downloaded_file) / (1024 * 1024)
                max_size_mb = RENDER_CONFIG.get('MAX_VIDEO_SIZE_MB', 20)
                
                if file_size_mb > max_size_mb:
                    # Clean up large file
                    os.remove(downloaded_file)
                    error_msg = f"Video file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
                    logger.warning(f"⚠️ {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "video_url": loom_url,
                        "company_name": company_name
                    }
                
                logger.info(f"📊 Video file size: {file_size_mb:.1f}MB")
            
            # Transcribe video
            chunks = self.transcribe_video(downloaded_file, company_name, loom_url)
            
            # IMMEDIATELY DELETE THE VIDEO FILE
            if os.path.exists(downloaded_file):
                os.remove(downloaded_file)
                logger.info(f"🗑️ Cleaned up Loom video file: {downloaded_file}")
            
            return {
                "success": True,
                "chunks": chunks,
                "video_url": loom_url,
                "company_name": company_name,
                "processing_method": "loom_ytdlp"
            }
            
        except Exception as e:
            # Cleanup on error
            temp_filename = f"loom_video_{os.getpid()}.mp4"
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.info(f"🗑️ Cleaned up Loom video file on error: {temp_filename}")
            
            logger.error(f"❌ Loom video processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "video_url": loom_url,
                "company_name": company_name
            }

def is_loom_url(url: str) -> bool:
    """Check if URL is a Loom video URL"""
    loom_patterns = [
        r'loom\.com/share/',
        r'loom\.com/embed/',
        r'loom\.com/recordings/',
        r'^[a-zA-Z0-9-]+$'  # Just an ID
    ]
    
    for pattern in loom_patterns:
        if re.search(pattern, url):
            return True
    
    return False 