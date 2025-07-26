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

# Import yt-dlp for video downloading
try:
    import yt_dlp
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("❌ yt-dlp not installed. Please install with: pip install yt-dlp")
    yt_dlp = None

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
        'LOOM_FALLBACK_TO_API': True,
        'MAX_VIDEO_SIZE_MB': 15,
        'enable_aggressive_cleanup': True,
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
        """Log current memory usage and check limits"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"💾 {stage} - Memory: {memory_mb:.1f}MB")
        
        # Check if we're approaching the limit
        if memory_mb > self.max_memory_mb * 0.8:  # 80% threshold
            logger.warning(f"⚠️ High memory usage at {stage}: {memory_mb:.1f}MB (limit: {self.max_memory_mb}MB)")
            
        # Force garbage collection if memory is high
        if memory_mb > self.max_memory_mb * 0.7:  # 70% threshold
            import gc
            gc.collect()
            logger.info(f"🧹 Forced garbage collection at {stage}")
            
        return memory_mb
    
    def aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup"""
        import gc
        import sys
        
        # Force garbage collection multiple times
        for i in range(3):
            gc.collect()
        
        # Clear any cached objects
        if hasattr(sys, 'exc_clear'):
            sys.exc_clear()
        
        # Clear any module-level caches
        for module in list(sys.modules.keys()):
            if module.startswith('whisper') or module.startswith('yt_dlp'):
                try:
                    del sys.modules[module]
                except:
                    pass
        
        # Force another garbage collection
        gc.collect()
        
        current_memory = self.log_memory("After aggressive cleanup")
        return current_memory
    
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
    
    def validate_video_file_enhanced(self, file_path: str) -> bool:
        """Enhanced video file validation with multiple checks"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ File does not exist: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.warning(f"⚠️ File is empty: {file_path}")
                return False
            
            # Check if file is too small (likely HTML error page)
            if file_size < 1024:  # Less than 1KB
                logger.warning(f"⚠️ File too small ({file_size} bytes), likely HTML error: {file_path}")
                return False
            
            # Read first few bytes to check file signature
            with open(file_path, 'rb') as f:
                header = f.read(12)
            
            # Check for common video file signatures
            video_signatures = [
                b'\x00\x00\x00\x20ftyp',  # MP4
                b'\x00\x00\x00\x18ftyp',  # MP4
                b'\x00\x00\x00\x1cftyp',  # MP4
                b'RIFF',  # AVI
                b'\x1a\x45\xdf\xa3',  # WebM/MKV
            ]
            
            is_video = any(header.startswith(sig) for sig in video_signatures)
            
            # Check for HTML content (error pages)
            html_indicators = [b'<!DOCTYPE', b'<html', b'<HTML', b'<title>', b'<TITLE>']
            is_html = any(header.startswith(indicator) for indicator in html_indicators)
            
            if is_html:
                logger.warning(f"⚠️ File appears to be HTML (error page): {file_path}")
                return False
            
            if is_video:
                logger.info(f"✅ Valid video file signature detected: {file_path}")
                return True
            else:
                logger.warning(f"⚠️ Unknown file format, not a recognized video: {file_path}")
                # Try to read more to see if it's HTML
                with open(file_path, 'rb') as f:
                    content = f.read(1024)
                    if b'<html' in content.lower() or b'<!doctype' in content.lower():
                        logger.warning(f"⚠️ File contains HTML content (error page): {file_path}")
                        return False
                
                # If we can't determine, assume it's valid but log
                logger.info(f"⚠️ File format unclear, assuming valid: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error validating file {file_path}: {e}")
            return False
    
    def _download_progress_hook(self, d):
        """Progress hook for yt-dlp downloads"""
        if d['status'] == 'downloading':
            # Log progress every 10% or when memory usage is high
            if 'total_bytes' in d and d['total_bytes']:
                percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                if percent % 10 < 1 or self.log_memory("Download progress") > self.max_memory_mb * 0.8:
                    logger.info(f"📥 Download progress: {percent:.1f}% ({d['downloaded_bytes']}/{d['total_bytes']} bytes)")
        elif d['status'] == 'finished':
            logger.info(f"✅ Download completed: {d['filename']}")
        elif d['status'] == 'error':
            logger.error(f"❌ Download error: {d.get('error', 'Unknown error')}")
    
    def download_loom_video(self, loom_url: str, output_filename: str) -> str:
        """Download Loom video using yt-dlp with fallback formats"""
        try:
            # Check if yt-dlp is available
            if yt_dlp is None:
                raise Exception("yt-dlp is not installed. Please install with: pip install yt-dlp")
            
            # Get configuration settings
            try:
                from render_deployment_config import get_render_optimized_settings
                config = get_render_optimized_settings()
                ytdlp_format = config.get('ytdlp_format', 'worst[height<=240]')
                ytdlp_max_filesize = config.get('ytdlp_max_filesize', '200M')
            except ImportError:
                ytdlp_format = 'worst[height<=240]'
                ytdlp_max_filesize = '200M'
            
            logger.info(f"🎬 Processing Loom video ID: {self.extract_loom_id(loom_url)}")
            
            # First, try to get available formats
            try:
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Extract video info to see available formats
                    info = ydl.extract_info(loom_url, download=False)
                    formats = info.get('formats', [])
                    
                    logger.info(f"✅ Video info extracted: {info.get('title', 'Unknown')} - Duration: {info.get('duration', 0)}s")
                    logger.info(f"📊 Available formats: {len(formats)} total")
                    
                    # Log available formats for debugging
                    for fmt in formats[:5]:  # Show first 5 formats
                        height = fmt.get('height', 'unknown')
                        filesize = fmt.get('filesize', 'unknown bytes')
                        logger.info(f"   - {height}x{fmt.get('width', 'unknown')}: {filesize}")
                    
                    # Try to find the best available format
                    if formats:
                        # If only one format is available, use it directly
                        if len(formats) == 1:
                            format_id = formats[0].get('format_id', 'http-raw')
                            logger.info(f"🔍 Single format available, using format ID: {format_id}")
                            try:
                                ydl_opts = {
                                    'format': format_id,
                                    'outtmpl': output_filename,
                                    'quiet': True,
                                    'max_filesize': ytdlp_max_filesize,
                                    'progress_hooks': [self._download_progress_hook],
                                    'no_warnings': True,
                                    'ignoreerrors': False,
                                }
                                
                                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                    ydl.download([loom_url])
                                
                                # Validate the downloaded file
                                if os.path.exists(output_filename) and self.validate_video_file_enhanced(output_filename):
                                    file_size = os.path.getsize(output_filename)
                                    logger.info(f"✅ Successfully downloaded using format ID {format_id}: {output_filename} ({file_size} bytes)")
                                    return output_filename
                                else:
                                    logger.warning(f"⚠️ Downloaded file validation failed for format ID {format_id}")
                                    if os.path.exists(output_filename):
                                        os.remove(output_filename)
                                        
                            except Exception as e:
                                logger.warning(f"⚠️ Format ID download failed: {e}")
                                if os.path.exists(output_filename):
                                    os.remove(output_filename)
                                
                                # Fallback: Try direct download from format URL
                                try:
                                    format_url = formats[0].get('url')
                                    if format_url:
                                        logger.info(f"🔍 Trying direct download from format URL")
                                        import requests
                                        
                                        # Download directly using requests
                                        r = requests.get(format_url, stream=True, timeout=30)
                                        r.raise_for_status()
                                        
                                        with open(output_filename, 'wb') as f:
                                            for chunk in r.iter_content(chunk_size=8192):
                                                if chunk:
                                                    f.write(chunk)
                                        
                                        # Validate the downloaded file
                                        if os.path.exists(output_filename) and self.validate_video_file_enhanced(output_filename):
                                            file_size = os.path.getsize(output_filename)
                                            logger.info(f"✅ Successfully downloaded using direct URL: {output_filename} ({file_size} bytes)")
                                            return output_filename
                                        else:
                                            logger.warning(f"⚠️ Direct download validation failed")
                                            if os.path.exists(output_filename):
                                                os.remove(output_filename)
                                                
                                except Exception as direct_error:
                                    logger.warning(f"⚠️ Direct download failed: {direct_error}")
                                    if os.path.exists(output_filename):
                                        os.remove(output_filename)
                        else:
                            # Try different format preferences - start with simple ones
                            format_preferences = [
                                'worst',  # Any worst quality (works with single format)
                                'best',   # Any best quality (works with single format)
                                'worst[height<=720]',
                                'worst[height<=480]',
                                'worst[height<=360]',
                                'worst[height<=240]',
                                'best[height<=720]',
                                'best[height<=480]',
                                'best[height<=360]',
                                'best[height<=240]',
                            ]
                            
                            for format_spec in format_preferences:
                                logger.info(f"🔍 Trying yt-dlp format: {format_spec}")
                                
                                try:
                                    ydl_opts = {
                                        'format': format_spec,
                                        'outtmpl': output_filename,
                                        'quiet': True,
                                        'max_filesize': ytdlp_max_filesize,
                                        'progress_hooks': [self._download_progress_hook],
                                        'no_warnings': True,
                                        'ignoreerrors': False,
                                    }
                                    
                                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                        ydl.download([loom_url])
                                    
                                    # Validate the downloaded file
                                    if os.path.exists(output_filename) and self.validate_video_file_enhanced(output_filename):
                                        file_size = os.path.getsize(output_filename)
                                        logger.info(f"✅ Successfully downloaded using yt-dlp: {output_filename} ({file_size} bytes) with format {format_spec}")
                                        return output_filename
                                    else:
                                        logger.warning(f"⚠️ Downloaded file validation failed for format {format_spec}")
                                        if os.path.exists(output_filename):
                                            os.remove(output_filename)
                                            
                                except Exception as e:
                                    logger.warning(f"⚠️ yt-dlp download failed for format {format_spec}: {e}")
                                    if os.path.exists(output_filename):
                                        os.remove(output_filename)
                                    continue
                    
                    # If no formats worked, try the original configured format
                    logger.info(f"🔍 Trying configured format: {ytdlp_format}")
                    try:
                        ydl_opts = {
                            'format': ytdlp_format,
                            'outtmpl': output_filename,
                            'quiet': True,
                            'max_filesize': ytdlp_max_filesize,
                            'progress_hooks': [self._download_progress_hook],
                            'no_warnings': True,
                            'ignoreerrors': False,
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([loom_url])
                        
                        if os.path.exists(output_filename) and self.validate_video_file_enhanced(output_filename):
                            file_size = os.path.getsize(output_filename)
                            logger.info(f"✅ Successfully downloaded using configured format: {output_filename} ({file_size} bytes)")
                            return output_filename
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Configured format download failed: {e}")
                        if os.path.exists(output_filename):
                            os.remove(output_filename)
                    

            
            except Exception as e:
                logger.warning(f"⚠️ Could not extract video info: {e}")
            
            # If all yt-dlp attempts failed, raise an exception
            raise Exception("All yt-dlp formats failed to download video")
            
        except Exception as e:
            logger.error(f"❌ Loom video download failed: {e}")
            raise
    
    def transcribe_video(self, video_path: str, company_name: str, original_video_url: str = None) -> List[Dict]:
        """Transcribe video using Whisper with audio conversion for memory optimization"""
        try:
            # Memory check before transcription
            self.log_memory("Before transcription")
            
            # Convert video to audio first for memory optimization
            audio_path = self.convert_video_to_audio(video_path)
            
            if not audio_path or not os.path.exists(audio_path):
                raise Exception("Failed to convert video to audio")
            
            logger.info(f"🎵 Converted video to audio: {audio_path}")
            self.log_memory("After audio conversion")
            
            # Load Whisper model with memory optimization
            logger.info(f"🤖 Loading Whisper model: {self.whisper_model}")
            model = whisper.load_model(self.whisper_model)
            self.log_memory("After Whisper model load")
            
            # Transcribe audio (much more memory efficient than video)
            logger.info(f"🎤 Transcribing audio: {audio_path}")
            
            # Add transcription options for faster processing
            transcription_options = {
                'language': 'en',  # Specify language for faster processing
                'task': 'transcribe',
                'fp16': False,  # Disable FP16 to avoid warnings
                'verbose': False,  # Reduce logging
            }
            
            # For longer videos, use chunking to avoid timeouts
            audio_duration = self._get_audio_duration(audio_path)
            logger.info(f"🎤 Audio duration: {audio_duration:.1f} seconds")
            
            if audio_duration > 300:  # If longer than 5 minutes
                logger.info(f"🎤 Long video detected ({audio_duration:.1f}s), using chunked transcription")
                result = self._transcribe_in_chunks(audio_path, model, transcription_options)
            else:
                # Regular transcription with optimized settings
                logger.info(f"🎤 Starting transcription for {audio_duration:.1f}s audio")
                result = model.transcribe(audio_path, **transcription_options)
                logger.info(f"✅ Transcription completed successfully")
                    
            self.log_memory("After transcription")
            
            # Clean up audio file immediately
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"🗑️ Cleaned up audio file: {audio_path}")
            
            # Clean up Whisper model to free memory
            del model
            import gc
            gc.collect()
            self.log_memory("After model cleanup")
            
            # Create context for chunks
            context = f"Video transcription for {company_name}"
            
            def format_time(t):
                """Format time in SRT format"""
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = t % 60
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
            
        except Exception as e:
            logger.error(f"❌ Video transcription failed: {e}")
            # Clean up audio file if it exists
            audio_path = video_path.replace('.mp4', '.wav').replace('.webm', '.wav')
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"🗑️ Cleaned up audio file on error: {audio_path}")
            raise
    
    def convert_video_to_audio(self, video_path: str) -> str:
        """Convert video file to audio using ffmpeg for memory optimization"""
        try:
            # Create audio filename
            audio_path = video_path.replace('.mp4', '.wav').replace('.webm', '.wav')
            if audio_path == video_path:  # If no extension change, add .wav
                audio_path = f"{video_path}.wav"
            
            logger.info(f"🎵 Converting video to audio: {video_path} -> {audio_path}")
            
            # Use ffmpeg to extract audio with optimized settings
            import subprocess
            
            # Get audio settings from configuration
            try:
                from render_deployment_config import get_render_optimized_settings
                config = get_render_optimized_settings()
                sample_rate = config.get('audio_sample_rate', 16000)
                channels = config.get('audio_channels', 1)
                codec = config.get('audio_codec', 'pcm_s16le')
            except ImportError:
                # Fallback settings
                sample_rate = 16000
                channels = 1
                codec = 'pcm_s16le'
            
            # FFmpeg command optimized for memory usage and Whisper compatibility
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', codec,  # Audio codec from config
                '-ar', str(sample_rate),  # Sample rate from config
                '-ac', str(channels),  # Number of channels from config
                '-y',  # Overwrite output file
                audio_path
            ]
            
            logger.info(f"🎵 FFmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"❌ FFmpeg conversion failed: {result.stderr}")
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")
            
            # Verify audio file was created and has content
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                if file_size > 1024:  # At least 1KB
                    logger.info(f"✅ Audio conversion successful: {audio_path} ({file_size} bytes)")
                    logger.info(f"🎵 Audio settings: {sample_rate}Hz, {channels} channel(s), {codec}")
                    return audio_path
                else:
                    logger.error(f"❌ Audio file too small: {file_size} bytes")
                    os.remove(audio_path)
                    raise Exception("Audio file too small - conversion may have failed")
            else:
                raise Exception("Audio file was not created")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ FFmpeg conversion timed out")
            # Clean up partial file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise Exception("Audio conversion timed out")
        except Exception as e:
            logger.error(f"❌ Audio conversion failed: {e}")
            # Clean up partial file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe"""
        try:
            import subprocess
            import json
            
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'json', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data['format']['duration'])
                return duration
            else:
                logger.warning(f"⚠️ Could not get audio duration, assuming 60s")
                return 60.0
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting audio duration: {e}, assuming 60s")
            return 60.0
    
    def _transcribe_in_chunks(self, audio_path: str, model, options: dict) -> dict:
        """Transcribe long audio in chunks to avoid timeouts"""
        try:
            import subprocess
            import tempfile
            import os
            
            # Get total duration
            total_duration = self._get_audio_duration(audio_path)
            chunk_duration = 300  # 5 minutes per chunk
            chunks = []
            
            logger.info(f"🎤 Transcribing {total_duration:.1f}s audio in {chunk_duration}s chunks")
            
            for start_time in range(0, int(total_duration), chunk_duration):
                end_time = min(start_time + chunk_duration, int(total_duration))
                
                # Create temporary chunk file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    chunk_path = temp_file.name
                
                try:
                    # Extract audio chunk using ffmpeg
                    cmd = [
                        'ffmpeg', '-i', audio_path,
                        '-ss', str(start_time),
                        '-t', str(end_time - start_time),
                        '-c', 'copy',
                        '-y', chunk_path
                    ]
                    
                    subprocess.run(cmd, capture_output=True, timeout=30)
                    
                    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                        # Transcribe chunk
                        logger.info(f"🎤 Transcribing chunk {start_time}-{end_time}s")
                        chunk_result = model.transcribe(chunk_path, **options)
                        
                        # Adjust timestamps
                        for segment in chunk_result['segments']:
                            segment['start'] += start_time
                            segment['end'] += start_time
                        
                        chunks.extend(chunk_result['segments'])
                        
                finally:
                    # Clean up chunk file
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            # Combine all chunks
            combined_result = {
                'text': ' '.join([seg['text'] for seg in chunks]),
                'segments': chunks
            }
            
            logger.info(f"✅ Chunked transcription completed: {len(chunks)} segments")
            return combined_result
            
        except Exception as e:
            logger.error(f"❌ Chunked transcription failed: {e}")
            # Fallback to regular transcription
            logger.info("🔄 Falling back to regular transcription")
            return model.transcribe(audio_path, **options)
    
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
                
                # Get configuration settings
                try:
                    from render_deployment_config import get_render_optimized_settings
                    config = get_render_optimized_settings()
                    max_size_mb = config.get('MAX_VIDEO_SIZE_MB', 300)  # Default to 300MB for 2GB plan
                except ImportError:
                    max_size_mb = 300  # Fallback for 2GB RAM plan
                
                if file_size_mb > max_size_mb:
                    # Clean up large file
                    os.remove(downloaded_file)
                    error_msg = f"Video file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB). With 2GB RAM, you can process videos up to {max_size_mb}MB. Consider using a lower quality video or contact support for larger videos."
                    logger.warning(f"⚠️ {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "video_url": loom_url,
                        "company_name": company_name
                    }
                
                logger.info(f"📊 Video file size: {file_size_mb:.1f}MB (within {max_size_mb}MB limit for 2GB RAM)")
            
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

 