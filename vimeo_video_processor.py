#!/usr/bin/env python3
"""
Vimeo Video Processor - Process Vimeo videos similar to Loom videos
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
        'VIMEO_FALLBACK_TO_API': True,
        'MAX_VIDEO_SIZE_MB': 15,
        'enable_aggressive_cleanup': True,
    }

logger = logging.getLogger(__name__)

class VimeoVideoProcessor:
    def __init__(self, max_memory_mb: int = None):
        """
        Initialize Vimeo video processor
        
        Args:
            max_memory_mb: Maximum memory usage in MB (uses Render config if None)
        """
        self.max_memory_mb = max_memory_mb or RENDER_CONFIG['max_memory_mb']
        self.vimeo_api_key = os.getenv('VIMEO_API_KEY')
        self.vimeo_api_base = "https://api.vimeo.com"
        
        # Use Render-optimized settings
        self.ytdlp_format = RENDER_CONFIG['ytdlp_format']
        self.ytdlp_max_filesize = RENDER_CONFIG['ytdlp_max_filesize']
        self.whisper_model = RENDER_CONFIG['whisper_model']
        self.prefer_ytdlp = RENDER_CONFIG['prefer_ytdlp']
        
        logger.info(f"🎬 VimeoVideoProcessor initialized - Memory: {self.max_memory_mb}MB, Format: {self.ytdlp_format}")
    
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
    
    def extract_vimeo_id(self, vimeo_url: str) -> Optional[str]:
        """Extract Vimeo video ID from various URL formats"""
        # Handle different Vimeo URL formats
        patterns = [
            r'vimeo\.com/(\d+)',  # vimeo.com/VIDEO_ID
            r'vimeo\.com/channels/[^/]+/(\d+)',  # vimeo.com/channels/CHANNEL/VIDEO_ID
            r'vimeo\.com/groups/[^/]+/videos/(\d+)',  # vimeo.com/groups/GROUP/videos/VIDEO_ID
            r'vimeo\.com/album/[^/]+/video/(\d+)',  # vimeo.com/album/ALBUM/video/VIDEO_ID
            r'player\.vimeo\.com/video/(\d+)',  # player.vimeo.com/video/VIDEO_ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, vimeo_url)
            if match:
                return match.group(1)
        
        # If it's already just an ID
        if re.match(r'^\d+$', vimeo_url):
            return vimeo_url
            
        return None
    
    def get_vimeo_video_info(self, vimeo_id: str) -> Dict:
        """Get video information from Vimeo API"""
        if not self.vimeo_api_key:
            logger.warning("⚠️ VIMEO_API_KEY not found - using yt-dlp fallback method")
            return {"id": vimeo_id, "title": f"Vimeo Video {vimeo_id}", "fallback": True}
        
        headers = {
            'Authorization': f'Bearer {self.vimeo_api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            logger.info(f"🔍 Fetching Vimeo video info for ID: {vimeo_id}")
            response = requests.get(f"{self.vimeo_api_base}/videos/{vimeo_id}", headers=headers, timeout=30)
            
            if response.status_code == 200:
                video_info = response.json()
                logger.info(f"📹 Vimeo video info: {video_info.get('name', 'Untitled')}")
                return video_info
            elif response.status_code == 401:
                logger.error(f"❌ Vimeo API authentication failed")
                return {"id": vimeo_id, "title": f"Vimeo Video {vimeo_id}", "fallback": True}
            else:
                logger.warning(f"⚠️ Vimeo API returned status {response.status_code}")
                return {"id": vimeo_id, "title": f"Vimeo Video {vimeo_id}", "fallback": True}
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Failed to access Vimeo API: {e}")
            return {"id": vimeo_id, "title": f"Vimeo Video {vimeo_id}", "fallback": True}
    
    def check_vimeo_video_accessibility(self, vimeo_url: str) -> bool:
        """Check if Vimeo video is accessible by trying to fetch the page"""
        try:
            logger.info(f"🔍 Checking Vimeo video accessibility: {vimeo_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(vimeo_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                content = response.text.lower()
                
                # Check if the page contains video-related content
                if 'vimeo' in content and ('video' in content or 'player' in content):
                    logger.info("✅ Vimeo video page is accessible")
                    return True
                else:
                    logger.warning("⚠️ Vimeo page accessible but doesn't appear to be a video")
                    return False
            elif response.status_code == 404:
                logger.error("❌ Vimeo video not found (404)")
                return False
            elif response.status_code == 403:
                logger.error("❌ Vimeo video access forbidden (403) - may be private")
                return False
            else:
                logger.warning(f"⚠️ Vimeo page returned status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to check Vimeo video accessibility: {e}")
            return False
    
    def validate_video_file(self, file_path: str) -> bool:
        """Validate that the downloaded file is actually a video"""
        try:
            if not os.path.exists(file_path):
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # Less than 1KB is probably not a video
                logger.warning(f"⚠️ File too small to be a video: {file_size} bytes")
                return False
            
            # Check file extension
            if not file_path.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv')):
                logger.warning(f"⚠️ File doesn't have video extension: {file_path}")
                return False
            
            # Try to read first few bytes to check for video file signatures
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
            # Check for common video file signatures
            video_signatures = [
                b'\x00\x00\x00\x20ftyp',  # MP4
                b'\x00\x00\x00\x18ftyp',  # MP4
                b'\x00\x00\x00\x1Cftyp',  # MP4
                b'\x1A\x45\xDF\xA3',      # WebM/MKV
                b'\x52\x49\x46\x46',      # AVI
                b'\x6D\x6F\x6F\x76',      # MOV
            ]
            
            for signature in video_signatures:
                if header.startswith(signature):
                    logger.info(f"✅ Valid video file signature detected: {file_path}")
                    return True
            
            # If no signature matches, it might still be a valid video
            # Let's check if the file contains video-related strings
            try:
                with open(file_path, 'rb') as f:
                    content = f.read(1024)  # Read first 1KB
                    if b'video' in content.lower() or b'mp4' in content.lower():
                        logger.info(f"✅ Video-related content detected: {file_path}")
                        return True
            except:
                pass
            
            logger.warning(f"⚠️ File doesn't appear to be a valid video: {file_path}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error validating video file {file_path}: {e}")
            return False
    
    def validate_video_file_enhanced(self, file_path: str) -> bool:
        """Enhanced video file validation"""
        try:
            if not self.validate_video_file(file_path):
                return False
            
            # Additional checks for HTML error pages
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1024)
                    if '<html' in content.lower() or '<!doctype' in content.lower():
                        logger.warning(f"⚠️ File appears to be HTML (error page): {file_path}")
                        return False
            except:
                # If we can't read as text, it's probably a binary video file
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced validation failed for {file_path}: {e}")
            return False
    
    def _download_progress_hook(self, d):
        """Progress hook for yt-dlp downloads"""
        if d['status'] == 'downloading':
            try:
                downloaded = d.get('downloaded_bytes', 0)
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                if total > 0:
                    percentage = (downloaded / total) * 100
                    logger.info(f"📥 Download progress: {percentage:.1f}% ({downloaded}/{total} bytes)")
            except:
                pass
        elif d['status'] == 'finished':
            logger.info(f"✅ Download completed: {d.get('filename', 'unknown')}")
    
    def download_vimeo_video(self, vimeo_url: str, output_filename: str) -> str:
        """Download Vimeo video using multiple methods including direct extraction"""
        try:
            self.log_memory("Before Vimeo download")
            
            logger.info(f"📥 Downloading Vimeo video: {vimeo_url}")
            logger.info(f"📁 Output file: {output_filename}")
            
            # Add a small delay to avoid rate limiting
            import time
            time.sleep(1)
            logger.info("⏳ Added delay to avoid rate limiting")
            
            # Method 1: Try direct extraction from Vimeo page (bypasses OAuth issues)
            try:
                logger.info("🔄 Attempting Method 1: Direct Vimeo page extraction")
                direct_url = self._extract_vimeo_direct_url(vimeo_url)
                if direct_url:
                    logger.info(f"🔗 Extracted direct URL: {direct_url}")
                    self._download_direct_url(direct_url, output_filename)
                    
                    if self.validate_video_file_enhanced(output_filename):
                        self.log_memory("After Vimeo download")
                        logger.info(f"✅ Vimeo video downloaded successfully (Method 1): {output_filename}")
                        return output_filename
                    else:
                        logger.warning("⚠️ Method 1 failed validation, trying Method 2")
                        if os.path.exists(output_filename):
                            os.remove(output_filename)
                else:
                    logger.warning("⚠️ Could not extract direct URL, trying Method 2")
                        
            except Exception as e:
                logger.warning(f"⚠️ Method 1 failed: {e}")
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            
            # Method 2: Try with yt-dlp but with different user agent and no OAuth
            try:
                ydl_opts = {
                    'format': 'best[height<=720]/best[height<=480]/worst',
                    'outtmpl': output_filename,
                    'quiet': False,
                    'max_filesize': self.ytdlp_max_filesize,
                    'progress_hooks': [self._download_progress_hook],
                    'no_check_certificate': True,
                    'extractor_retries': 1,  # Reduced retries to avoid rate limiting
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    },
                    'cookiesfrombrowser': None,
                    'cookiefile': None,
                    'sleep_interval': 2,  # Add delay between requests
                    'max_sleep_interval': 5,  # Maximum sleep interval
                    'extract_flat': False,
                    'no_playlist': True,
                }
                
                logger.info("🔄 Attempting Method 2: yt-dlp with custom headers")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([vimeo_url])
                
                # Validate the downloaded file
                if self.validate_video_file_enhanced(output_filename):
                    self.log_memory("After Vimeo download")
                    logger.info(f"✅ Vimeo video downloaded successfully (Method 2): {output_filename}")
                    return output_filename
                else:
                    logger.warning("⚠️ Method 2 failed validation, trying Method 3")
                    if os.path.exists(output_filename):
                        os.remove(output_filename)
                        
            except Exception as e:
                logger.warning(f"⚠️ Method 2 failed: {e}")
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            
            # Method 3: Try with Vimeo API if available
            if self.vimeo_api_key:
                try:
                    logger.info("🔄 Attempting Method 3: Vimeo API download")
                    api_url = self._get_vimeo_api_download_url(vimeo_url)
                    if api_url:
                        self._download_direct_url(api_url, output_filename)
                        
                        if self.validate_video_file_enhanced(output_filename):
                            self.log_memory("After Vimeo download")
                            logger.info(f"✅ Vimeo video downloaded successfully (Method 3 - API): {output_filename}")
                            return output_filename
                        else:
                            logger.warning("⚠️ Method 3 failed validation, trying Method 4")
                            if os.path.exists(output_filename):
                                os.remove(output_filename)
                    else:
                        logger.warning("⚠️ Could not get API download URL, trying Method 4")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Method 3 (API) failed: {e}")
                    if os.path.exists(output_filename):
                        os.remove(output_filename)
            
            # Method 4: Try with very basic yt-dlp options
            try:
                ydl_opts = {
                    'format': 'worst[ext=mp4]/worst',
                    'outtmpl': output_filename,
                    'quiet': False,
                    'max_filesize': '50M',
                    'progress_hooks': [self._download_progress_hook],
                    'no_check_certificate': True,
                    'extractor_retries': 1,
                    'ignoreerrors': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'no_playlist': True,
                }
                
                logger.info("🔄 Attempting Method 4: Basic yt-dlp fallback")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([vimeo_url])
                
                # Validate the downloaded file
                if self.validate_video_file_enhanced(output_filename):
                    self.log_memory("After Vimeo download")
                    logger.info(f"✅ Vimeo video downloaded successfully (Method 4): {output_filename}")
                    return output_filename
                else:
                    logger.warning("⚠️ Method 4 failed validation, trying Method 5")
                    if os.path.exists(output_filename):
                        os.remove(output_filename)
                        
            except Exception as e:
                logger.warning(f"⚠️ Method 4 failed: {e}")
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            
            # Method 5: Try with Vimeo embed URL format
            try:
                vimeo_id = self.extract_vimeo_id(vimeo_url)
                if vimeo_id:
                    embed_url = f"https://player.vimeo.com/video/{vimeo_id}"
                    logger.info(f"🔄 Attempting Method 5: Vimeo embed URL: {embed_url}")
                    
                    ydl_opts = {
                        'format': 'worst[ext=mp4]/worst',
                        'outtmpl': output_filename,
                        'quiet': False,
                        'max_filesize': '50M',
                        'progress_hooks': [self._download_progress_hook],
                        'no_check_certificate': True,
                        'extractor_retries': 1,
                        'ignoreerrors': True,
                        'no_warnings': True,
                        'extract_flat': False,
                        'no_playlist': True,
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([embed_url])
                    
                    # Validate the downloaded file
                    if self.validate_video_file_enhanced(output_filename):
                        self.log_memory("After Vimeo download")
                        logger.info(f"✅ Vimeo video downloaded successfully (Method 5 - embed): {output_filename}")
                        return output_filename
                    else:
                        logger.error("❌ Method 5 failed validation")
                        if os.path.exists(output_filename):
                            os.remove(output_filename)
                else:
                    logger.warning("⚠️ Could not extract Vimeo ID for embed URL method")
                        
            except Exception as e:
                logger.warning(f"⚠️ Method 5 failed: {e}")
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            
            # All methods failed
            logger.error("❌ All download methods failed")
            if os.path.exists(output_filename):
                os.remove(output_filename)
            
            # Try to get more specific error information from the API
            specific_error = self._get_vimeo_error_details(vimeo_url)
            
            # Provide a clear error message for restricted videos
            if specific_error:
                error_msg = specific_error
            else:
                error_msg = "This Vimeo video has download restrictions. The video may be private, password-protected, or have special access requirements. Please try a different Vimeo video or ensure the video is publicly accessible."
            
            raise Exception(error_msg)
            
        except Exception as e:
            # Cleanup failed download
            if os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                    logger.info(f"🗑️ Cleaned up failed download: {output_filename}")
                except:
                    pass
            
            # Re-raise the exception to ensure the calling code knows the download failed
            raise
    
    def _extract_vimeo_direct_url(self, vimeo_url: str) -> Optional[str]:
        """Extract direct video URL from Vimeo page HTML"""
        try:
            # Extract video ID
            vimeo_id = self.extract_vimeo_id(vimeo_url)
            if not vimeo_id:
                return None
            
            # Fetch the Vimeo page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(vimeo_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            html_content = response.text
            
            # Look for video configuration in the HTML
            # Vimeo often embeds video config in a script tag
            import re
            
            # Pattern 1: Look for window.vimeo.clip_page_config
            pattern1 = r'window\.vimeo\.clip_page_config\s*=\s*({.*?});'
            match1 = re.search(pattern1, html_content, re.DOTALL)
            
            if match1:
                try:
                    config_data = json.loads(match1.group(1))
                    if 'player' in config_data and 'config_url' in config_data['player']:
                        config_url = config_data['player']['config_url']
                        logger.info(f"🔍 Found config URL: {config_url}")
                        
                        # Fetch the player config
                        config_response = requests.get(config_url, headers=headers, timeout=30)
                        config_response.raise_for_status()
                        config_json = config_response.json()
                        
                        # Extract video URLs from config
                        if 'request' in config_json and 'files' in config_json['request']:
                            files = config_json['request']['files']
                            
                            # Look for progressive download URLs
                            if 'progressive' in files:
                                progressive_files = files['progressive']
                                if progressive_files:
                                    # Get the best quality available
                                    best_quality = max(progressive_files, key=lambda x: x.get('width', 0))
                                    video_url = best_quality.get('url')
                                    if video_url:
                                        logger.info(f"✅ Extracted direct video URL: {video_url}")
                                        return video_url
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse config data: {e}")
            
            # Pattern 2: Look for direct video URLs in the HTML
            pattern2 = r'"url":"(https://[^"]*\.mp4[^"]*)"'
            matches2 = re.findall(pattern2, html_content)
            
            if matches2:
                # Get the first valid URL
                for url in matches2:
                    if 'vimeo' in url and '.mp4' in url:
                        logger.info(f"✅ Found direct MP4 URL: {url}")
                        return url
            
            # Pattern 3: Look for video sources
            pattern3 = r'<source[^>]*src="([^"]*)"[^>]*>'
            matches3 = re.findall(pattern3, html_content)
            
            if matches3:
                for url in matches3:
                    if 'vimeo' in url and '.mp4' in url:
                        logger.info(f"✅ Found source MP4 URL: {url}")
                        return url
            
            logger.warning("⚠️ Could not extract direct video URL from Vimeo page")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting direct URL: {e}")
            return None
    
    def _download_direct_url(self, direct_url: str, output_filename: str):
        """Download video from direct URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            logger.info(f"📥 Downloading from direct URL: {direct_url}")
            
            response = requests.get(direct_url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(output_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress every 1MB
                        if downloaded_size % (1024 * 1024) == 0:
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                logger.info(f"📊 Download progress: {progress:.1f}% ({downloaded_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
                            else:
                                logger.info(f"📊 Downloaded: {downloaded_size / (1024*1024):.1f}MB")
            
            logger.info(f"✅ Direct download completed: {output_filename}")
            
        except Exception as e:
            logger.error(f"❌ Direct download failed: {e}")
            raise
    
    def _get_vimeo_api_download_url(self, vimeo_url: str) -> Optional[str]:
        """Get download URL using Vimeo API"""
        try:
            if not self.vimeo_api_key:
                logger.warning("⚠️ VIMEO_API_KEY not found - skipping API method")
                return None
            
            vimeo_id = self.extract_vimeo_id(vimeo_url)
            if not vimeo_id:
                logger.warning("⚠️ Could not extract Vimeo ID from URL")
                return None
            
            # Use Vimeo API to get video info
            api_url = f"{self.vimeo_api_base}/videos/{vimeo_id}"
            headers = {
                'Authorization': f'Bearer {self.vimeo_api_key}',
                'Content-Type': 'application/json'
            }
            
            logger.info(f"🔍 Fetching video info from Vimeo API: {api_url}")
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            video_data = response.json()
            
            # Debug: Log the response structure
            logger.info(f"📊 Vimeo API response keys: {list(video_data.keys())}")
            
            # Method 1: Look for download links in the main response
            if 'download' in video_data and video_data['download']:
                download_links = video_data['download']
                logger.info(f"📥 Found {len(download_links)} download links")
                
                # Get the best quality available
                best_quality = None
                for link in download_links:
                    logger.info(f"🔗 Download link: {link.get('quality', 'unknown')} - {link.get('type', 'unknown')}")
                    if link.get('quality') == 'hd' and link.get('type') == 'source':
                        best_quality = link
                        break
                
                if not best_quality:
                    # Fall back to any available download link
                    for link in download_links:
                        if link.get('type') == 'source':
                            best_quality = link
                            break
                
                if best_quality and 'link' in best_quality:
                    logger.info(f"✅ Found API download URL: {best_quality['link']}")
                    return best_quality['link']
            
            # Method 2: Look for progressive download links
            if 'files' in video_data:
                files = video_data['files']
                logger.info(f"📁 Found files section with keys: {list(files.keys()) if isinstance(files, dict) else 'not a dict'}")
                
                # Look for progressive files
                if 'progressive' in files and files['progressive']:
                    progressive_files = files['progressive']
                    logger.info(f"📥 Found {len(progressive_files)} progressive files")
                    
                    # Get the best quality available
                    best_quality = None
                    for file_info in progressive_files:
                        logger.info(f"🔗 Progressive file: {file_info.get('width', 'unknown')}x{file_info.get('height', 'unknown')} - {file_info.get('quality', 'unknown')}")
                        if file_info.get('quality') == 'hd':
                            best_quality = file_info
                            break
                    
                    if not best_quality and progressive_files:
                        # Get the highest quality available
                        best_quality = max(progressive_files, key=lambda x: x.get('width', 0))
                    
                    if best_quality and 'link' in best_quality:
                        logger.info(f"✅ Found progressive download URL: {best_quality['link']}")
                        return best_quality['link']
            
            # Method 3: Look for direct video files
            if 'files' in video_data:
                files = video_data['files']
                if 'source' in files and files['source']:
                    source_file = files['source']
                    if 'link' in source_file:
                        logger.info(f"✅ Found source file download URL: {source_file['link']}")
                        return source_file['link']
            
            logger.warning("⚠️ No download links found in Vimeo API response")
            logger.info(f"🔍 Full API response structure: {list(video_data.keys())}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting Vimeo API download URL: {e}")
            return None
    
    def _get_vimeo_error_details(self, vimeo_url: str) -> Optional[str]:
        """Get specific error details from Vimeo API"""
        try:
            if not self.vimeo_api_key:
                return None
            
            vimeo_id = self.extract_vimeo_id(vimeo_url)
            if not vimeo_id:
                return None
            
            # Use Vimeo API to get video info
            api_url = f"{self.vimeo_api_base}/videos/{vimeo_id}"
            headers = {
                'Authorization': f'Bearer {self.vimeo_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(api_url, headers=headers, timeout=30)
            
            if response.status_code == 404:
                return "Vimeo video not found. Please check if the URL is correct and the video is publicly accessible."
            elif response.status_code == 403:
                return "This Vimeo video is private or requires special access. Please ensure the video is public or you have permission to access it."
            elif response.status_code == 401:
                return "Authentication failed. Please check your Vimeo API key and ensure it has the correct permissions."
            
            if response.status_code == 200:
                video_data = response.json()
                
                # Check if video has download restrictions
                if 'download' in video_data:
                    download_info = video_data['download']
                    if not download_info:
                        return "This Vimeo video does not allow downloads. The video owner has disabled download access."
                
                # Check privacy settings
                if 'privacy' in video_data:
                    privacy = video_data['privacy']
                    if privacy.get('view') == 'nobody':
                        return "This Vimeo video is private and not accessible to the public."
                    elif privacy.get('view') == 'password':
                        return "This Vimeo video is password-protected and requires a password to access."
                    elif privacy.get('view') == 'contacts':
                        return "This Vimeo video is only accessible to the owner's contacts."
                
                # Check if video is embeddable
                if 'embed' in video_data:
                    embed_info = video_data['embed']
                    if not embed_info.get('playbar', True):
                        return "This Vimeo video has restricted embedding and may not be accessible for processing."
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Could not get detailed error info: {e}")
            return None
    
    def transcribe_video(self, video_path: str, company_name: str, original_video_url: str = None) -> List[Dict]:
        """Transcribe video using Whisper and return chunks"""
        try:
            self.log_memory("Before transcription")
            
            # Convert video to audio for transcription
            audio_path = self.convert_video_to_audio(video_path)
            
            # Load Whisper model
            logger.info(f"🎤 Loading Whisper model: {self.whisper_model}")
            model = whisper.load_model(self.whisper_model)
            
            # Transcribe audio
            logger.info(f"🎤 Transcribing Vimeo video: {video_path}")
            result = model.transcribe(audio_path, task="translate", verbose=True)
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"🗑️ Cleaned up audio file: {audio_path}")
            
            # Process segments into chunks
            chunks = []
            for i, segment in enumerate(result["segments"]):
                chunk = {
                    "id": f"vimeo_{company_name}_{i}",
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                    "video_url": original_video_url or video_path,
                    "company_name": company_name,
                    "source": "vimeo",
                    "chunk_index": i
                }
                chunks.append(chunk)
            
            self.log_memory("After transcription")
            
            logger.info(f"✅ Vimeo video transcription completed: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Vimeo video transcription failed: {e}")
            raise
    
    def convert_video_to_audio(self, video_path: str) -> str:
        """Convert video to audio for transcription"""
        try:
            import subprocess
            
            # Create temporary audio file
            audio_path = video_path.replace('.mp4', '.wav').replace('.webm', '.wav').replace('.avi', '.wav')
            if audio_path == video_path:
                audio_path = video_path + '.wav'
            
            logger.info(f"🎵 Converting video to audio: {video_path} -> {audio_path}")
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"❌ FFmpeg conversion failed: {result.stderr}")
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")
            
            # Validate audio file
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise Exception("Audio conversion failed - output file is empty or missing")
            
            logger.info(f"✅ Audio conversion completed: {audio_path}")
            return audio_path
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Audio conversion timed out")
            raise Exception("Audio conversion timed out")
        except Exception as e:
            logger.error(f"❌ Audio conversion failed: {e}")
            raise
    
    def process_vimeo_video(self, vimeo_url: str, company_name: str) -> Dict:
        """Process Vimeo video: download, transcribe, and cleanup"""
        try:
            # Check if video is accessible first
            if not self.check_vimeo_video_accessibility(vimeo_url):
                return {
                    "success": False,
                    "error": "Vimeo video is not accessible. Please ensure the video is public and the URL is correct.",
                    "video_url": vimeo_url,
                    "company_name": company_name
                }
            
            # Create temporary file with unique name
            temp_filename = f"vimeo_video_{os.getpid()}.mp4"
            
            # Download video
            try:
                downloaded_file = self.download_vimeo_video(vimeo_url, temp_filename)
            except Exception as download_error:
                # Don't log here - let the main exception handler deal with it
                return {
                    "success": False,
                    "error": str(download_error),
                    "video_url": vimeo_url,
                    "company_name": company_name
                }
            
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
                        "video_url": vimeo_url,
                        "company_name": company_name
                    }
                
                logger.info(f"📊 Video file size: {file_size_mb:.1f}MB (within {max_size_mb}MB limit for 2GB RAM)")
            
            # Transcribe video
            chunks = self.transcribe_video(downloaded_file, company_name, vimeo_url)
            
            # IMMEDIATELY DELETE THE VIDEO FILE
            if os.path.exists(downloaded_file):
                os.remove(downloaded_file)
                logger.info(f"🗑️ Cleaned up Vimeo video file: {downloaded_file}")
            
            return {
                "success": True,
                "chunks": chunks,
                "video_url": vimeo_url,
                "company_name": company_name,
                "processing_method": "vimeo_ytdlp"
            }
            
        except Exception as e:
            # Cleanup on error
            temp_filename = f"vimeo_video_{os.getpid()}.mp4"
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.info(f"🗑️ Cleaned up Vimeo video file on error: {temp_filename}")
            
            # Provide more helpful error messages for common Vimeo issues
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg:
                user_friendly_error = "Vimeo is temporarily rate limiting requests. Please wait a few minutes and try again, or try a different Vimeo video."
            elif "400" in error_msg or "Bad Request" in error_msg:
                user_friendly_error = "This Vimeo video has special restrictions that prevent downloading. Please try a different Vimeo video or contact the video owner to make it publicly downloadable."
            elif "OAuth token" in error_msg or "authentication" in error_msg.lower():
                user_friendly_error = "This Vimeo video requires authentication or is private. Please ensure the video is public and accessible, or try a different Vimeo video."
            elif "not found" in error_msg.lower() or "404" in error_msg:
                user_friendly_error = "Vimeo video not found. Please check if the URL is correct and the video is publicly accessible."
            elif "download" in error_msg.lower() and "failed" in error_msg.lower():
                user_friendly_error = "Unable to download this Vimeo video. The video may be private, restricted, or require authentication."
            elif "All Vimeo download methods failed" in error_msg:
                user_friendly_error = "Unable to download this Vimeo video. This video appears to have special restrictions. Please try a different Vimeo video or ensure the video is publicly downloadable."
            else:
                user_friendly_error = f"Vimeo video processing failed: {str(e)}. Please check if the video URL is accessible and try again."
            
            return {
                "success": False,
                "error": user_friendly_error,
                "video_url": vimeo_url,
                "company_name": company_name
            }

def is_vimeo_url(url: str) -> bool:
    """Check if URL is a Vimeo video URL"""
    vimeo_patterns = [
        r'vimeo\.com/\d+',
        r'vimeo\.com/channels/[^/]+/\d+',
        r'vimeo\.com/groups/[^/]+/videos/\d+',
        r'vimeo\.com/album/[^/]+/video/\d+',
        r'player\.vimeo\.com/video/\d+',
        r'^\d+$'  # Just an ID
    ]
    
    for pattern in vimeo_patterns:
        if re.search(pattern, url):
            return True
    
    return False 