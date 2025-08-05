#!/usr/bin/env python3
"""
PoToken Video Processor - Enhanced YouTube Download with Bot Detection Bypass
Integrates with Node.js PoToken service for reliable video processing on Render
"""

import os
import tempfile
import logging
import requests
import json
import time
import random
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import yt_dlp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoTokenVideoProcessor:
    def __init__(self, node_backend_url: str = None):
        """
        Initialize PoToken video processor
        
        Args:
            node_backend_url: URL of the Node.js backend with PoToken service
        """
        self.node_backend_url = node_backend_url or os.getenv('NODE_BACKEND_URL', 'http://localhost:5000')
        self.potoken_service_url = f"{self.node_backend_url}/api/potoken"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        domain = urlparse(url).netloc.lower()
        return 'youtube.com' in domain or 'youtu.be' in domain
    
    def check_potoken_service(self) -> bool:
        """Check if PoToken service is available"""
        try:
            response = self.session.get(f"{self.potoken_service_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ PoToken service available: {data}")
                return data.get('nodeAvailable', False)
            else:
                logger.warning(f"⚠️ PoToken service health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ PoToken service not available: {e}")
            return False
    
    def generate_potoken(self, video_url: str) -> Optional[Dict]:
        """Generate PoToken for YouTube video"""
        try:
            logger.info(f"🔐 Generating PoToken for: {video_url}")
            
            response = self.session.post(
                f"{self.potoken_service_url}/generate",
                json={'videoUrl': video_url},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    logger.info("✅ PoToken generated successfully")
                    return data.get('data')
                else:
                    logger.error(f"❌ PoToken generation failed: {data.get('error')}")
                    return None
            else:
                logger.error(f"❌ PoToken service error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ PoToken generation error: {e}")
            return None
    
    def download_with_potoken(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Download video using PoToken"""
        try:
            logger.info(f"📥 Downloading with PoToken: {video_url}")
            
            response = self.session.post(
                f"{self.potoken_service_url}/download",
                json={
                    'videoUrl': video_url,
                    'outputPath': output_path
                },
                timeout=300  # 5 minutes timeout for download
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    logger.info("✅ PoToken download completed successfully")
                    return data.get('data')
                else:
                    logger.error(f"❌ PoToken download failed: {data.get('error')}")
                    return None
            else:
                logger.error(f"❌ PoToken download service error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ PoToken download error: {e}")
            return None
    
    def download_with_ytdlp_fallback(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Fallback to yt-dlp if PoToken fails"""
        try:
            logger.info(f"🔄 Using yt-dlp fallback for: {video_url}")
            
            # Enhanced yt-dlp configuration
            ydl_opts = {
                'format': 'best[height<=720]/best',
                'outtmpl': output_path.replace('.mp4', '.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
                
                # Anti-bot headers
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                
                # Multiple client approach
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web', 'android'],
                        'player_skip': ['webpage'],
                        'player_params': {'hl': 'en', 'gl': 'US'},
                    }
                },
                
                # Rate limiting
                'sleep_interval': random.randint(2, 5),
                'max_sleep_interval': random.randint(5, 10),
                'retries': 5,
                'fragment_retries': 5,
                
                # Additional options
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_color': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                
            # Check for downloaded file
            base_name = output_path.replace('.mp4', '')
            for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv']:
                test_file = base_name + ext
                if os.path.exists(test_file) and os.path.getsize(test_file) > 1000:
                    return {
                        'success': True,
                        'filePath': test_file,
                        'title': info.get('title', 'Unknown Title'),
                        'method': 'yt-dlp-fallback'
                    }
            
            raise Exception("No valid file found after download")
            
        except Exception as e:
            logger.error(f"❌ yt-dlp fallback error: {e}")
            return None
    
    def process_video(self, video_url: str, output_filename: str) -> Optional[Dict]:
        """
        Main video processing method with PoToken integration
        
        Args:
            video_url: URL of the video to download
            output_filename: Output filename for the video
            
        Returns:
            Dict with processing results or None if failed
        """
        try:
            # Validate URL
            if not self.is_youtube_url(video_url):
                logger.warning(f"⚠️ Non-YouTube URL detected: {video_url}")
                return self.download_with_ytdlp_fallback(video_url, output_filename)
            
            # Check PoToken service availability
            if not self.check_potoken_service():
                logger.warning("⚠️ PoToken service not available, using yt-dlp fallback")
                return self.download_with_ytdlp_fallback(video_url, output_filename)
            
            # Try PoToken method first
            logger.info("🚀 Attempting PoToken download...")
            potoken_result = self.download_with_potoken(video_url, output_filename)
            
            if potoken_result and potoken_result.get('success'):
                logger.info("✅ PoToken download successful")
                return {
                    **potoken_result,
                    'method': 'potoken',
                    'bypass_success': True
                }
            
            # Fallback to yt-dlp
            logger.info("🔄 PoToken failed, trying yt-dlp fallback...")
            fallback_result = self.download_with_ytdlp_fallback(video_url, output_filename)
            
            if fallback_result:
                logger.info("✅ yt-dlp fallback successful")
                return {
                    **fallback_result,
                    'method': 'yt-dlp-fallback',
                    'bypass_success': False
                }
            
            logger.error("❌ All download methods failed")
            return None
            
        except Exception as e:
            logger.error(f"❌ Video processing error: {e}")
            return None
    
    def get_video_info(self, video_url: str) -> Optional[Dict]:
        """Get video information using PoToken or fallback"""
        try:
            if not self.is_youtube_url(video_url):
                logger.warning(f"⚠️ Non-YouTube URL: {video_url}")
                return None
            
            # Try PoToken service first
            if self.check_potoken_service():
                response = self.session.get(
                    f"{self.potoken_service_url}/info",
                    params={'videoUrl': video_url},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        logger.info("✅ Video info retrieved via PoToken")
                        return data.get('data')
            
            # Fallback to yt-dlp
            logger.info("🔄 Using yt-dlp for video info...")
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
            return {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'uploader': info.get('uploader'),
                'view_count': info.get('view_count'),
                'method': 'yt-dlp-fallback'
            }
            
        except Exception as e:
            logger.error(f"❌ Video info error: {e}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Test the PoToken processor
    processor = PoTokenVideoProcessor()
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("🧪 Testing PoToken Video Processor...")
    
    # Check service availability
    if processor.check_potoken_service():
        print("✅ PoToken service is available")
        
        # Get video info
        info = processor.get_video_info(test_url)
        if info:
            print(f"📹 Video info: {info}")
        
        # Test download (commented out to avoid actual download)
        # result = processor.process_video(test_url, "/tmp/test_video.mp4")
        # if result:
        #     print(f"✅ Download result: {result}")
        # else:
        #     print("❌ Download failed")
    else:
        print("⚠️ PoToken service not available, using fallback methods") 