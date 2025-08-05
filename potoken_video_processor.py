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
        
        # Debug logging to see what URL is being used
        logger.info(f"🔗 PoToken service URL: {self.potoken_service_url}")
        logger.info(f"🔗 Node backend URL: {self.node_backend_url}")
        logger.info(f"🔗 Environment NODE_BACKEND_URL: {os.getenv('NODE_BACKEND_URL', 'NOT_SET')}")
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
    

    
    def process_video(self, video_url: str, output_filename: str) -> Optional[Dict]:
        """
        Main video processing method with PoToken integration ONLY
        
        Args:
            video_url: URL of the video to download
            output_filename: Output filename for the video
            
        Returns:
            Dict with processing results or None if failed
        """
        try:
            # Validate URL
            if not self.is_youtube_url(video_url):
                logger.error(f"❌ Non-YouTube URL detected: {video_url}")
                raise Exception("PoToken processor only supports YouTube videos")
            
            # Check PoToken service availability
            if not self.check_potoken_service():
                logger.error("❌ PoToken service not available")
                raise Exception("PoToken service is not available. Please ensure Node.js backend is deployed and running.")
            
            # Try PoToken method ONLY
            logger.info("🚀 Attempting PoToken download...")
            potoken_result = self.download_with_potoken(video_url, output_filename)
            
            if potoken_result and potoken_result.get('success'):
                logger.info("✅ PoToken download successful")
                return {
                    **potoken_result,
                    'method': 'potoken',
                    'bypass_success': True
                }
            
            # No fallback - PoToken failed
            logger.error("❌ PoToken download failed")
            raise Exception("PoToken download failed. No fallback available.")
            
        except Exception as e:
            logger.error(f"❌ Video processing error: {e}")
            raise e
    
    def get_video_info(self, video_url: str) -> Optional[Dict]:
        """Get video information using PoToken ONLY"""
        try:
            if not self.is_youtube_url(video_url):
                logger.error(f"❌ Non-YouTube URL: {video_url}")
                raise Exception("PoToken processor only supports YouTube videos")
            
            # Check PoToken service availability
            if not self.check_potoken_service():
                logger.error("❌ PoToken service not available")
                raise Exception("PoToken service is not available. Please ensure Node.js backend is deployed and running.")
            
            # Try PoToken service only
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
            
            # PoToken failed
            logger.error("❌ PoToken video info failed")
            raise Exception("PoToken video info failed. No fallback available.")
            
        except Exception as e:
            logger.error(f"❌ Video info error: {e}")
            raise e

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