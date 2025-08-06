#!/usr/bin/env python3
"""
Simple Video Processor - YouTube Download with Advanced yt-dlp Configurations
Bypasses bot detection using multiple yt-dlp strategies without browser dependencies
"""

import os
import tempfile
import logging
import time
import random
import json
import subprocess
from typing import Dict, Optional, List
from urllib.parse import urlparse
import yt_dlp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVideoProcessor:
    def __init__(self):
        """
        Initialize Simple video processor with advanced yt-dlp configurations
        """
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
        ]
        
        logger.info("🔧 Initializing Simple Video Processor...")
        
    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        domain = urlparse(url).netloc.lower()
        return 'youtube.com' in domain or 'youtu.be' in domain
    
    def add_random_delay(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """Add random delay to avoid detection"""
        delay = random.uniform(min_delay, max_delay)
        logger.info(f"⏱️ Adding random delay: {delay:.2f}s")
        time.sleep(delay)
    
    def rotate_user_agent(self) -> str:
        """Get a random user agent"""
        return random.choice(self.user_agents)
    
    def get_advanced_headers(self) -> Dict[str, str]:
        """Get advanced headers to mimic real browser"""
        return {
            'User-Agent': self.rotate_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"'
        }
    
    def download_with_advanced_config(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Download using advanced yt-dlp configuration"""
        try:
            logger.info("📥 Using advanced yt-dlp configuration...")
            
            # Advanced configuration with multiple strategies
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best[ext=mp4]/best[height<=720]/best',
                'quiet': True,
                'no_warnings': True,
                'http_headers': self.get_advanced_headers(),
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'live'],
                        'player_client': ['android', 'web'],
                        'player_skip': ['webpage', 'configs'],
                        'player_params': {
                            'hl': 'en',
                            'gl': 'US'
                        }
                    }
                },
                'cookiesfrombrowser': None,  # Disable cookie extraction
                'extract_flat': False,
                'ignoreerrors': False,
                'no_check_certificate': True,
                'prefer_insecure': True,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'geo_bypass_ip_block': '1.0.0.1',
                'socket_timeout': 30,
                'retries': 3
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Advanced download completed: {output_path} ({file_size} bytes)")
                return {
                    'success': True,
                    'filePath': output_path,
                    'method': 'advanced-ytdlp',
                    'fileSize': file_size
                }
            else:
                raise Exception("Downloaded file is empty or missing")
                
        except Exception as e:
            logger.error(f"❌ Advanced download failed: {e}")
            return None
    
    def download_with_mobile_config(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Download using mobile client configuration"""
        try:
            logger.info("📱 Using mobile client configuration...")
            
            mobile_headers = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'http_headers': mobile_headers,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android'],
                        'player_skip': ['webpage', 'configs'],
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Mobile download completed: {output_path} ({file_size} bytes)")
                return {
                    'success': True,
                    'filePath': output_path,
                    'method': 'mobile-ytdlp',
                    'fileSize': file_size
                }
            else:
                raise Exception("Downloaded file is empty or missing")
                
        except Exception as e:
            logger.error(f"❌ Mobile download failed: {e}")
            return None
    
    def download_with_minimal_config(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Download using minimal configuration"""
        try:
            logger.info("🔧 Using minimal configuration...")
            
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best',
                'quiet': True,
                'no_warnings': True,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web'],
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Minimal download completed: {output_path} ({file_size} bytes)")
                return {
                    'success': True,
                    'filePath': output_path,
                    'method': 'minimal-ytdlp',
                    'fileSize': file_size
                }
            else:
                raise Exception("Downloaded file is empty or missing")
                
        except Exception as e:
            logger.error(f"❌ Minimal download failed: {e}")
            return None
    
    def download_with_legacy_config(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Download using legacy configuration for older videos"""
        try:
            logger.info("📺 Using legacy configuration...")
            
            legacy_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'http_headers': legacy_headers,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web'],
                        'player_skip': ['configs'],
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Legacy download completed: {output_path} ({file_size} bytes)")
                return {
                    'success': True,
                    'filePath': output_path,
                    'method': 'legacy-ytdlp',
                    'fileSize': file_size
                }
            else:
                raise Exception("Downloaded file is empty or missing")
                
        except Exception as e:
            logger.error(f"❌ Legacy download failed: {e}")
            return None
    
    def process_video(self, video_url: str, output_filename: str) -> Optional[Dict]:
        """
        Main video processing method with multiple strategies
        
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
                raise Exception("Simple processor only supports YouTube videos")
            
            logger.info(f"🎯 Processing video with multiple strategies: {video_url}")
            
            # Add initial delay
            self.add_random_delay(1, 2)
            
            # Strategy 1: Advanced configuration
            logger.info("🔄 Strategy 1: Advanced configuration...")
            result = self.download_with_advanced_config(video_url, output_filename)
            if result and result.get('success'):
                logger.info("✅ Advanced strategy successful")
                return {
                    **result,
                    'method': 'advanced',
                    'bypass_success': True
                }
            
            # Strategy 2: Mobile configuration
            logger.info("🔄 Strategy 2: Mobile configuration...")
            self.add_random_delay(1, 2)
            result = self.download_with_mobile_config(video_url, output_filename)
            if result and result.get('success'):
                logger.info("✅ Mobile strategy successful")
                return {
                    **result,
                    'method': 'mobile',
                    'bypass_success': True
                }
            
            # Strategy 3: Minimal configuration
            logger.info("🔄 Strategy 3: Minimal configuration...")
            self.add_random_delay(1, 2)
            result = self.download_with_minimal_config(video_url, output_filename)
            if result and result.get('success'):
                logger.info("✅ Minimal strategy successful")
                return {
                    **result,
                    'method': 'minimal',
                    'bypass_success': True
                }
            
            # Strategy 4: Legacy configuration
            logger.info("🔄 Strategy 4: Legacy configuration...")
            self.add_random_delay(1, 2)
            result = self.download_with_legacy_config(video_url, output_filename)
            if result and result.get('success'):
                logger.info("✅ Legacy strategy successful")
                return {
                    **result,
                    'method': 'legacy',
                    'bypass_success': True
                }
            
            # All strategies failed
            logger.error("❌ All download strategies failed")
            raise Exception("All download strategies failed. Video may be protected or unavailable.")
            
        except Exception as e:
            logger.error(f"❌ Video processing error: {e}")
            raise e
    
    def get_video_info(self, video_url: str) -> Optional[Dict]:
        """Get video information using simple configuration"""
        try:
            if not self.is_youtube_url(video_url):
                logger.error(f"❌ Non-YouTube URL: {video_url}")
                raise Exception("Simple processor only supports YouTube videos")
            
            logger.info(f"📋 Getting video info: {video_url}")
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'http_headers': self.get_advanced_headers(),
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
            if info:
                logger.info("✅ Video info retrieved successfully")
                return info
            else:
                raise Exception("Failed to extract video info")
            
        except Exception as e:
            logger.error(f"❌ Video info error: {e}")
            raise e

# Example usage and testing
if __name__ == "__main__":
    # Test the Simple processor
    processor = SimpleVideoProcessor()
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("🧪 Testing Simple Video Processor...")
    
    # Get video info
    try:
        info = processor.get_video_info(test_url)
        if info:
            print(f"📹 Video info: {info.get('title', 'Unknown')}")
    except Exception as e:
        print(f"❌ Video info failed: {e}")
    
    # Test download (commented out to avoid actual download)
    # result = processor.process_video(test_url, "/tmp/test_video.mp4")
    # if result:
    #     print(f"✅ Download result: {result}")
    # else:
    #     print("❌ Download failed") 