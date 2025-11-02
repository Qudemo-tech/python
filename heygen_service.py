"""
HeyGen Service for AI Avatar Video Generation
Handles direct integration with HeyGen API for creating avatar videos
"""

import os
import requests
import asyncio
import logging
from typing import Dict, List, Optional, Any
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class HeyGenService:
    """Service for generating AI avatar videos using HeyGen API"""
    
    def __init__(self):
        """Initialize HeyGen service"""
        self.api_key = os.getenv('HEYGEN_API_KEY')
        if not self.api_key:
            logger.warning("⚠️ HEYGEN_API_KEY not found in .env - HeyGen features will be disabled")
        
        self.upload_url = "https://upload.heygen.com/v1/asset"
        self.generate_url = "https://api.heygen.com/v2/video/av4/generate"
        self.status_url = "https://api.heygen.com/v1/video_status.get"
        
        # Default voice ID (you can make this configurable)
        self.default_voice_id = "466986b2ee27456bbf9757b7ed72c177"
        
        logger.info("✅ HeyGen Service initialized")
    
    def upload_presenter_photo(self, photo_url: str) -> Optional[str]:
        """
        Upload presenter photo to HeyGen and get image_key
        
        Args:
            photo_url: GCS URL of the presenter photo
            
        Returns:
            str: image_key to use for video generation, or None if failed
        """
        try:
            if not self.api_key:
                logger.error("❌ HeyGen API key not configured")
                return None
            
            logger.info(f"📤 Uploading presenter photo to HeyGen: {photo_url}")
            
            # Download photo from GCS
            photo_response = requests.get(photo_url, timeout=30)
            if photo_response.status_code != 200:
                logger.error(f"❌ Failed to download photo from GCS: {photo_response.status_code}")
                return None
            
            photo_data = photo_response.content
            
            # Upload to HeyGen
            headers = {
                "Content-Type": "image/jpeg",
                "x-api-key": self.api_key
            }
            
            response = requests.post(
                self.upload_url,
                data=photo_data,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"📋 HeyGen upload response: {result}")
                
                # HeyGen returns different fields - try image_key, id, or url
                data = result.get('data', {})
                
                # Try different field options
                image_key = data.get('image_key')
                
                if not image_key:
                    # Fallback: try using just the 'id' field with 'image/' prefix
                    image_id = data.get('id')
                    if image_id:
                        image_key = f"image/{image_id}"
                        logger.info(f"🔧 Using image ID as key: {image_key}")
                
                if image_key:
                    logger.info(f"✅ Photo uploaded to HeyGen successfully: {image_key}")
                    return image_key
                else:
                    logger.error(f"❌ No image_key/id in HeyGen response: {result}")
                    return None
            else:
                logger.error(f"❌ HeyGen photo upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error uploading photo to HeyGen: {e}")
            return None
    
    def truncate_script(self, text: str, max_length: int = 1000) -> str:
        """
        Truncate script to max length while keeping it complete
        
        Args:
            text: Original text
            max_length: Maximum character length (default 1000)
            
        Returns:
            str: Truncated text ending at a sentence boundary
        """
        if len(text) <= max_length:
            return text
        
        # Truncate to max length
        truncated = text[:max_length]
        
        # Find last sentence ending (., !, ?)
        last_period = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_period > max_length * 0.7:  # At least 70% of max length
            return truncated[:last_period + 1].strip()
        else:
            # No good sentence boundary found, just truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > 0:
                return truncated[:last_space].strip() + '...'
            else:
                return truncated.strip() + '...'
    
    def generate_video(self, image_key: str, script: str, video_title: str, 
                      voice_id: Optional[str] = None) -> Optional[str]:
        """
        Generate avatar video using HeyGen API
        
        Args:
            image_key: HeyGen image_key from upload (with /original suffix removed)
            script: Text for avatar to speak (max 1000 chars)
            video_title: Title for the video
            voice_id: Optional voice ID (uses default if not provided)
            
        Returns:
            str: video_id if successful, None if failed
        """
        try:
            if not self.api_key:
                logger.error("❌ HeyGen API key not configured")
                return None
            
            # Truncate script to 1000 characters
            truncated_script = self.truncate_script(script, 1000)
            
            if len(script) > len(truncated_script):
                logger.info(f"✂️ Truncated script from {len(script)} to {len(truncated_script)} characters")
            
            logger.info(f"🎬 Generating video: {video_title}")
            
            payload = {
                "video_orientation": "portrait",
                "script": truncated_script,
                "voice_id": voice_id or self.default_voice_id,
                "image_key": image_key,  # HeyGen requires image_key field
                "video_title": video_title
            }
            
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "x-api-key": self.api_key
            }
            
            response = requests.post(
                self.generate_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                video_id = result.get('data', {}).get('video_id')
                
                if video_id:
                    logger.info(f"✅ Video generation started: {video_id}")
                    return video_id
                else:
                    logger.error(f"❌ No video_id in HeyGen response: {result}")
                    return None
            else:
                logger.error(f"❌ HeyGen video generation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating video: {e}")
            return None
    
    def check_video_status(self, video_id: str) -> Dict[str, Any]:
        """
        Check video generation status
        
        Args:
            video_id: HeyGen video ID
            
        Returns:
            dict: Status information including 'status' and 'video_url' if completed
        """
        try:
            if not self.api_key:
                return {"status": "error", "error": "API key not configured"}
            
            params = {"video_id": video_id}
            headers = {
                "Accept": "application/json",
                "X-Api-Key": self.api_key
            }
            
            response = requests.get(
                self.status_url,
                headers=headers,
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                data = result.get('data', {})
                
                return {
                    "status": data.get('status', 'unknown'),
                    "video_url": data.get('video_url'),
                    "thumbnail_url": data.get('thumbnail_url'),
                    "duration": data.get('duration'),
                    "error": data.get('error'),
                    "created_at": data.get('created_at')
                }
            else:
                logger.error(f"❌ Status check failed: {response.status_code}")
                return {"status": "error", "error": response.text}
                
        except Exception as e:
            logger.error(f"❌ Error checking video status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def wait_for_video_completion(self, video_id: str, max_wait_seconds: int = 600, 
                                       poll_interval: int = 10) -> Optional[str]:
        """
        Wait for video to complete and return video URL (async, non-blocking)
        
        Args:
            video_id: HeyGen video ID
            max_wait_seconds: Maximum time to wait (default 10 minutes)
            poll_interval: Seconds between status checks (default 10)
            
        Returns:
            str: video_url if successful, None if failed or timeout
        """
        try:
            start_time = time.time()
            attempts = 0
            
            logger.info(f"⏳ Waiting for video {video_id} to complete (max {max_wait_seconds}s)...")
            
            while True:
                attempts += 1
                elapsed = time.time() - start_time
                
                if elapsed > max_wait_seconds:
                    logger.error(f"❌ Timeout waiting for video {video_id} after {elapsed:.0f}s")
                    return None
                
                status_info = self.check_video_status(video_id)
                status = status_info.get('status')
                
                logger.info(f"📊 Video {video_id} status: {status} (attempt {attempts}, {elapsed:.0f}s elapsed)")
                
                if status == 'completed':
                    video_url = status_info.get('video_url')
                    logger.info(f"✅ Video completed: {video_url}")
                    return video_url
                elif status == 'failed' or status == 'error':
                    error = status_info.get('error', 'Unknown error')
                    logger.error(f"❌ Video generation failed: {error}")
                    return None
                elif status in ['processing', 'pending', 'unknown']:
                    # Still processing, wait and retry (non-blocking)
                    await asyncio.sleep(poll_interval)
                else:
                    logger.warning(f"⚠️ Unknown status: {status}")
                    await asyncio.sleep(poll_interval)
                    
        except Exception as e:
            logger.error(f"❌ Error waiting for video completion: {e}")
            return None
    
    def download_video(self, video_url: str) -> Optional[bytes]:
        """
        Download video from HeyGen URL
        
        Args:
            video_url: HeyGen video URL
            
        Returns:
            bytes: Video file content, or None if failed
        """
        try:
            logger.info(f"⬇️ Downloading video from HeyGen...")
            
            response = requests.get(video_url, stream=True, timeout=120)
            
            if response.status_code == 200:
                video_data = b''
                for chunk in response.iter_content(chunk_size=8192):
                    video_data += chunk
                
                logger.info(f"✅ Video downloaded: {len(video_data)} bytes")
                return video_data
            else:
                logger.error(f"❌ Failed to download video: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error downloading video: {e}")
            return None

