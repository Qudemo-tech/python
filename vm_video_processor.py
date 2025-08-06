#!/usr/bin/env python3
"""
Direct VM Video Processor - YouTube Download with Direct GCP VM Access
Bypasses bot detection by using a clean VM environment directly
"""

import os
import tempfile
import logging
import subprocess
import json
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectVMVideoProcessor:
    def __init__(self):
        """
        Initialize Direct VM video processor
        """
        # VM configuration from environment variables
        self.vm_project_id = os.getenv('GCP_PROJECT_ID')
        self.vm_name = os.getenv('GCP_VM_NAME', 'youtube-downloader-vm')
        self.vm_zone = os.getenv('GCP_VM_ZONE', 'us-central1-a')
        self.vm_user = os.getenv('GCP_VM_USER', 'abhis')
        
        logger.info(f"🔧 VM Configuration: {self.vm_name} in {self.vm_zone} (Project: {self.vm_project_id})")
        
    def is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        domain = urlparse(url).netloc.lower()
        return 'youtube.com' in domain or 'youtu.be' in domain
    
    def check_vm_health(self) -> bool:
        """Check if VM is accessible and healthy"""
        try:
            logger.info("🔍 Checking VM health...")
            
            # Check if gcloud is available
            result = subprocess.run(['which', 'gcloud'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("❌ gcloud CLI not available on this system")
                return False
            
            # Test VM connectivity
            ssh_command = [
                'gcloud', 'compute', 'ssh', f'{self.vm_user}@{self.vm_name}',
                '--zone', self.vm_zone,
                '--command', 'yt-dlp --version'
            ]
            
            result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"✅ VM health check passed: yt-dlp version {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ VM health check failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ VM health check timed out")
            return False
        except Exception as e:
            logger.error(f"❌ VM health check error: {e}")
            return False
    
    def download_with_vm(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Download video using direct VM access"""
        try:
            logger.info(f"🌐 Downloading with direct VM access: {video_url}")
            
            # Validate URL
            if not self.is_youtube_url(video_url):
                raise Exception("Direct VM processor only supports YouTube videos")
            
            # Check VM health first
            if not self.check_vm_health():
                raise Exception("VM is not accessible or healthy")
            
            # Extract filename from output_path
            import pathlib
            filename = pathlib.Path(output_path).name
            vm_filename = f"vm_{int(time.time())}_{filename}"
            
            logger.info(f"📥 Step 1: Downloading to VM as {vm_filename}...")
            
            # Step 1: Download to VM
            download_command = [
                'gcloud', 'compute', 'ssh', f'{self.vm_user}@{self.vm_name}',
                '--zone', self.vm_zone,
                '--command', f'cd ~/youtube-downloader && /home/abhis/.local/bin/yt-dlp --output {vm_filename} {video_url}'
            ]
            
            result = subprocess.run(download_command, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"❌ VM download failed: {result.stderr}")
                raise Exception(f"VM download failed: {result.stderr}")
            
            logger.info(f"✅ VM download successful: {vm_filename}")
            
            # Step 2: Copy file from VM to local
            logger.info(f"📋 Step 2: Copying file from VM to local...")
            copy_command = [
                'gcloud', 'compute', 'scp',
                f'{self.vm_user}@{self.vm_name}:/home/abhis/youtube-downloader/{vm_filename}',
                output_path,
                '--zone', self.vm_zone
            ]
            
            result = subprocess.run(copy_command, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"❌ File copy failed: {result.stderr}")
                raise Exception(f"File copy failed: {result.stderr}")
            
            logger.info(f"✅ File copy successful: {output_path}")
            
            # Step 3: Clean up file on VM
            logger.info(f"🧹 Step 3: Cleaning up file on VM...")
            cleanup_command = [
                'gcloud', 'compute', 'ssh', f'{self.vm_user}@{self.vm_name}',
                '--zone', self.vm_zone,
                '--command', f'cd ~/youtube-downloader && rm -f {vm_filename}'
            ]
            
            subprocess.run(cleanup_command, capture_output=True, text=True, timeout=30)
            logger.info("✅ Cleanup successful")
            
            # Check if file was actually copied
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Direct VM download completed: {output_path} ({file_size} bytes)")
                return {
                    'success': True,
                    'filePath': output_path,
                    'method': 'direct-vm',
                    'fileSize': file_size
                }
            else:
                raise Exception("File copy completed but file is missing or empty")
                
        except Exception as e:
            logger.error(f"❌ Direct VM download error: {e}")
            return None
    
    def process_video(self, video_url: str, output_filename: str) -> Optional[Dict]:
        """
        Main video processing method with direct VM access
        
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
                raise Exception("Direct VM processor only supports YouTube videos")
            
            # Check VM health
            if not self.check_vm_health():
                logger.error("❌ VM is not accessible or healthy")
                raise Exception("VM is not accessible or healthy. Please check VM configuration and connectivity.")
            
            # Direct VM download
            logger.info("🌐 Attempting direct VM download...")
            vm_result = self.download_with_vm(video_url, output_filename)
            
            if vm_result and vm_result.get('success'):
                logger.info("✅ Direct VM download successful")
                return {
                    **vm_result,
                    'method': 'direct-vm',
                    'bypass_success': True
                }
            
            # VM download failed
            logger.error("❌ Direct VM download failed")
            raise Exception("Direct VM download failed. This is the only available method.")
            
        except Exception as e:
            logger.error(f"❌ Video processing error: {e}")
            raise e
    
    def get_video_info(self, video_url: str) -> Optional[Dict]:
        """Get video information using direct VM access"""
        try:
            if not self.is_youtube_url(video_url):
                logger.error(f"❌ Non-YouTube URL: {video_url}")
                raise Exception("Direct VM processor only supports YouTube videos")
            
            # Check VM health
            if not self.check_vm_health():
                logger.error("❌ VM is not accessible or healthy")
                raise Exception("VM is not accessible or healthy. Please check VM configuration and connectivity.")
            
            logger.info(f"📋 Getting video info via direct VM: {video_url}")
            
            # Get video info from VM
            info_command = [
                'gcloud', 'compute', 'ssh', f'{self.vm_user}@{self.vm_name}',
                '--zone', self.vm_zone,
                '--command', f'cd ~/youtube-downloader && /home/abhis/.local/bin/yt-dlp --dump-json {video_url}'
            ]
            
            result = subprocess.run(info_command, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                try:
                    video_info = json.loads(result.stdout)
                    logger.info("✅ Video info retrieved via direct VM")
                    return video_info
                except json.JSONDecodeError:
                    logger.error("❌ Failed to parse video info JSON")
                    raise Exception("Failed to parse video info from VM")
            else:
                logger.error(f"❌ VM video info failed: {result.stderr}")
                raise Exception(f"VM video info failed: {result.stderr}")
            
        except Exception as e:
            logger.error(f"❌ Video info error: {e}")
            raise e

# Example usage and testing
if __name__ == "__main__":
    # Test the Direct VM processor
    processor = DirectVMVideoProcessor()
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("🧪 Testing Direct VM Video Processor...")
    
    # Check VM health
    if processor.check_vm_health():
        print("✅ VM is healthy")
        
        # Get video info
        info = processor.get_video_info(test_url)
        if info:
            print(f"📹 Video info: {info.get('title', 'Unknown')}")
        
        # Test download (commented out to avoid actual download)
        # result = processor.process_video(test_url, "/tmp/test_video.mp4")
        # if result:
        #     print(f"✅ Download result: {result}")
        # else:
        #     print("❌ Download failed")
    else:
        print("⚠️ VM is not healthy, check configuration") 