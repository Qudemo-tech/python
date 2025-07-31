#!/usr/bin/env python3
"""
Fast video downloader - prioritizes speed over thoroughness
"""

import os
import time
import random
import yt_dlp
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

class FastVideoDownloader:
    """Fast video downloader with minimal delays and aggressive settings"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
    
    def download_video(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> str:
        """Fast download with minimal strategies"""
        
        strategies = [
            self._strategy_ultra_fast,
            self._strategy_curl_fast,
            self._strategy_audio_only,
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"🚀 Fast strategy {i+1}/{len(strategies)}: {strategy.__name__}")
                
                # Minimal delay
                if i > 0:
                    time.sleep(1)
                
                result = strategy(video_url, output_filename, cookies_path)
                if result and self._validate_video_file(result):
                    logger.info(f"✅ Fast download successful: {strategy.__name__}")
                    return result
                    
            except Exception as e:
                logger.error(f"❌ Fast strategy {strategy.__name__} failed: {e}")
                continue
        
        raise Exception("All fast download strategies failed.")
    
    def _validate_video_file(self, file_path: str) -> bool:
        """Quick validation - just check file size and basic header"""
        try:
            if not os.path.exists(file_path):
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size < 10000:  # Less than 10KB is probably HTML
                return False
            
            # Quick header check
            with open(file_path, 'rb') as f:
                header = f.read(12)
            
            # Check for video signatures
            if header.startswith(b'\x00\x00\x00\x20ftyp'):  # MP4
                return True
            elif header.startswith(b'RIFF'):  # AVI
                return True
            elif header.startswith(b'\x1a\x45\xdf\xa3'):  # WebM/MKV
                return True
            elif header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
                return False
            
            # If we can't determine, assume it's valid if it's large enough
            return file_size > 50000  # At least 50KB
                
        except Exception as e:
            logger.error(f"❌ File validation error: {e}")
            return False
    
    def _strategy_ultra_fast(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Ultra-fast yt-dlp with minimal options"""
        
        ydl_opts = {
            'format': 'worst',  # Get worst quality for speed
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 0,  # No retries
            'fragment_retries': 0,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
            },
            'sleep_interval': 0,  # No sleep
            'max_sleep_interval': 0,
            'socket_timeout': 15,  # Short timeout
            'extractor_retries': 0,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
        }
        
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    return output_filename
                    
        except Exception as e:
            logger.error(f"Ultra-fast yt-dlp failed: {e}")
            raise
        
        return None
    
    def _strategy_curl_fast(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Fast curl download"""
        
        try:
            user_agent = random.choice(self.user_agents)
            
            cmd = [
                'curl', '-L', '-o', output_filename,
                '-H', f'User-Agent: {user_agent}',
                '--connect-timeout', '10',
                '--max-time', '30',
                '--retry', '0',
            ]
            
            if cookies_path and os.path.exists(cookies_path):
                cmd.extend(['-b', cookies_path])
            
            cmd.append(video_url)
            
            logger.info(f"🔄 Running fast curl command")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            
            if result.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                return output_filename
            else:
                logger.error(f"❌ Fast curl failed: {result.stderr}")
                raise Exception(f"Fast curl download failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ Fast curl strategy failed: {e}")
            raise
    
    def _strategy_audio_only(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Download audio only and convert to video (fastest option)"""
        
        # Change output filename to audio format
        audio_filename = output_filename.replace('.mp4', '.m4a')
        
        ydl_opts = {
            'format': 'worstaudio/worst',
            'outtmpl': audio_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 0,
            'fragment_retries': 0,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
            },
            'sleep_interval': 0,
            'max_sleep_interval': 0,
            'socket_timeout': 15,
            'extractor_retries': 0,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
        }
        
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
                if os.path.exists(audio_filename) and os.path.getsize(audio_filename) > 0:
                    # Convert audio to video using ffmpeg (fast)
                    try:
                        cmd = [
                            'ffmpeg', '-i', audio_filename,
                            '-f', 'lavfi', '-i', 'color=black:size=640x480:duration=1',
                            '-c:v', 'libx264', '-c:a', 'copy',
                            '-shortest', output_filename,
                            '-y'
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0 and os.path.exists(output_filename):
                            # Clean up audio file
                            os.remove(audio_filename)
                            return output_filename
                        else:
                            logger.error(f"FFmpeg conversion failed: {result.stderr}")
                            
                    except Exception as e:
                        logger.error(f"FFmpeg conversion failed: {e}")
                    
                    # If ffmpeg fails, just return the audio file
                    if os.path.exists(audio_filename):
                        os.rename(audio_filename, output_filename)
                        return output_filename
                    
        except Exception as e:
            logger.error(f"Audio-only yt-dlp failed: {e}")
            raise
        
        return None

def test_fast_downloader():
    """Test the fast downloader"""
    
    downloader = FastVideoDownloader()
    test_url = "https://youtu.be/ZAGxqOT2l2U?si=uB03UNTGKGzgIJ7L"
    output_file = "test_fast_output.mp4"
    
    try:
        logger.info(f"🧪 Testing fast downloader with: {test_url}")
        result = downloader.download_video(test_url, output_file)
        logger.info(f"✅ Fast download test successful: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Fast download test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_fast_downloader() 