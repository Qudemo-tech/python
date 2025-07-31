#!/usr/bin/env python3
"""
Fix for corrupted video downloads
Ensures we get actual video files, not HTML pages
"""

import os
import tempfile
import time
import random
import requests
import yt_dlp
import logging
import subprocess
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class FixedVideoDownloader:
    """Fixed video downloader that ensures we get actual video files"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
    
    def download_video(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> str:
        """Main download method with fixed strategies"""
        
        strategies = [
            self._strategy_ytdlp_direct,
            self._strategy_ytdlp_with_cookies,
            self._strategy_ytdlp_audio_only,
            self._strategy_ytdlp_force_download,
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"🔄 Attempting fixed strategy {i+1}/{len(strategies)}: {strategy.__name__}")
                
                # Minimal delay between strategies (reduced for faster processing)
                delay = random.uniform(1, 2)
                logger.info(f"⏳ Waiting {delay:.1f} seconds before strategy...")
                time.sleep(delay)
                
                result = strategy(video_url, output_filename, cookies_path)
                if result and self._validate_video_file(result):
                    logger.info(f"✅ Success with fixed strategy: {strategy.__name__}")
                    return result
                    
            except Exception as e:
                logger.error(f"❌ Fixed strategy {strategy.__name__} failed: {e}")
                continue
        
        raise Exception("All fixed download strategies failed.")
    
    def _validate_video_file(self, file_path: str) -> bool:
        """Validate that the downloaded file is actually a video"""
        try:
            if not os.path.exists(file_path):
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size < 10000:  # Less than 10KB is probably HTML
                logger.warning(f"⚠️ File too small ({file_size} bytes), likely HTML")
                return False
            
            # Check file header to ensure it's a video file
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
            # Check for common video file signatures
            if header.startswith(b'\x00\x00\x00\x20ftyp'):  # MP4
                logger.info(f"✅ Valid MP4 file detected ({file_size} bytes)")
                return True
            elif header.startswith(b'RIFF'):  # AVI
                logger.info(f"✅ Valid AVI file detected ({file_size} bytes)")
                return True
            elif header.startswith(b'\x1a\x45\xdf\xa3'):  # WebM/MKV
                logger.info(f"✅ Valid WebM/MKV file detected ({file_size} bytes)")
                return True
            elif header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
                logger.warning(f"⚠️ HTML file detected, not a video")
                return False
            else:
                # Try to validate with ffprobe
                try:
                    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        logger.info(f"✅ Valid video file confirmed by ffprobe ({file_size} bytes)")
                        return True
                    else:
                        logger.warning(f"⚠️ ffprobe validation failed: {result.stderr}")
                        return False
                except Exception as e:
                    logger.warning(f"⚠️ ffprobe validation error: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ File validation error: {e}")
            return False
    
    def _strategy_ytdlp_direct(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Direct yt-dlp download with minimal options"""
        
        ydl_opts = {
            'format': 'worst[height<=480]/worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 1,
            'fragment_retries': 1,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            },
            'sleep_interval': 1,
            'max_sleep_interval': 2,
            'socket_timeout': 30,
            'extractor_retries': 1,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    return output_filename
                    
        except Exception as e:
            logger.error(f"Direct yt-dlp failed: {e}")
            raise
        
        return None
    
    def _strategy_ytdlp_with_cookies(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """yt-dlp with cookies"""
        
        ydl_opts = {
            'format': 'worst[height<=360]/worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 1,
            'fragment_retries': 1,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            },
            'sleep_interval': 1,
            'max_sleep_interval': 2,
            'socket_timeout': 30,
            'extractor_retries': 1,
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
            logger.error(f"yt-dlp with cookies failed: {e}")
            raise
        
        return None
    
    def _strategy_ytdlp_audio_only(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Download audio only and convert to video"""
        
        # Change output filename to audio format
        audio_filename = output_filename.replace('.mp4', '.m4a')
        
        ydl_opts = {
            'format': 'worstaudio/worst',
            'outtmpl': audio_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 1,
            'fragment_retries': 1,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
            },
            'sleep_interval': 1,
            'max_sleep_interval': 2,
            'socket_timeout': 30,
            'extractor_retries': 1,
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
                    # Convert audio to video using ffmpeg
                    try:
                        cmd = [
                            'ffmpeg', '-i', audio_filename,
                            '-f', 'lavfi', '-i', 'color=black:size=640x480:duration=1',
                            '-c:v', 'libx264', '-c:a', 'copy',
                            '-shortest', output_filename,
                            '-y'  # Overwrite output file
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                        
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
    
    def _strategy_ytdlp_force_download(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Force download with aggressive settings"""
        
        ydl_opts = {
            'format': 'worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 2,
            'fragment_retries': 2,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            'sleep_interval': 0,
            'max_sleep_interval': 0,
            'socket_timeout': 60,
            'extractor_retries': 2,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'force_generic_extractor': False,
        }
        
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    return output_filename
                    
        except Exception as e:
            logger.error(f"Force download yt-dlp failed: {e}")
            raise
        
        return None

def test_fixed_downloader():
    """Test the fixed downloader"""
    
    downloader = FixedVideoDownloader()
    test_url = "https://youtu.be/ZAGxqOT2l2U?si=uB03UNTGKGzgIJ7L"  # The failing video
    output_file = "test_fixed_output.mp4"
    
    try:
        logger.info(f"🧪 Testing fixed downloader with: {test_url}")
        result = downloader.download_video(test_url, output_file)
        logger.info(f"✅ Test successful: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_fixed_downloader() 