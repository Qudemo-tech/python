#!/usr/bin/env python3
"""
Alternative YouTube Downloader
Uses different approaches to bypass YouTube's blocking
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

class AlternativeYouTubeDownloader:
    """Alternative YouTube downloader using different approaches"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/118.0 Firefox/119.0',
        ]
    
    def download_video(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> str:
        """Main download method with alternative strategies"""
        
        strategies = [
            self._strategy_ytdlp_ultra_minimal,
            self._strategy_ytdlp_no_cookies,
            self._strategy_ytdlp_audio_only,
            self._strategy_ytdlp_force_generic,
            self._strategy_ytdlp_flat_extraction,
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"🔄 Attempting alternative strategy {i+1}/{len(strategies)}: {strategy.__name__}")
                
                # Random delay between strategies
                delay = random.uniform(5, 15)
                logger.info(f"⏳ Waiting {delay:.1f} seconds before strategy...")
                time.sleep(delay)
                
                result = strategy(video_url, output_filename, cookies_path)
                if result:
                    logger.info(f"✅ Success with alternative strategy: {strategy.__name__}")
                    return result
                    
            except Exception as e:
                logger.error(f"❌ Alternative strategy {strategy.__name__} failed: {e}")
                continue
        
        raise Exception("All alternative download strategies failed.")
    
    def _strategy_ytdlp_ultra_minimal(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Ultra minimal yt-dlp with no cookies and minimal options"""
        
        ydl_opts = {
            'format': 'worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 0,
            'fragment_retries': 0,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
            },
            'sleep_interval': 0,
            'max_sleep_interval': 0,
            'socket_timeout': 30,
            'extractor_retries': 0,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'extract_flat': False,
            'force_generic_extractor': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    file_size = os.path.getsize(output_filename)
                    logger.info(f"✅ Ultra minimal download successful ({file_size} bytes)")
                    return output_filename
                    
        except Exception as e:
            logger.error(f"Ultra minimal yt-dlp failed: {e}")
            raise
        
        return None
    
    def _strategy_ytdlp_no_cookies(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """yt-dlp without any cookies or authentication"""
        
        ydl_opts = {
            'format': 'worst[height<=360]/worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 0,
            'fragment_retries': 0,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            },
            'sleep_interval': 0,
            'max_sleep_interval': 0,
            'socket_timeout': 30,
            'extractor_retries': 0,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    file_size = os.path.getsize(output_filename)
                    logger.info(f"✅ No cookies download successful ({file_size} bytes)")
                    return output_filename
                    
        except Exception as e:
            logger.error(f"No cookies yt-dlp failed: {e}")
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
            'retries': 0,
            'fragment_retries': 0,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
            },
            'sleep_interval': 0,
            'max_sleep_interval': 0,
            'socket_timeout': 30,
            'extractor_retries': 0,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
        }
        
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
                            
                            file_size = os.path.getsize(output_filename)
                            logger.info(f"✅ Audio-only download successful ({file_size} bytes)")
                            return output_filename
                        else:
                            logger.error(f"FFmpeg conversion failed: {result.stderr}")
                            
                    except Exception as e:
                        logger.error(f"FFmpeg conversion failed: {e}")
                    
                    # If ffmpeg fails, just return the audio file
                    if os.path.exists(audio_filename):
                        os.rename(audio_filename, output_filename)
                        file_size = os.path.getsize(output_filename)
                        logger.info(f"✅ Audio download successful ({file_size} bytes)")
                        return output_filename
                    
        except Exception as e:
            logger.error(f"Audio-only yt-dlp failed: {e}")
            raise
        
        return None
    
    def _strategy_ytdlp_force_generic(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Force generic extractor"""
        
        ydl_opts = {
            'format': 'worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 0,
            'fragment_retries': 0,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
            },
            'sleep_interval': 0,
            'max_sleep_interval': 0,
            'socket_timeout': 30,
            'extractor_retries': 0,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'force_generic_extractor': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    file_size = os.path.getsize(output_filename)
                    logger.info(f"✅ Force generic download successful ({file_size} bytes)")
                    return output_filename
                    
        except Exception as e:
            logger.error(f"Force generic yt-dlp failed: {e}")
            raise
        
        return None
    
    def _strategy_ytdlp_flat_extraction(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Use flat extraction to get direct URLs"""
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
            },
            'socket_timeout': 30,
            'extractor_retries': 0,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info without downloading
                info = ydl.extract_info(video_url, download=False)
                
                if info and 'url' in info:
                    direct_url = info['url']
                    logger.info(f"🎯 Found direct URL via flat extraction: {direct_url[:100]}...")
                    
                    # Download using requests
                    headers = {'User-Agent': random.choice(self.user_agents)}
                    response = requests.get(direct_url, stream=True, timeout=60, headers=headers)
                    response.raise_for_status()
                    
                    with open(output_filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                        file_size = os.path.getsize(output_filename)
                        logger.info(f"✅ Flat extraction download successful ({file_size} bytes)")
                        return output_filename
                else:
                    raise Exception("No direct URL found in flat extraction")
                    
        except Exception as e:
            logger.error(f"Flat extraction yt-dlp failed: {e}")
            raise
        
        return None

def test_alternative_downloader():
    """Test the alternative downloader"""
    
    downloader = AlternativeYouTubeDownloader()
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll
    output_file = "test_alternative_output.mp4"
    
    try:
        logger.info(f"🧪 Testing alternative downloader with: {test_url}")
        result = downloader.download_video(test_url, output_file)
        logger.info(f"✅ Test successful: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_alternative_downloader() 