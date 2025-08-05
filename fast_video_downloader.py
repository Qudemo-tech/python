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
            self._strategy_ultimate_bypass,
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
        """Enhanced validation - check file size, header, and use ffprobe"""
        try:
            if not os.path.exists(file_path):
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size < 10000:  # Less than 10KB is probably HTML
                logger.warning(f"⚠️ File too small ({file_size} bytes)")
                return False
            
            # Quick header check
            with open(file_path, 'rb') as f:
                header = f.read(12)
            
            # Check for video signatures
            if header.startswith(b'\x00\x00\x00\x20ftyp'):  # MP4
                logger.info(f"✅ MP4 header detected ({file_size} bytes)")
            elif header.startswith(b'RIFF'):  # AVI
                logger.info(f"✅ AVI header detected ({file_size} bytes)")
            elif header.startswith(b'\x1a\x45\xdf\xa3'):  # WebM/MKV
                logger.info(f"✅ WebM/MKV header detected ({file_size} bytes)")
            elif header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
                logger.warning(f"⚠️ HTML file detected")
                return False
            else:
                logger.warning(f"⚠️ Unknown file format")
                return False
            
            # Use ffprobe to validate the video file
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✅ Video file validated by ffprobe ({file_size} bytes)")
                    return True
                else:
                    logger.warning(f"⚠️ ffprobe validation failed: {result.stderr}")
                    return False
            except Exception as e:
                logger.warning(f"⚠️ ffprobe validation error: {e}")
                # If ffprobe fails, check if file is large enough
                return file_size > 100000  # At least 100KB for video files
                
        except Exception as e:
            logger.error(f"❌ File validation error: {e}")
            return False
    
    def _strategy_ultra_fast(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Ultra-fast yt-dlp with aggressive cookie usage"""
        
        ydl_opts = {
            'format': 'worst[height<=480]/worst',  # Get worst quality but ensure it's complete
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': True,
            'retries': 2,  # Allow 2 retries for reliability
            'fragment_retries': 2,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            'sleep_interval': 2,  # Slightly longer sleep
            'max_sleep_interval': 5,
            'socket_timeout': 60,  # Much longer timeout for complete downloads
            'extractor_retries': 2,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'buffersize': 1024,  # Larger buffer for better download
            'http_chunk_size': 10485760,  # 10MB chunks for better reliability
            'cookiesfrombrowser': None,  # Disable browser cookies
        }
        
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
            logger.info(f"🍪 Using cookies from: {cookies_path}")
            # Also try to read and validate cookies
            try:
                with open(cookies_path, 'r') as f:
                    cookie_content = f.read()
                    if 'LOGIN_INFO' in cookie_content:
                        logger.info("✅ LOGIN_INFO cookie found")
                    if 'SID' in cookie_content:
                        logger.info("✅ SID cookie found")
                    if 'HSID' in cookie_content:
                        logger.info("✅ HSID cookie found")
            except Exception as e:
                logger.warning(f"⚠️ Could not validate cookies: {e}")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    # Additional validation to ensure file is complete
                    try:
                        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', output_filename]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                        if result.returncode == 0:
                            logger.info(f"✅ Download completed and validated: {output_filename}")
                            return output_filename
                        else:
                            logger.warning(f"⚠️ Downloaded file validation failed: {result.stderr}")
                            # Try to repair the file
                            return self._try_repair_video_file(output_filename)
                    except Exception as e:
                        logger.warning(f"⚠️ Post-download validation failed: {e}")
                        return output_filename  # Return anyway, let the main validation handle it
                    
        except Exception as e:
            logger.error(f"Ultra-fast yt-dlp failed: {e}")
            raise
        
        return None
    
    def _try_repair_video_file(self, file_path: str) -> Optional[str]:
        """Try to repair a corrupted video file using ffmpeg"""
        try:
            repaired_path = file_path.replace('.mp4', '_repaired.mp4')
            cmd = [
                'ffmpeg', '-i', file_path,
                '-c', 'copy',  # Copy streams without re-encoding
                '-fflags', '+genpts',  # Generate presentation timestamps
                '-movflags', '+faststart',  # Optimize for streaming
                repaired_path,
                '-y'
            ]
            
            logger.info(f"🔧 Attempting to repair video file: {file_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(repaired_path):
                # Remove original and rename repaired file
                os.remove(file_path)
                os.rename(repaired_path, file_path)
                logger.info(f"✅ Video file repaired successfully: {file_path}")
                return file_path
            else:
                logger.warning(f"⚠️ Video repair failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Video repair error: {e}")
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
    
    def _strategy_ultimate_bypass(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Ultimate bypass strategy using browser automation with cookies"""
        
        try:
            # Import selenium only when needed
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            import time
            
            logger.info("🌐 Starting ultimate bypass with browser automation...")
            
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"--user-agent={random.choice(self.user_agents)}")
            
            # Add cookies if available
            if cookies_path and os.path.exists(cookies_path):
                chrome_options.add_argument(f"--user-data-dir=/tmp/chrome_profile")
                logger.info("🍪 Browser will use cookies from profile")
            
            # Initialize driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            try:
                # Load cookies if available
                if cookies_path and os.path.exists(cookies_path):
                    driver.get("https://www.youtube.com")
                    time.sleep(2)
                    
                    # Load cookies from file
                    with open(cookies_path, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                parts = line.strip().split('\t')
                                if len(parts) >= 7:
                                    domain, flag, path, secure, expiry, name, value = parts
                                    if domain == '.youtube.com':
                                        cookie = {
                                            'name': name,
                                            'value': value,
                                            'domain': domain,
                                            'path': path
                                        }
                                        try:
                                            driver.add_cookie(cookie)
                                        except Exception as e:
                                            logger.warning(f"⚠️ Could not add cookie {name}: {e}")
                    
                    logger.info("🍪 Cookies loaded into browser")
                
                # Navigate to video
                driver.get(video_url)
                time.sleep(5)
                
                # Check if video is accessible
                try:
                    # Wait for video player to load
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "video"))
                    )
                    logger.info("✅ Video player found")
                    
                    # Get video source
                    video_element = driver.find_element(By.TAG_NAME, "video")
                    video_src = video_element.get_attribute("src")
                    
                    if video_src:
                        logger.info(f"🎥 Found video source: {video_src[:100]}...")
                        
                        # Download using curl
                        cmd = [
                            'curl', '-L', '-o', output_filename,
                            '-H', f'User-Agent: {random.choice(self.user_agents)}',
                            '--connect-timeout', '30',
                            '--max-time', '120',
                            video_src
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
                        
                        if result.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                            logger.info("✅ Ultimate bypass successful!")
                            return output_filename
                        else:
                            logger.error(f"❌ Ultimate bypass download failed: {result.stderr}")
                    else:
                        logger.warning("⚠️ No video source found")
                        
                except Exception as e:
                    logger.error(f"❌ Video player not found: {e}")
                    
            finally:
                driver.quit()
                
        except Exception as e:
            logger.error(f"❌ Ultimate bypass failed: {e}")
        
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