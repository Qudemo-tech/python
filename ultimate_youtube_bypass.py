#!/usr/bin/env python3
"""
Ultimate YouTube Bypass Solution
Multiple strategies to bypass YouTube's bot detection and rate limiting
"""

import os
import tempfile
import time
import random
import requests
import yt_dlp
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import subprocess
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class UltimateYouTubeBypass:
    """Comprehensive YouTube bypass solution with multiple strategies"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/118.0 Firefox/119.0',
        ]
        
        self.proxy_list = [
            None,  # Direct connection
            # Add proxy servers here if available
            # 'http://proxy1:port',
            # 'http://proxy2:port',
        ]
    
    def download_video(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> str:
        """Main download method with multiple fallback strategies"""
        
        strategies = [
            self._strategy_ytdlp_advanced,
            self._strategy_ytdlp_mobile,
            self._strategy_ytdlp_minimal,
            self._strategy_browser_automation,
            self._strategy_curl_download,
            self._strategy_wget_download,
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"🔄 Attempting strategy {i+1}/{len(strategies)}: {strategy.__name__}")
                
                # Random delay between strategies
                delay = random.uniform(10, 30)
                logger.info(f"⏳ Waiting {delay:.1f} seconds before strategy...")
                time.sleep(delay)
                
                result = strategy(video_url, output_filename, cookies_path)
                if result:
                    logger.info(f"✅ Success with strategy: {strategy.__name__}")
                    return result
                    
            except Exception as e:
                logger.error(f"❌ Strategy {strategy.__name__} failed: {e}")
                continue
        
        raise Exception("All download strategies failed. YouTube is actively blocking this IP.")
    
    def _strategy_ytdlp_advanced(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Advanced yt-dlp with rotating headers and multiple formats"""
        
        # Rotate user agents
        user_agent = random.choice(self.user_agents)
        
        ydl_opts = {
            'format': 'worst[height<=720]/worst[height<=480]/worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': False,
            'retries': 1,
            'fragment_retries': 1,
            'http_headers': {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            },
            'sleep_interval': random.uniform(5, 15),
            'max_sleep_interval': random.uniform(15, 30),
            'socket_timeout': 60,
            'extractor_retries': 1,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'extract_flat': False,
            'force_generic_extractor': False,
        }
        
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(video_url, download=False)
                logger.info(f"✅ Video info extracted: {info.get('title', 'Unknown')}")
                
                # Download
                ydl.download([video_url])
                
                # Verify file exists
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    return output_filename
                    
        except Exception as e:
            logger.error(f"Advanced yt-dlp failed: {e}")
            raise
        
        return None
    
    def _strategy_ytdlp_mobile(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Mobile browser simulation"""
        
        ydl_opts = {
            'format': 'worst[height<=480]/worst[height<=360]/worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': False,
            'retries': 1,
            'fragment_retries': 1,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'X-Requested-With': 'XMLHttpRequest',
            },
            'sleep_interval': random.uniform(10, 20),
            'max_sleep_interval': random.uniform(20, 40),
            'socket_timeout': 60,
            'extractor_retries': 1,
            'ignoreerrors': False,
        }
        
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    return output_filename
        except Exception as e:
            logger.error(f"Mobile yt-dlp failed: {e}")
            raise
        
        return None
    
    def _strategy_ytdlp_minimal(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Minimal headers approach"""
        
        ydl_opts = {
            'format': 'worst',
            'outtmpl': output_filename,
            'quiet': True,
            'no_warnings': False,
            'retries': 1,
            'fragment_retries': 1,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            },
            'sleep_interval': random.uniform(15, 25),
            'max_sleep_interval': random.uniform(30, 50),
            'socket_timeout': 60,
            'extractor_retries': 1,
            'ignoreerrors': False,
        }
        
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                    return output_filename
        except Exception as e:
            logger.error(f"Minimal yt-dlp failed: {e}")
            raise
        
        return None
    
    def _strategy_browser_automation(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Browser automation using Selenium"""
        
        try:
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={random.choice(self.user_agents)}')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize driver
            driver = webdriver.Chrome(options=chrome_options)
            
            try:
                # Navigate to video
                driver.get(video_url)
                logger.info(f"🌐 Navigated to video page")
                
                # Wait for page to load
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "video"))
                )
                
                # Get video source
                video_element = driver.find_element(By.TAG_NAME, "video")
                video_src = video_element.get_attribute("src")
                
                if video_src:
                    logger.info(f"🎥 Found video source: {video_src}")
                    
                    # Download using requests
                    headers = {'User-Agent': random.choice(self.user_agents)}
                    response = requests.get(video_src, stream=True, timeout=60, headers=headers)
                    response.raise_for_status()
                    
                    with open(output_filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info(f"✅ Video downloaded via browser automation")
                    return output_filename
                else:
                    raise Exception("No video source found in browser")
                    
            finally:
                driver.quit()
                
        except ImportError:
            logger.warning("⚠️ Selenium not available, skipping browser automation")
            raise Exception("Browser automation not available")
        except Exception as e:
            logger.error(f"❌ Browser automation failed: {e}")
            raise
    
    def _strategy_curl_download(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Download using curl command"""
        
        try:
            user_agent = random.choice(self.user_agents)
            cmd = [
                'curl', '-L', '-o', output_filename,
                '-H', f'User-Agent: {user_agent}',
                '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                '--connect-timeout', '30',
                '--max-time', '300',
                '--retry', '1',
                '--retry-delay', '5',
            ]
            
            if cookies_path and os.path.exists(cookies_path):
                cmd.extend(['-b', cookies_path])
            
            cmd.append(video_url)
            
            logger.info(f"🔄 Running curl command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                logger.info(f"✅ Curl download successful")
                return output_filename
            else:
                logger.error(f"❌ Curl failed: {result.stderr}")
                raise Exception(f"Curl download failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ Curl strategy failed: {e}")
            raise
    
    def _strategy_wget_download(self, video_url: str, output_filename: str, cookies_path: Optional[str] = None) -> Optional[str]:
        """Download using wget command"""
        
        try:
            user_agent = random.choice(self.user_agents)
            cmd = [
                'wget', '-O', output_filename,
                '--user-agent', user_agent,
                '--timeout', '30',
                '--tries', '1',
                '--retry-connrefused',
                '--no-check-certificate',
            ]
            
            if cookies_path and os.path.exists(cookies_path):
                cmd.extend(['--load-cookies', cookies_path])
            
            cmd.append(video_url)
            
            logger.info(f"🔄 Running wget command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
                logger.info(f"✅ Wget download successful")
                return output_filename
            else:
                logger.error(f"❌ Wget failed: {result.stderr}")
                raise Exception(f"Wget download failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ Wget strategy failed: {e}")
            raise

def test_youtube_bypass():
    """Test the bypass solution with a sample video"""
    
    bypass = UltimateYouTubeBypass()
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll
    output_file = "test_output.mp4"
    
    try:
        logger.info(f"🧪 Testing YouTube bypass with: {test_url}")
        result = bypass.download_video(test_url, output_file)
        logger.info(f"✅ Test successful: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_youtube_bypass() 