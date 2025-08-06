#!/usr/bin/env python3
"""
Stealth Video Processor - YouTube Download with Undetected Chrome + Selenium Stealth
Bypasses bot detection using advanced browser automation techniques
"""

import os
import tempfile
import logging
import time
import random
import requests
import json
import subprocess
from typing import Dict, Optional, List
from urllib.parse import urlparse
import yt_dlp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StealthVideoProcessor:
    def __init__(self):
        """
        Initialize Stealth video processor with undetected Chrome
        """
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        logger.info("🔧 Initializing Stealth Video Processor...")
        
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
    
    def create_stealth_driver(self):
        """Create undetected Chrome driver with stealth patches"""
        try:
            import undetected_chromedriver as uc
            from selenium_stealth import stealth
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            
            logger.info("🔧 Setting up undetected Chrome driver...")
            
            # Set up Chrome options
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            chrome_options.add_argument("--ignore-certificate-errors")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-javascript")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"--user-agent={self.rotate_user_agent()}")
            
            # Add experimental options
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Get Chrome binary path
            chrome_binary = "/usr/bin/google-chrome"
            if not os.path.exists(chrome_binary):
                chrome_binary = "/usr/bin/chromium-browser"
            
            # Set Chrome binary location
            chrome_options.binary_location = chrome_binary
            
            # Initialize driver with explicit binary location
            driver = uc.Chrome(
                options=chrome_options,
                service=Service(ChromeDriverManager().install()),
                browser_executable_path=chrome_binary
            )
            
            # Apply stealth patches
            stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
            )
            
            logger.info("✅ Stealth driver created successfully")
            return driver
            
        except ImportError as e:
            logger.error(f"❌ Missing required packages: {e}")
            logger.info("📦 Installing required packages...")
            self.install_stealth_packages()
            raise Exception("Please restart the application after installing stealth packages")
        except Exception as e:
            logger.error(f"❌ Failed to create stealth driver: {e}")
            raise e
    
    def install_stealth_packages(self):
        """Install required stealth packages"""
        packages = [
            "undetected-chromedriver",
            "selenium-stealth",
            "webdriver-manager"
        ]
        
        for package in packages:
            try:
                logger.info(f"📦 Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             capture_output=True, text=True, check=True)
                logger.info(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                raise Exception(f"Failed to install {package}")
    
    def download_with_stealth(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Download video using stealth browser automation"""
        driver = None
        try:
            logger.info(f"🌐 Starting stealth download: {video_url}")
            
            # Add random delay before starting
            self.add_random_delay(2, 4)
            
            # Create stealth driver
            driver = self.create_stealth_driver()
            
            # Navigate to video URL
            logger.info("🌐 Navigating to video URL...")
            driver.get(video_url)
            
            # Add delay after page load
            self.add_random_delay(3, 6)
            
            # Get page source for analysis
            page_source = driver.page_source
            
            # Check for bot detection
            if "bot" in page_source.lower() or "captcha" in page_source.lower():
                logger.warning("⚠️ Bot detection detected, trying alternative method...")
                return self.download_with_ytdlp_fallback(video_url, output_path)
            
            logger.info("✅ Page loaded successfully, no bot detection")
            
            # Use yt-dlp with stealth headers
            return self.download_with_ytdlp_stealth(video_url, output_path, driver)
            
        except Exception as e:
            logger.error(f"❌ Stealth download failed: {e}")
            # Fallback to yt-dlp with stealth
            return self.download_with_ytdlp_fallback(video_url, output_path)
        finally:
            if driver:
                try:
                    driver.quit()
                    logger.info("🧹 Browser driver closed")
                except:
                    pass
    
    def download_with_ytdlp_stealth(self, video_url: str, output_path: str, driver=None) -> Optional[Dict]:
        """Download using yt-dlp with stealth headers from browser"""
        try:
            logger.info("📥 Using yt-dlp with stealth headers...")
            
            # Get cookies and headers from browser if available
            cookies = {}
            headers = {}
            
            if driver:
                try:
                    # Get cookies from browser
                    browser_cookies = driver.get_cookies()
                    for cookie in browser_cookies:
                        cookies[cookie['name']] = cookie['value']
                    
                    # Get headers
                    headers = {
                        'User-Agent': driver.execute_script("return navigator.userAgent"),
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                    
                    logger.info(f"🍪 Extracted {len(cookies)} cookies from browser")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract browser data: {e}")
            
            # Configure yt-dlp with stealth options
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'cookiefile': None,  # We'll pass cookies directly
                'http_headers': headers,
                'cookiesfrombrowser': None,
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'live'],
                        'player_client': ['android'],
                        'player_skip': ['webpage', 'configs'],
                    }
                }
            }
            
            # Add cookies if available
            if cookies:
                ydl_opts['cookiefile'] = self.create_cookie_file(cookies)
            
            # Download with yt-dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Verify download
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Stealth download completed: {output_path} ({file_size} bytes)")
                return {
                    'success': True,
                    'filePath': output_path,
                    'method': 'stealth-ytdlp',
                    'fileSize': file_size
                }
            else:
                raise Exception("Downloaded file is empty or missing")
                
        except Exception as e:
            logger.error(f"❌ Stealth yt-dlp download failed: {e}")
            return None
    
    def download_with_ytdlp_fallback(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Fallback download using yt-dlp with enhanced headers and cookies"""
        try:
            logger.info("📥 Using yt-dlp fallback with enhanced headers and cookies...")
            
            # Enhanced headers to mimic real browser
            headers = {
                'User-Agent': self.rotate_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'http_headers': headers,
                'cookiesfrombrowser': ('chrome',),  # Try to get cookies from Chrome
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'live'],
                        'player_client': ['android'],
                        'player_skip': ['webpage', 'configs'],
                    }
                }
            }
            
            # Try with cookies first
            try:
                logger.info("🍪 Attempting download with browser cookies...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path)
                    logger.info(f"✅ Cookie-based download completed: {output_path} ({file_size} bytes)")
                    return {
                        'success': True,
                        'filePath': output_path,
                        'method': 'cookie-ytdlp',
                        'fileSize': file_size
                    }
            except Exception as cookie_error:
                logger.warning(f"⚠️ Cookie-based download failed: {cookie_error}")
            
            # Fallback to basic download without cookies
            logger.info("📥 Attempting basic download without cookies...")
            basic_opts = {
                'outtmpl': output_path,
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'http_headers': headers,
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'live'],
                        'player_client': ['android'],
                        'player_skip': ['webpage', 'configs'],
                    }
                }
            }
            
            with yt_dlp.YoutubeDL(basic_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Basic download completed: {output_path} ({file_size} bytes)")
                return {
                    'success': True,
                    'filePath': output_path,
                    'method': 'basic-ytdlp',
                    'fileSize': file_size
                }
            else:
                raise Exception("Downloaded file is empty or missing")
                
        except Exception as e:
            logger.error(f"❌ Fallback download failed: {e}")
            return None
    
    def create_cookie_file(self, cookies: Dict) -> str:
        """Create a temporary cookie file for yt-dlp"""
        cookie_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        try:
            for name, value in cookies.items():
                cookie_file.write(f".youtube.com\tTRUE\t/\tTRUE\t{int(time.time() + 86400)}\t{name}\t{value}\n")
            cookie_file.close()
            return cookie_file.name
        except Exception as e:
            logger.error(f"❌ Failed to create cookie file: {e}")
            return None
    
    def download_with_alternative_methods(self, video_url: str, output_path: str) -> Optional[Dict]:
        """Try alternative yt-dlp configurations to bypass bot detection"""
        alternative_configs = [
            {
                'name': 'mobile_client',
                'opts': {
                    'outtmpl': output_path,
                    'format': 'best[ext=mp4]/best',
                    'quiet': True,
                    'no_warnings': True,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1'
                    },
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android'],
                            'player_skip': ['webpage', 'configs'],
                        }
                    }
                }
            },
            {
                'name': 'web_client',
                'opts': {
                    'outtmpl': output_path,
                    'format': 'best[ext=mp4]/best',
                    'quiet': True,
                    'no_warnings': True,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    },
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['web'],
                            'player_skip': ['configs'],
                        }
                    }
                }
            },
            {
                'name': 'minimal_config',
                'opts': {
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
            }
        ]
        
        for config in alternative_configs:
            try:
                logger.info(f"🔄 Trying alternative method: {config['name']}")
                
                with yt_dlp.YoutubeDL(config['opts']) as ydl:
                    ydl.download([video_url])
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path)
                    logger.info(f"✅ {config['name']} method successful: {output_path} ({file_size} bytes)")
                    return {
                        'success': True,
                        'filePath': output_path,
                        'method': f'alternative-{config["name"]}',
                        'fileSize': file_size
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ {config['name']} method failed: {e}")
                continue
        
        return None

    def process_video(self, video_url: str, output_filename: str) -> Optional[Dict]:
        """
        Main video processing method with stealth techniques
        
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
                raise Exception("Stealth processor only supports YouTube videos")
            
            logger.info(f"🎯 Processing video with stealth techniques: {video_url}")
            
            # Try stealth download first
            result = self.download_with_stealth(video_url, output_filename)
            
            if result and result.get('success'):
                logger.info("✅ Stealth download successful")
                return {
                    **result,
                    'method': 'stealth',
                    'bypass_success': True
                }
            
            # If stealth fails, try fallback with cookies
            logger.warning("⚠️ Stealth download failed, trying fallback with cookies...")
            fallback_result = self.download_with_ytdlp_fallback(video_url, output_filename)
            
            if fallback_result and fallback_result.get('success'):
                logger.info("✅ Fallback download successful")
                return {
                    **fallback_result,
                    'method': 'fallback',
                    'bypass_success': True
                }
            
            # Try alternative methods
            logger.warning("⚠️ Fallback failed, trying alternative methods...")
            alternative_result = self.download_with_alternative_methods(video_url, output_filename)
            
            if alternative_result and alternative_result.get('success'):
                logger.info("✅ Alternative method successful")
                return {
                    **alternative_result,
                    'method': 'alternative',
                    'bypass_success': True
                }
            
            # All methods failed
            logger.error("❌ All download methods failed")
            raise Exception("All download methods failed. Video may be protected or unavailable.")
            
        except Exception as e:
            logger.error(f"❌ Video processing error: {e}")
            raise e
    
    def get_video_info(self, video_url: str) -> Optional[Dict]:
        """Get video information using stealth techniques"""
        try:
            if not self.is_youtube_url(video_url):
                logger.error(f"❌ Non-YouTube URL: {video_url}")
                raise Exception("Stealth processor only supports YouTube videos")
            
            logger.info(f"📋 Getting video info with stealth: {video_url}")
            
            # Use yt-dlp to get video info with stealth headers
            headers = {
                'User-Agent': self.rotate_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'http_headers': headers,
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
    import sys
    
    # Test the Stealth processor
    processor = StealthVideoProcessor()
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("🧪 Testing Stealth Video Processor...")
    
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