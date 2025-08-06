#!/usr/bin/env python3
"""
Browser-Based Video Processor - Like Loom
Uses Selenium to control a real browser for video processing
"""

import os
import tempfile
import logging
import time
import json
from flask import Flask, request, send_file, jsonify
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to track active downloads
active_downloads = {}
download_lock = threading.Lock()

def setup_chrome_driver():
    """Setup Chrome driver with anti-detection measures"""
    chrome_options = Options()
    
    # Anti-detection options
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # User agent
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Additional options for cloud deployment
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    # Execute script to remove webdriver property
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def download_video_with_browser(url, use_cookies=False):
    """Download video using browser automation"""
    driver = None
    temp_file = None
    
    try:
        # Create temporary file
        temp_fd, temp_file = tempfile.mkstemp(suffix='.mp4', dir='/tmp')
        os.close(temp_fd)
        
        # Setup browser
        driver = setup_chrome_driver()
        
        # Navigate to video
        logger.info(f"Navigating to: {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(5)
        
        # Get video title
        try:
            title_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "title"))
            )
            title = title_element.get_attribute("textContent") or "Downloaded Video"
        except:
            title = "Downloaded Video"
        
        # For YouTube, try to find video element and get source
        if 'youtube.com' in url or 'youtu.be' in url:
            try:
                # Wait for video player to load
                video_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "video"))
                )
                
                # Get video source
                video_src = video_element.get_attribute("src")
                if video_src:
                    logger.info(f"Found video source: {video_src}")
                    # Download using curl/wget
                    subprocess.run(['curl', '-L', '-o', temp_file, video_src], check=True)
                else:
                    raise Exception("No video source found")
                    
            except Exception as e:
                logger.error(f"Failed to get video source: {e}")
                raise Exception("Could not extract video from YouTube")
        
        # For other platforms, use yt-dlp as fallback
        else:
            # Use yt-dlp with browser cookies
            cmd = [
                'yt-dlp',
                '--format', 'best[height<=720]',
                '--output', temp_file.replace('.mp4', '.%(ext)s'),
                '--cookies-from-browser', 'chrome',
                '--no-warnings',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"yt-dlp failed: {result.stderr}")
        
        # Check if file was downloaded
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
            # Check for other extensions
            base_name = temp_file.replace('.mp4', '')
            for ext in ['.webm', '.mkv', '.avi', '.mov', '.flv']:
                test_file = base_name + ext
                if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
                    temp_file = test_file
                    break
        
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
            raise Exception("No video file downloaded")
        
        logger.info(f"Download completed: {temp_file} (size: {os.path.getsize(temp_file)})")
        return temp_file, title
        
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        raise e
    finally:
        if driver:
            driver.quit()

def cleanup_temp_file(file_path):
    """Safely remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up {file_path}: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'browser-video-processor',
        'timestamp': time.time(),
        'method': 'browser-automation'
    })

@app.route('/download', methods=['GET'])
def download():
    """Browser-based download endpoint"""
    url = request.args.get('url')
    use_cookies = request.args.get('use_cookies', 'false').lower() == 'true'
    
    if not url:
        return jsonify({'error': 'URL parameter is required'}), 400
    
    # Check for concurrent downloads
    with download_lock:
        if url in active_downloads:
            return jsonify({'error': 'Download already in progress for this URL'}), 409
        active_downloads[url] = True
    
    temp_file = None
    try:
        # Download using browser automation
        temp_file, title = download_video_with_browser(url, use_cookies)
        
        # Send the file
        response = send_file(
            temp_file,
            as_attachment=True,
            download_name=f"{title[:50]}.mp4",
            mimetype='video/mp4'
        )
        
        # Add cleanup callback
        @response.call_on_close
        def cleanup():
            cleanup_temp_file(temp_file)
            with download_lock:
                active_downloads.pop(url, None)
        
        return response
        
    except Exception as e:
        # Clean up on error
        if temp_file:
            cleanup_temp_file(temp_file)
        with download_lock:
            active_downloads.pop(url, None)
        
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get current download status"""
    with download_lock:
        active_count = len(active_downloads)
    
    return jsonify({
        'active_downloads': active_count,
        'active_urls': list(active_downloads.keys())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False) 