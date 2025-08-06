#!/usr/bin/env python3
"""
Ultimate Video Processor - Multi-Platform Download Service
Supports YouTube, Vimeo, Dailymotion, and other platforms
Uses multiple bypass techniques for reliable cloud hosting
"""

import os
import tempfile
import logging
import random
import time
import json
import requests
from flask import Flask, request, send_file, jsonify
import yt_dlp
import threading
from urllib.parse import urlparse
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to track active downloads
active_downloads = {}
download_lock = threading.Lock()

def cleanup_temp_file(file_path):
    """Safely remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up {file_path}: {e}")

def get_rotating_headers():
    """Generate rotating headers to avoid detection"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
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
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }

def detect_platform(url):
    """Detect video platform from URL"""
    domain = urlparse(url).netloc.lower()
    
    if 'youtube.com' in domain or 'youtu.be' in domain:
        return 'youtube'
    elif 'vimeo.com' in domain:
        return 'vimeo'
    elif 'dailymotion.com' in domain or 'dai.ly' in domain:
        return 'dailymotion'
    elif 'bilibili.com' in domain:
        return 'bilibili'
    else:
        return 'unknown'

def download_with_yt_dlp_advanced(url, use_cookies=False, retry_count=0):
    """Advanced yt-dlp download with multiple bypass techniques"""
    temp_file = None
    max_retries = 3
    platform = detect_platform(url)
    
    try:
        # Create temporary file
        temp_fd, temp_file = tempfile.mkstemp(suffix='.mp4', dir='/tmp')
        os.close(temp_fd)
        
        # Random delay
        time.sleep(random.uniform(1, 3))
        
        # Platform-specific configurations
        if platform == 'youtube':
            ydl_opts = {
                'format': 'best[height<=720]/best',
                'outtmpl': temp_file.replace('.mp4', '.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
                
                # Anti-bot headers
                'http_headers': get_rotating_headers(),
                
                # Multiple client approach
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web', 'android'],
                        'player_skip': ['webpage'],
                        'player_params': {'hl': 'en', 'gl': 'US'},
                    }
                },
                
                # Rate limiting
                'sleep_interval': random.randint(2, 5),
                'max_sleep_interval': random.randint(5, 10),
                'retries': 5,
                'fragment_retries': 5,
                
                # Cookie handling
                'cookiesfrombrowser': None,
                
                # Additional options
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_color': True,
                
                # Format selection
                'format_sort': ['res:720', 'ext:mp4:m4a', 'hasvid', 'hasaud'],
                'format_sort_force': True,
            }
        else:
            # For other platforms, use simpler config
            ydl_opts = {
                'format': 'best[height<=720]/best',
                'outtmpl': temp_file.replace('.mp4', '.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
                'http_headers': get_rotating_headers(),
                'retries': 3,
                'fragment_retries': 3,
            }
        
        # Add cookies if requested and available
        if use_cookies and platform == 'youtube':
            cookies_file = '/opt/ytapp/youtube_cookies.txt'
            if os.path.exists(cookies_file):
                ydl_opts['cookiefile'] = cookies_file
                logger.info("Using cookies file for download")
            else:
                logger.warning("Cookies requested but file not found")
        
        # Download with retry logic
        logger.info(f"Starting {platform} download (attempt {retry_count + 1}): {url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
        # Check for downloaded file
        downloaded_file = None
        base_name = temp_file.replace('.mp4', '')
        
        # Check for various extensions
        for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv']:
            test_file = base_name + ext
            if os.path.exists(test_file) and os.path.getsize(test_file) > 1000:  # At least 1KB
                downloaded_file = test_file
                break
        
        # Handle mhtml files (webpage snapshots)
        if not downloaded_file:
            mhtml_file = base_name + '.mhtml'
            if os.path.exists(mhtml_file):
                logger.error(f"Downloaded mhtml file instead of video: {mhtml_file}")
                if retry_count < max_retries:
                    logger.info(f"Retrying download (attempt {retry_count + 2})")
                    return download_with_yt_dlp_advanced(url, use_cookies, retry_count + 1)
                else:
                    raise Exception("Download failed - got webpage snapshot instead of video")
        
        if not downloaded_file:
            if retry_count < max_retries:
                logger.info(f"Retrying download (attempt {retry_count + 2})")
                return download_with_yt_dlp_advanced(url, use_cookies, retry_count + 1)
            else:
                raise Exception("Download failed - no valid file found after all retries")
            
        logger.info(f"Download completed: {downloaded_file} (size: {os.path.getsize(downloaded_file)})")
        return downloaded_file, info.get('title', 'Unknown Title')
        
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            cleanup_temp_file(temp_file)
        
        # Retry on certain errors
        if retry_count < max_retries and any(keyword in str(e).lower() for keyword in ['precondition', 'signature', 'extraction', 'gvs', 'po_token']):
            logger.info(f"Retrying due to error: {str(e)}")
            time.sleep(random.uniform(5, 10))  # Longer delay before retry
            return download_with_yt_dlp_advanced(url, use_cookies, retry_count + 1)
        
        raise e

def download_with_alternative_method(url, use_cookies=False):
    """Alternative download method using different approach"""
    try:
        # Try using yt-dlp with different configuration
        temp_fd, temp_file = tempfile.mkstemp(suffix='.mp4', dir='/tmp')
        os.close(temp_fd)
        
        # Use command-line yt-dlp with different options
        cmd = [
            'yt-dlp',
            '--format', 'best[height<=480]',
            '--output', temp_file.replace('.mp4', '.%(ext)s'),
            '--no-warnings',
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            '--sleep-interval', '2',
            '--max-sleep-interval', '5',
            '--retries', '3',
            url
        ]
        
        if use_cookies:
            cookies_file = '/opt/ytapp/youtube_cookies.txt'
            if os.path.exists(cookies_file):
                cmd.extend(['--cookies', cookies_file])
        
        logger.info(f"Trying alternative method: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise Exception(f"Alternative method failed: {result.stderr}")
        
        # Check for downloaded file
        base_name = temp_file.replace('.mp4', '')
        for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv']:
            test_file = base_name + ext
            if os.path.exists(test_file) and os.path.getsize(test_file) > 1000:
                return test_file, "Downloaded Video"
        
        raise Exception("Alternative method: no valid file found")
        
    except Exception as e:
        logger.error(f"Alternative method error: {str(e)}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ultimate-video-processor',
        'timestamp': time.time(),
        'supported_platforms': ['youtube', 'vimeo', 'dailymotion', 'bilibili']
    })

@app.route('/download', methods=['GET'])
def download():
    """Main download endpoint with multiple fallback methods"""
    url = request.args.get('url')
    use_cookies = request.args.get('use_cookies', 'false').lower() == 'true'
    method = request.args.get('method', 'auto')  # auto, yt-dlp, alternative
    
    if not url:
        return jsonify({'error': 'URL parameter is required'}), 400
    
    # Validate URL
    platform = detect_platform(url)
    if platform == 'unknown':
        return jsonify({'error': 'Unsupported platform. Supported: YouTube, Vimeo, Dailymotion, Bilibili'}), 400
    
    # Check for concurrent downloads
    with download_lock:
        if url in active_downloads:
            return jsonify({'error': 'Download already in progress for this URL'}), 409
        active_downloads[url] = True
    
    temp_file = None
    try:
        # Try different methods based on request or auto-fallback
        if method == 'alternative':
            temp_file, title = download_with_alternative_method(url, use_cookies)
        elif method == 'yt-dlp':
            temp_file, title = download_with_yt_dlp_advanced(url, use_cookies)
        else:  # auto - try yt-dlp first, then alternative
            try:
                temp_file, title = download_with_yt_dlp_advanced(url, use_cookies)
            except Exception as e:
                logger.info(f"yt-dlp failed, trying alternative method: {str(e)}")
                temp_file, title = download_with_alternative_method(url, use_cookies)
        
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

@app.route('/platforms', methods=['GET'])
def platforms():
    """Get supported platforms"""
    return jsonify({
        'supported_platforms': {
            'youtube': {
                'domains': ['youtube.com', 'youtu.be'],
                'features': ['cookies_support', 'age_restricted', 'private_videos']
            },
            'vimeo': {
                'domains': ['vimeo.com'],
                'features': ['high_quality', 'less_restrictions']
            },
            'dailymotion': {
                'domains': ['dailymotion.com', 'dai.ly'],
                'features': ['fast_download', 'good_quality']
            },
            'bilibili': {
                'domains': ['bilibili.com'],
                'features': ['asian_content', 'different_region']
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False) 