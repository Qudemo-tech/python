#!/usr/bin/env python3
"""
Enhanced Scraper with Anti-Bot Detection and Failure Handling
Handles website scraping failures gracefully and provides clear user feedback
"""

import asyncio
import json
import time
import os
from typing import List, Dict, Optional, Tuple
from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
import google.generativeai as genai
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedScraperWithFailureHandling:
    """Enhanced scraper that detects anti-bot protection and handles failures gracefully"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize enhanced scraper with failure handling"""
        try:
            if not gemini_api_key:
                raise ValueError("Gemini API key is required")
            
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.browser = None
            self.page = None
            self.gemini_available = True
            
            # Anti-bot detection patterns
            self.anti_bot_patterns = {
                "cloudflare": [
                    "please wait while we check your browser",
                    "checking your browser before accessing",
                    "cloudflare",
                    "ddos protection by cloudflare",
                    "ray id:",
                    "performance & security by cloudflare"
                ],
                "recaptcha": [
                    "recaptcha",
                    "verify you're human",
                    "captcha",
                    "i'm not a robot",
                    "security check"
                ],
                "rate_limiting": [
                    "too many requests",
                    "rate limit exceeded",
                    "please slow down",
                    "request limit reached",
                    "try again later"
                ],
                "ip_blocking": [
                    "access denied",
                    "ip blocked",
                    "forbidden",
                    "unauthorized access",
                    "blocked by administrator"
                ],
                "javascript_challenge": [
                    "javascript required",
                    "enable javascript to continue",
                    "dynamic content loading",
                    "please enable javascript"
                ],
                "login_required": [
                    "login required",
                    "sign in to continue",
                    "authentication required",
                    "members only",
                    "please log in"
                ],
                "maintenance": [
                    "under maintenance",
                    "temporarily unavailable",
                    "service temporarily down",
                    "maintenance mode"
                ]
            }
            
        except Exception as e:
            print(f"❌ Failed to initialize enhanced scraper: {e}")
            raise
    
    async def setup_browser(self):
        """Setup browser with anti-bot detection capabilities"""
        try:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ]
            )
            
            context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            
            self.page = await context.new_page()
            
            # Add stealth measures
            await self.page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            print("✅ Browser setup completed with anti-bot detection")
            
        except Exception as e:
            print(f"❌ Browser setup failed: {e}")
            raise
    
    async def detect_anti_bot_protection(self, url: str) -> Tuple[bool, str, str]:
        """
        Detect anti-bot protection on a website
        Returns: (is_protected, protection_type, error_message)
        """
        try:
            if not self.page:
                await self.setup_browser()
            
            print(f"🔍 Checking for anti-bot protection on: {url}")
            
            # Navigate to the page
            response = await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            if not response:
                return True, "connection_failed", "Failed to connect to the website"
            
            # Check response status
            if response.status >= 400:
                if response.status == 403:
                    return True, "ip_blocking", "Access forbidden - IP may be blocked"
                elif response.status == 429:
                    return True, "rate_limiting", "Too many requests - rate limited"
                elif response.status == 503:
                    return True, "maintenance", "Service temporarily unavailable"
                else:
                    return True, "http_error", f"HTTP error {response.status}"
            
            # Wait a bit for dynamic content to load
            await asyncio.sleep(2)
            
            # Get page content
            content = await self.page.content()
            content_lower = content.lower()
            
            # Check for anti-bot patterns
            for protection_type, patterns in self.anti_bot_patterns.items():
                for pattern in patterns:
                    if pattern in content_lower:
                        error_message = self._get_error_message(protection_type, url)
                        print(f"🛡️ Anti-bot protection detected: {protection_type}")
                        return True, protection_type, error_message
            
            # Check for empty or minimal content (possible blocking)
            if len(content.strip()) < 1000:
                return True, "minimal_content", "Website returned minimal content - possible blocking"
            
            # Check for redirect loops or suspicious behavior
            current_url = self.page.url
            if current_url != url and "blocked" in current_url.lower():
                return True, "redirect_blocking", "Website redirected to blocking page"
            
            print(f"✅ No anti-bot protection detected on: {url}")
            return False, "none", "No anti-bot protection detected"
            
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                return True, "timeout", "Website took too long to respond - possible blocking"
            elif "net::err_" in error_msg:
                return True, "connection_error", "Connection error - website may be blocking requests"
            else:
                return True, "unknown_error", f"Unknown error: {e}"
    
    def _get_error_message(self, protection_type: str, url: str) -> str:
        """Get user-friendly error message for different protection types"""
        error_messages = {
            "cloudflare": f"The website '{url}' has Cloudflare protection that prevents automated scraping. This is a security measure to protect against bots.",
            "recaptcha": f"The website '{url}' requires human verification (reCAPTCHA) that cannot be completed by automated systems.",
            "rate_limiting": f"The website '{url}' has rate limiting that blocks automated requests. Too many requests have been made.",
            "ip_blocking": f"The website '{url}' has blocked this IP address or requires authentication to access.",
            "javascript_challenge": f"The website '{url}' requires JavaScript challenges that cannot be completed by automated systems.",
            "login_required": f"The website '{url}' requires login or authentication to access the content.",
            "maintenance": f"The website '{url}' is currently under maintenance or temporarily unavailable.",
            "connection_failed": f"Failed to connect to the website '{url}'. The website may be down or unreachable.",
            "http_error": f"The website '{url}' returned an HTTP error, indicating access restrictions.",
            "minimal_content": f"The website '{url}' returned minimal content, suggesting possible blocking or restrictions.",
            "redirect_blocking": f"The website '{url}' redirected to a blocking page, indicating access restrictions.",
            "timeout": f"The website '{url}' took too long to respond, suggesting possible blocking or server issues.",
            "connection_error": f"Connection error when accessing '{url}'. The website may be blocking automated requests.",
            "unknown_error": f"An unknown error occurred when trying to access '{url}'."
        }
        
        return error_messages.get(protection_type, f"Access to '{url}' is restricted or blocked.")
    
    async def scrape_website_with_failure_handling(self, url: str) -> Dict:
        """
        Scrape website with comprehensive failure handling
        Returns: {
            'success': bool,
            'content': List[Dict] or None,
            'error_type': str or None,
            'error_message': str or None,
            'protection_detected': bool
        }
        """
        try:
            print(f"🚀 Starting enhanced scraping of: {url}")
            
            # First, check for anti-bot protection
            is_protected, protection_type, error_message = await self.detect_anti_bot_protection(url)
            
            if is_protected:
                print(f"🛡️ Anti-bot protection detected: {protection_type}")
                return {
                    'success': False,
                    'content': None,
                    'error_type': protection_type,
                    'error_message': error_message,
                    'protection_detected': True
                }
            
            # If no protection detected, proceed with normal scraping
            print(f"✅ No anti-bot protection detected, proceeding with scraping...")
            
            # Import and use the existing scraper
            from final_gemini_scraper import FinalGeminiScraper
            
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            scraper = FinalGeminiScraper(gemini_api_key=gemini_api_key)
            
            # Scrape the website
            content = await scraper.scrape_website_comprehensive(url)
            
            if content and len(content) > 0:
                print(f"✅ Successfully scraped {len(content)} content pieces from: {url}")
                return {
                    'success': True,
                    'content': content,
                    'error_type': None,
                    'error_message': None,
                    'protection_detected': False
                }
            else:
                print(f"⚠️ No content extracted from: {url}")
                return {
                    'success': False,
                    'content': None,
                    'error_type': 'no_content',
                    'error_message': f"No content could be extracted from '{url}'. The website may have restrictions or the content may not be accessible.",
                    'protection_detected': False
                }
                
        except Exception as e:
            print(f"❌ Scraping error for {url}: {e}")
            return {
                'success': False,
                'content': None,
                'error_type': 'scraping_error',
                'error_message': f"An error occurred while scraping '{url}': {str(e)}",
                'protection_detected': False
            }
        finally:
            # Clean up browser
            if self.browser:
                await self.browser.close()
                self.browser = None
                self.page = None
    
    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
                self.page = None
                print("🧹 Browser cleanup completed")
        except Exception as e:
            print(f"⚠️ Browser cleanup error: {e}")


# Global instance for singleton pattern
_enhanced_scraper_instance = None

def initialize_enhanced_scraper():
    """Initialize the enhanced scraper with failure handling"""
    global _enhanced_scraper_instance
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            print("❌ GEMINI_API_KEY not found in environment variables")
            return False
        
        _enhanced_scraper_instance = EnhancedScraperWithFailureHandling(gemini_api_key)
        print("✅ Enhanced scraper with failure handling initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize enhanced scraper: {e}")
        return False

def get_enhanced_scraper():
    """Get the enhanced scraper instance"""
    global _enhanced_scraper_instance
    if _enhanced_scraper_instance is None:
        print("⚠️ Enhanced scraper not initialized, initializing now...")
        initialize_enhanced_scraper()
    return _enhanced_scraper_instance
