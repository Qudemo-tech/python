#!/usr/bin/env python3
"""
Advanced Website Scraper with Dynamic Depth and Bot Detection Bypass
Handles various website types including CRM sites with anti-bot protection
"""

import os
import json
import logging
import time
import random
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime
import re

# Web scraping libraries
import requests
from bs4 import BeautifulSoup
import aiohttp
# Playwright imports (optional - only for complex sites)
try:
    from playwright.async_api import async_playwright, Browser, Page
    from playwright_stealth import stealth_async
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not available - using lightweight scraping only")

# Configure logging
logger = logging.getLogger(__name__)

class WebsiteScraper:
    """Advanced website scraper with bot detection bypass and dynamic depth"""
    
    def __init__(self):
        """Initialize the website scraper"""
        self.session = None
        self.browser = None
        self.page = None
        self.scraped_urls = set()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
        
        # CRM detection patterns
        self.crm_patterns = {
            'salesforce': ['salesforce.com', 'force.com', 'lightning.force.com'],
            'hubspot': ['hubspot.com', 'app.hubspot.com'],
            'zendesk': ['zendesk.com', 'support.zendesk.com'],
            'pipedrive': ['pipedrive.com', 'app.pipedrive.com'],
            'monday': ['monday.com', 'app.monday.com'],
            'asana': ['asana.com', 'app.asana.com']
        }
        
        logger.info("✅ Website Scraper initialized")
    
    async def analyze_site_complexity(self, url: str) -> Dict[str, Any]:
        """
        Quick analysis to determine site complexity and estimate pages
        """
        try:
            logger.info(f"🔍 Analyzing site complexity for: {url}")
            
            # Quick request to analyze the site
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    content = await response.text()
                    
                    # Detect complexity indicators
                    has_js = 'javascript' in content.lower() or '<script' in content.lower()
                    has_forms = '<form' in content.lower()
                    has_dynamic = any(keyword in content.lower() for keyword in ['ajax', 'fetch', 'xhr', 'react', 'vue', 'angular'])
                    has_spa = any(keyword in content.lower() for keyword in ['single page application', 'spa', 'client-side routing'])
                    
                    # Detect CRM
                    is_crm = self._detect_crm(url, content)
                    
                    # Estimate pages (quick scan)
                    estimated_pages = await self._estimate_page_count(url, content)
                    
                    # Determine complexity
                    if is_crm:
                        complexity = 'crm'
                    elif has_spa or (has_dynamic and has_js):
                        complexity = 'complex'
                    elif has_js or has_forms:
                        complexity = 'medium'
                    else:
                        complexity = 'simple'
                    
                    analysis = {
                        'complexity': complexity,
                        'estimated_pages': estimated_pages,
                        'has_js': has_js,
                        'has_forms': has_forms,
                        'has_dynamic': has_dynamic,
                        'has_spa': has_spa,
                        'is_crm': is_crm,
                        'crm_type': self._get_crm_type(url) if is_crm else None
                    }
                    
                    logger.info(f"📊 Site analysis: {complexity} complexity, ~{estimated_pages} pages")
                    return analysis
                    
        except Exception as e:
            logger.warning(f"⚠️ Site analysis failed for {url}: {e}")
            # Return default analysis
            return {
                'complexity': 'medium',
                'estimated_pages': 20,
                'has_js': True,
                'has_forms': True,
                'has_dynamic': False,
                'has_spa': False,
                'is_crm': False,
                'crm_type': None
            }
    
    def _detect_crm(self, url: str, content: str) -> bool:
        """Detect if the site is a CRM platform"""
        domain = urlparse(url).netloc.lower()
        content_lower = content.lower()
        
        for crm_name, patterns in self.crm_patterns.items():
            if any(pattern in domain for pattern in patterns):
                return True
            if any(pattern in content_lower for pattern in patterns):
                return True
        
        return False
    
    def _get_crm_type(self, url: str) -> Optional[str]:
        """Get the specific CRM type"""
        domain = urlparse(url).netloc.lower()
        
        for crm_name, patterns in self.crm_patterns.items():
            if any(pattern in domain for pattern in patterns):
                return crm_name
        
        return None
    
    async def _estimate_page_count(self, url: str, content: str) -> int:
        """Quick estimation of total pages"""
        try:
            # Extract internal links from the page
            soup = BeautifulSoup(content, 'html.parser')
            base_domain = urlparse(url).netloc
            base_path = urlparse(url).path
            
            internal_links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)
                
                # Only count same domain links
                if parsed.netloc == base_domain:
                    # Check if it's within the same path structure
                    if self._is_within_path(base_path, parsed.path):
                        internal_links.add(full_url)
            
            # Estimate based on link count (rough heuristic)
            link_count = len(internal_links)
            if link_count <= 10:
                return max(link_count, 5)
            elif link_count <= 50:
                return min(link_count * 2, 50)
            else:
                return min(link_count * 1.5, 100)
                
        except Exception as e:
            logger.warning(f"⚠️ Page count estimation failed: {e}")
            return 20  # Default estimate
    
    def _is_within_path(self, base_path: str, target_path: str) -> bool:
        """Check if target path is within base path - STRICT matching"""
        if not base_path or base_path == '/':
            return True
        
        # Normalize paths
        base_path = base_path.rstrip('/')
        target_path = target_path.rstrip('/')
        
        # STRICT RULE: Target path must start with base path
        # This ensures we only crawl within the specific section provided
        if target_path.startswith(base_path):
            return True
        
        # Special case: If base path is a collection, allow articles that are linked from that collection
        # but only if they're explicitly linked (not general site navigation)
        base_segments = [seg for seg in base_path.split('/') if seg]
        target_segments = [seg for seg in target_path.split('/') if seg]
        
        # For collection pages, allow articles that are part of that collection
        # e.g., /en/collections/3480900-general-account should allow /en/articles/... 
        # but only if they're linked from that specific collection
        if len(base_segments) >= 3 and base_segments[1] == 'collections':
            if len(target_segments) >= 2 and target_segments[1] == 'articles':
                # Only allow if they share the same language prefix
                if base_segments[0] == target_segments[0]:
                    return True
        
        return False
    
    def _is_document_link(self, url: str) -> bool:
        """Check if URL points to a document that should be skipped"""
        document_extensions = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.txt', '.rtf', '.odt', '.ods', '.odp', '.csv', '.zip',
            '.rar', '.7z', '.tar', '.gz', '.mp4', '.avi', '.mov',
            '.mp3', '.wav', '.flac', '.jpg', '.jpeg', '.png', '.gif',
            '.svg', '.ico', '.bmp', '.tiff', '.webp'
        ]
        
        url_lower = url.lower()
        
        # Check file extension
        for ext in document_extensions:
            if url_lower.endswith(ext):
                return True
        
        # Check for common document patterns in URL
        document_patterns = [
            '/download/', '/files/', '/documents/', '/attachments/',
            '/media/', '/uploads/', '/assets/', '/static/'
        ]
        
        for pattern in document_patterns:
            if pattern in url_lower:
                return True
        
        return False
    
    def _is_external_service_link(self, url: str) -> bool:
        """Check if URL points to external services that should be skipped"""
        external_services = [
            'github.com', 'gitlab.com', 'bitbucket.org', 'stackoverflow.com',
            'stackexchange.com', 'reddit.com', 'twitter.com', 'facebook.com',
            'linkedin.com', 'youtube.com', 'vimeo.com', 'instagram.com',
            'discord.com', 'slack.com', 'zoom.us', 'meet.google.com',
            'teams.microsoft.com', 'dropbox.com', 'drive.google.com',
            'onedrive.live.com', 'box.com', 'aws.amazon.com', 'azure.microsoft.com',
            'cloud.google.com', 'heroku.com', 'vercel.com', 'netlify.com'
        ]
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix for comparison
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain in external_services
    
    def _is_error_page(self, content: str) -> bool:
        """Check if the page content indicates an error page"""
        content_lower = content.lower()
        
        # Common error page indicators
        error_indicators = [
            '404 not found', 'page not found', 'not found',
            '403 forbidden', 'access denied', 'forbidden',
            '500 internal server error', 'server error',
            '502 bad gateway', '503 service unavailable',
            '504 gateway timeout', 'timeout',
            'error occurred', 'something went wrong',
            'page cannot be displayed', 'page unavailable',
            'maintenance mode', 'under maintenance',
            'coming soon', 'under construction'
        ]
        
        # Check for error indicators in content
        for indicator in error_indicators:
            if indicator in content_lower:
                return True
        
        # Check for very short content (likely error page)
        if len(content.strip()) < 200:
            return True
        
        # Check for common error page HTML patterns
        error_patterns = [
            '<title>error</title>',
            '<title>not found</title>',
            '<title>404</title>',
            '<title>403</title>',
            '<title>500</title>',
            'class="error"',
            'id="error"',
            'class="not-found"',
            'class="404"'
        ]
        
        for pattern in error_patterns:
            if pattern in content_lower:
                return True
        
        return False
    
    def _is_bot_detection_page(self, content: str) -> bool:
        """Check if the page content indicates a bot detection page"""
        if not content or len(content.strip()) < 100:
            return True
            
        content_lower = content.lower()
        
        # Only check for very specific bot detection indicators
        strong_bot_indicators = [
            'please verify you are human',
            'captcha', 'recaptcha', 'hcaptcha',
            'cloudflare', 'checking your browser',
            'are you human', 'verify you are human',
            'bot detection', 'automated access blocked'
        ]
        
        # Check for strong bot detection indicators
        for indicator in strong_bot_indicators:
            if indicator in content_lower:
                return True
        
        # Check for specific bot detection HTML patterns
        bot_patterns = [
            'class="cf-browser-verification"',
            'id="cf-challenge"',
            'class="challenge"',
            'data-ray="'
        ]
        
        for pattern in bot_patterns:
            if pattern in content_lower:
                return True
        
        return False
    
    def calculate_site_timeout(self, analysis: Dict[str, Any]) -> int:
        """Calculate timeout for a single website"""
        base_timeout = 300  # 5 minutes base
        
        # Adjust based on estimated pages
        page_timeout = analysis.get('estimated_pages', 20) * 10  # 10 seconds per page
        
        # Adjust based on complexity
        complexity_multipliers = {
            'simple': 1.0,
            'medium': 1.5,
            'complex': 2.0,
            'crm': 3.0
        }
        
        complexity = analysis.get('complexity', 'medium')
        multiplier = complexity_multipliers.get(complexity, 1.5)
        
        timeout = int((base_timeout + page_timeout) * multiplier)
        logger.info(f"⏱️ Calculated timeout for {complexity} site: {timeout//60} minutes")
        return timeout
    
    async def scrape_website(self, url: str, progress_callback=None) -> Dict[str, Any]:
        """
        Main method to scrape a website with dynamic depth
        """
        try:
            logger.info(f"🌐 Starting website scraping for: {url}")
            
            # Analyze site complexity
            analysis = await self.analyze_site_complexity(url)
            
            # Initialize scraping
            self.scraped_urls = set()
            scraped_pages = []
            errors = []
            
            # Choose scraping method based on complexity
            if analysis['complexity'] == 'crm':
                scraped_pages, errors = await self._scrape_crm_site(url, analysis, progress_callback)
            elif analysis['complexity'] == 'complex':
                scraped_pages, errors = await self._scrape_complex_site(url, analysis, progress_callback)
            else:
                scraped_pages, errors = await self._scrape_simple_site(url, analysis, progress_callback)
            
            # Prepare result
            result = {
                'website_id': f"website_{int(time.time())}",
                'base_url': url,
                'scraped_pages': scraped_pages,
                'total_pages': len(scraped_pages),
                'scraping_status': 'completed' if not errors else 'partial',
                'errors': errors,
                'analysis': analysis,
                'scraped_at': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Website scraping completed: {len(scraped_pages)} pages scraped")
            return result
            
        except Exception as e:
            logger.error(f"❌ Website scraping failed for {url}: {e}")
            return {
                'website_id': f"website_{int(time.time())}",
                'base_url': url,
                'scraped_pages': [],
                'total_pages': 0,
                'scraping_status': 'failed',
                'errors': [str(e)],
                'analysis': analysis if 'analysis' in locals() else {},
                'scraped_at': datetime.now().isoformat()
            }
    
    async def _scrape_simple_site(self, url: str, analysis: Dict, progress_callback=None) -> Tuple[List[Dict], List[str]]:
        """Scrape simple static sites using requests"""
        scraped_pages = []
        errors = []
        
        try:
            # Get initial page
            page_content = await self._scrape_page_requests(url)
            if page_content:
                scraped_pages.append(page_content)
                self.scraped_urls.add(url)
            
            # Find and scrape additional pages
            if page_content:
                # We need to get the raw HTML content to extract links
                # Let's fetch the page again to get the raw HTML
                import aiohttp
                headers = {
                    'User-Agent': random.choice(self.user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=15) as response:
                        if response.status == 200:
                            raw_content = await response.text()
                            additional_urls = self._extract_internal_links(url, raw_content)
                            additional_urls = [u for u in additional_urls if u not in self.scraped_urls]
                        else:
                            additional_urls = []
                
                # Limit to reasonable number of pages
                max_pages = min(analysis.get('estimated_pages', 20), 50)
                additional_urls = additional_urls[:max_pages]
                
                for i, additional_url in enumerate(additional_urls):
                    if progress_callback:
                        progress_callback({
                            'current_page': i + 2,
                            'total_pages': len(additional_urls) + 1,
                            'current_url': additional_url,
                            'percentage': (i + 2) / (len(additional_urls) + 1) * 100
                        })
                    
                    try:
                        page_content = await self._scrape_page_requests(additional_url)
                        if page_content:
                            scraped_pages.append(page_content)
                            self.scraped_urls.add(additional_url)
                        
                        # Add delay between requests
                        await asyncio.sleep(random.uniform(1, 3))
                        
                    except Exception as e:
                        errors.append(f"Failed to scrape {additional_url}: {str(e)}")
                        logger.warning(f"⚠️ Failed to scrape {additional_url}: {e}")
            
        except Exception as e:
            errors.append(f"Failed to scrape main page: {str(e)}")
            logger.error(f"❌ Simple site scraping failed: {e}")
        
        return scraped_pages, errors
    
    async def _scrape_complex_site(self, url: str, analysis: Dict, progress_callback=None) -> Tuple[List[Dict], List[str]]:
        """Scrape complex sites using Playwright with stealth (if available) or fallback to requests"""
        scraped_pages = []
        errors = []
        
        # If Playwright is not available, fallback to simple scraping
        if not PLAYWRIGHT_AVAILABLE:
            logger.info("⚠️ Playwright not available, using lightweight scraping for complex site")
            return await self._scrape_simple_site(url, analysis, progress_callback)
        
        try:
            async with async_playwright() as p:
                # Launch browser with stealth settings
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--no-first-run',
                        '--no-default-browser-check',
                        '--disable-extensions'
                    ]
                )
                
                context = await browser.new_context(
                    user_agent=random.choice(self.user_agents),
                    viewport={'width': 1920, 'height': 1080},
                    extra_http_headers={
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                    }
                )
                
                page = await context.new_page()
                
                # Apply stealth
                await stealth_async(page)
                
                # Scrape main page
                page_content = await self._scrape_page_playwright(page, url)
                if page_content:
                    scraped_pages.append(page_content)
                    self.scraped_urls.add(url)
                
                # Find and scrape additional pages
                if page_content:
                    additional_urls = self._extract_internal_links(url, page_content['content'])
                    additional_urls = [u for u in additional_urls if u not in self.scraped_urls]
                    
                    # Limit pages
                    max_pages = min(analysis.get('estimated_pages', 20), 30)
                    additional_urls = additional_urls[:max_pages]
                    
                    for i, additional_url in enumerate(additional_urls):
                        if progress_callback:
                            progress_callback({
                                'current_page': i + 2,
                                'total_pages': len(additional_urls) + 1,
                                'current_url': additional_url,
                                'percentage': (i + 2) / (len(additional_urls) + 1) * 100
                            })
                        
                        try:
                            page_content = await self._scrape_page_playwright(page, additional_url)
                            if page_content:
                                scraped_pages.append(page_content)
                                self.scraped_urls.add(additional_url)
                            
                            # Add delay
                            await asyncio.sleep(random.uniform(2, 5))
                            
                        except Exception as e:
                            errors.append(f"Failed to scrape {additional_url}: {str(e)}")
                            logger.warning(f"⚠️ Failed to scrape {additional_url}: {e}")
                
                await browser.close()
                
        except Exception as e:
            errors.append(f"Playwright scraping failed: {str(e)}")
            logger.error(f"❌ Complex site scraping failed: {e}")
        
        return scraped_pages, errors
    
    async def _scrape_crm_site(self, url: str, analysis: Dict, progress_callback=None) -> Tuple[List[Dict], List[str]]:
        """Scrape CRM sites with specialized approaches"""
        scraped_pages = []
        errors = []
        
        crm_type = analysis.get('crm_type')
        logger.info(f"🏢 Scraping CRM site: {crm_type}")
        
        try:
            # Try multiple approaches for CRM sites
            approaches = [
                self._scrape_crm_public_pages,
                self._scrape_crm_help_center,
                self._scrape_crm_documentation
            ]
            
            for approach in approaches:
                try:
                    pages, approach_errors = await approach(url, analysis, progress_callback)
                    scraped_pages.extend(pages)
                    errors.extend(approach_errors)
                    
                    if pages:  # If we got some pages, that's good enough
                        break
                        
                except Exception as e:
                    errors.append(f"CRM approach failed: {str(e)}")
                    logger.warning(f"⚠️ CRM approach failed: {e}")
            
            # If no pages scraped, try basic scraping
            if not scraped_pages:
                pages, basic_errors = await self._scrape_simple_site(url, analysis, progress_callback)
                scraped_pages.extend(pages)
                errors.extend(basic_errors)
                
        except Exception as e:
            errors.append(f"CRM scraping failed: {str(e)}")
            logger.error(f"❌ CRM site scraping failed: {e}")
        
        return scraped_pages, errors
    
    async def _scrape_crm_public_pages(self, url: str, analysis: Dict, progress_callback=None) -> Tuple[List[Dict], List[str]]:
        """Try to scrape public pages from CRM sites"""
        scraped_pages = []
        errors = []
        
        # Common public page patterns for CRM sites
        public_paths = [
            '/help',
            '/support',
            '/documentation',
            '/knowledge-base',
            '/faq',
            '/guides',
            '/tutorials',
            '/community',
            '/blog'
        ]
        
        base_url = urlparse(url)
        base_domain = f"{base_url.scheme}://{base_url.netloc}"
        
        for path in public_paths:
            try:
                public_url = base_domain + path
                page_content = await self._scrape_page_requests(public_url)
                if page_content:
                    scraped_pages.append(page_content)
                    self.scraped_urls.add(public_url)
                    
                await asyncio.sleep(random.uniform(2, 4))
                
            except Exception as e:
                errors.append(f"Failed to scrape public page {path}: {str(e)}")
        
        return scraped_pages, errors
    
    async def _scrape_crm_help_center(self, url: str, analysis: Dict, progress_callback=None) -> Tuple[List[Dict], List[str]]:
        """Try to scrape help center content"""
        scraped_pages = []
        errors = []
        
        # This would be implemented based on specific CRM help center structures
        # For now, return empty results
        return scraped_pages, errors
    
    async def _scrape_crm_documentation(self, url: str, analysis: Dict, progress_callback=None) -> Tuple[List[Dict], List[str]]:
        """Try to scrape documentation content"""
        scraped_pages = []
        errors = []
        
        # This would be implemented based on specific CRM documentation structures
        # For now, return empty results
        return scraped_pages, errors
    
    async def _scrape_page_requests(self, url: str) -> Optional[Dict]:
        """Scrape a single page using requests with robust error handling"""
        try:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Check if content is valid HTML
                        if not content or len(content.strip()) < 100:
                            logger.warning(f"⚠️ Empty or too short content for {url}")
                            return None
                        
                        # Check for common error pages
                        if self._is_error_page(content):
                            logger.warning(f"⚠️ Error page detected for {url}")
                            return None
                        
                        # Check for bot detection pages
                        if self._is_bot_detection_page(content):
                            logger.warning(f"🤖 Bot detection page for {url}")
                            return None
                        
                        return self._extract_page_content(url, content)
                    else:
                        logger.warning(f"⚠️ HTTP {response.status} for {url}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Timeout scraping {url}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"⚠️ Client error scraping {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to scrape {url} with requests: {e}")
            return None
    
    async def _scrape_page_playwright(self, page, url: str) -> Optional[Dict]:
        """Scrape a single page using Playwright"""
        try:
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for content to load
            await asyncio.sleep(random.uniform(1, 3))
            
            # Get page content
            content = await page.content()
            return self._extract_page_content(url, content)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to scrape {url} with Playwright: {e}")
            return None
    
    def _extract_page_content(self, url: str, html_content: str) -> Dict:
        """Extract clean content from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else urlparse(url).path
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|body'))
            
            if main_content:
                content_text = main_content.get_text(separator=' ', strip=True)
            else:
                content_text = soup.get_text(separator=' ', strip=True)
            
            # Clean up content
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            # Extract headings for structure
            headings = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                headings.append({
                    'level': heading.name,
                    'text': heading.get_text().strip()
                })
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'headings': headings,
                'word_count': len(content_text.split()),
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Content extraction failed for {url}: {e}")
            return {
                'url': url,
                'title': urlparse(url).path,
                'content': '',
                'headings': [],
                'word_count': 0,
                'scraped_at': datetime.now().isoformat()
            }
    
    def _extract_internal_links(self, base_url: str, html_content: str) -> List[str]:
        """Extract internal links from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            base_domain = urlparse(base_url).netloc
            base_path = urlparse(base_url).path
            
            internal_links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                
                # Only include same domain links within the same path structure
                if (parsed.netloc == base_domain and 
                    self._is_within_path(base_path, parsed.path) and
                    full_url not in self.scraped_urls):
                    internal_links.add(full_url)
            
            return list(internal_links)
            
        except Exception as e:
            logger.warning(f"⚠️ Link extraction failed: {e}")
            return []
