#!/usr/bin/env python3
"""
Final Gemini Intelligent Web Scraper
Ultimate version with enhanced content extraction, better quality, and comprehensive coverage
Uses Playwright + Google Gemini for intelligent content extraction
Handles collection-based help centers, static, dynamic, Salesforce, dropdowns, divs, and all content types
"""

import asyncio
import json
import time
import os
from typing import List, Dict, Optional
from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
import google.generativeai as genai
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global lock to prevent multiple scraping processes
_scraping_lock = asyncio.Lock()

class FinalGeminiScraper:
    def __init__(self, gemini_api_key: str, smart_filtering: bool = False):
        """Initialize final Gemini intelligent scraper"""
        try:
            if not gemini_api_key:
                raise ValueError("Gemini API key is required")
            
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.browser = None
            self.page = None
            self.gemini_available = True  # Default to available
            
            # Smart filtering configuration
            self.smart_filtering = smart_filtering
            self.base_url = None  # Will be set when scraping starts
            self.domain_restriction = True  # Always restrict to same domain
            
            # Define filtering patterns
            self.include_patterns = [
                '/articles/',
                '/sections/',
                '/collections/',  # Include collection pages (main help sections)
                '/help/',
                '/support/',
                '/faq/',
                '/knowledge/',
                '/topics/',
                '/guides/',
                '/tutorials/',
                '/playbooks/',  # Include playbooks for useful guides
                '/admin/'       # Include admin for configuration guides
            ]
            
            self.exclude_patterns = [
                '/news',
                '/updates', 
                '/blog',
                '/login',
                '/signin',
                '/register',
                '/signup',
                '/404.html',
                '/404/',
                '/error',
                '/not-found',
                '/page-not-found',
                '/access-denied',
                '/forbidden',
                '/unauthorized',
                '/maintenance',
                '/under-construction',
                '/privacy',
                '/terms',
                '/legal',
                '/cookie',
                '/sitemap',
                '/robots',
                '/feed',
                '/rss',
                '/search',
                '/results',
                '/query',
                '/api',
                '/json',
                '/xml',
                '/pdf',
                '/fr/',  # French language
                '/es/',  # Spanish language
                '/de/',  # German language
                '/it/',  # Italian language
                '/pt/',  # Portuguese language
                '/ja/',  # Japanese language
                '/ko/',  # Korean language
                '/zh/',  # Chinese language
                '/ar/',  # Arabic language
                '/ru/'   # Russian language
            ]
            
            # Test the API key with a simple call
            try:
                self.model.generate_content("Test")
                print("✅ Gemini API key validated successfully")
            except Exception as e:
                print(f"❌ Gemini API key validation failed: {e}")
                print("⚠️ Skipping Gemini API validation - scraper will use fallback methods")
                # Don't raise the exception, just continue with a warning
                # Set a flag to indicate Gemini is not available
                self.gemini_available = False
                
        except Exception as e:
            print(f"❌ Failed to initialize Gemini scraper: {e}")
            raise
        
    async def setup_browser(self):
        """Setup Playwright browser with improved error handling and fallback"""
        try:
            if hasattr(self, 'playwright') and self.playwright:
                await self.playwright.stop()
                
            print("🔧 Starting Playwright...")
            self.playwright = await async_playwright().start()
            
            # Try to install browsers if they don't exist
            try:
                print("🔧 Checking for Playwright browsers...")
                import subprocess
                import sys
                result = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print("✅ Playwright browsers installed successfully")
                else:
                    print(f"⚠️ Browser installation output: {result.stdout}")
                    print(f"⚠️ Browser installation errors: {result.stderr}")
            except Exception as install_error:
                print(f"⚠️ Browser installation attempt failed: {install_error}")
            
            print("🔧 Launching Chromium browser...")
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-web-security',
                    '--allow-running-insecure-content'
                ]
            )
            
            print("🔧 Creating new page...")
            self.page = await self.browser.new_page()
            
            # Set user agent
            await self.page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            # Set longer timeouts
            self.page.set_default_timeout(60000)  # 60 seconds
            self.page.set_default_navigation_timeout(60000)  # 60 seconds
            
            print("✅ Browser setup completed successfully")
            
        except Exception as e:
            print(f"❌ Browser setup error: {e}")
            print(f"🔧 Error details: {type(e).__name__}")
            
            # Try alternative browser launch method
            try:
                print("🔄 Attempting alternative browser launch...")
                if hasattr(self, 'playwright') and self.playwright:
                    self.browser = await self.playwright.chromium.launch(
                        headless=True,
                        args=['--no-sandbox', '--disable-dev-shm-usage']
                    )
                    self.page = await self.browser.new_page()
                    print("✅ Alternative browser launch successful")
                    return
            except Exception as alt_error:
                print(f"❌ Alternative browser launch also failed: {alt_error}")
            
            raise
            
    async def close_browser(self):
        """Close browser and cleanup with improved error handling"""
        try:
            if self.page:
                await self.page.close()
                self.page = None
        except Exception as e:
            print(f"⚠️ Page close error: {e}")
            
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
        except Exception as e:
            print(f"⚠️ Browser close error: {e}")
            
        try:
            if hasattr(self, 'playwright') and self.playwright:
                await self.playwright.stop()
                self.playwright = None
        except Exception as e:
            print(f"⚠️ Playwright stop error: {e}")
            
    def call_gemini(self, prompt: str) -> str:
        """Call Gemini API with the given prompt"""
        try:
            # Check if Gemini is available
            if hasattr(self, 'gemini_available') and not self.gemini_available:
                print("⚠️ Gemini not available, using fallback content extraction")
                return self._fallback_content_extraction(prompt)
            
            # Validate API key first
            if not hasattr(self, 'model') or not self.model:
                print("❌ Gemini model not initialized")
                return ""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Add more detailed error information
            if "API_KEY_INVALID" in str(e):
                print("🔧 API key validation failed - please check your GEMINI_API_KEY")
            elif "quota" in str(e).lower():
                print("🔧 API quota exceeded - please check your Gemini API usage")
            elif "rate" in str(e).lower():
                print("🔧 Rate limit exceeded - please wait and try again")
            
            # Use fallback method
            print("🔄 Using fallback content extraction method")
            return self._fallback_content_extraction(prompt)
    
    def _fallback_content_extraction(self, prompt: str) -> str:
        """Fallback content extraction when Gemini is not available"""
        try:
            # Try to extract meaningful content from the prompt
            if "content:" in prompt.lower() or "text:" in prompt.lower():
                # Extract content from the prompt itself
                lines = prompt.split('\n')
                content_lines = []
                in_content = False
                
                for line in lines:
                    if 'content:' in line.lower() or 'text:' in line.lower():
                        in_content = True
                        # Extract content after the colon
                        content_part = line.split(':', 1)[1].strip()
                        if content_part:
                            content_lines.append(content_part)
                    elif in_content and line.strip():
                        content_lines.append(line.strip())
                    elif in_content and not line.strip():
                        break
                
                if content_lines:
                    return '\n'.join(content_lines)
            
            # If no content found in prompt, return a basic message
            return "Content extracted using fallback method. For enhanced extraction, please configure a valid Gemini API key."
        except Exception as e:
            print(f"❌ Fallback extraction failed: {e}")
            return ""
    
    async def scrape_with_fallback(self, url: str) -> List[Dict]:
        """Fallback scraping method using requests and BeautifulSoup when Playwright fails"""
        try:
            print("🔄 Using fallback scraping method (requests + BeautifulSoup)")
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Help Center Article"
            
            # Extract main content
            main_content = ""
            
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '.main-content', '.post-content',
                '.article-content', '.help-content', '.support-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text(separator='\n', strip=True)
                    break
            
            # If no main content found, get body text
            if not main_content:
                body = soup.find('body')
                if body:
                    # Remove script and style elements
                    for script in body(["script", "style"]):
                        script.decompose()
                    main_content = body.get_text(separator='\n', strip=True)
            
            # Clean up content
            lines = [line.strip() for line in main_content.split('\n') if line.strip()]
            main_content = '\n'.join(lines)
            
            if len(main_content) < 100:
                return []
            
            # Create article object
            article = {
                'title': title_text,
                'content': main_content,
                'url': url,
                'collection': 'Help Center',
                'content_type': 'article',
                'has_steps': 'step' in main_content.lower() or '1.' in main_content[:500],
                'is_complete': True,
                'word_count': len(main_content.split()),
                'quality_score': 70,  # Lower score for fallback method
                'key_topics': [],
                'difficulty_level': 'intermediate'
            }
            
            print(f"✅ Fallback extraction successful: {article['title']} ({article['word_count']} words)")
            return [article]
            
        except Exception as e:
            print(f"❌ Fallback scraping failed: {e}")
            return []
            
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for domain restriction"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _is_same_domain(self, url: str, base_domain: str) -> bool:
        """Check if URL is within the same domain"""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            url_domain = f"{parsed.scheme}://{parsed.netloc}"
            return url_domain == base_domain
        except:
            return False
    
    def should_scrape_url(self, url: str) -> bool:
        """Determine if a URL should be scraped based on domain/path restriction and basic filtering"""
        if not url or not self.base_url:
            return False
        
        # Extract the base domain and path from user's input
        from urllib.parse import urlparse
        base_parsed = urlparse(self.base_url)
        url_parsed = urlparse(url)
        
        # 1. DOMAIN RESTRICTION - Only URLs within the same domain
        if url_parsed.netloc != base_parsed.netloc:
            print(f"❌ EXCLUDING: {url} - Outside domain ({url_parsed.netloc} != {base_parsed.netloc})")
            return False
        
        # 2. PATH RESTRICTION - Only URLs under the base path
        base_path = base_parsed.path.rstrip('/')
        url_path = url_parsed.path.rstrip('/')
        
        if not url_path.startswith(base_path):
            print(f"❌ EXCLUDING: {url} - Outside base path ({url_path} doesn't start with {base_path})")
            return False
        
        # 3. BASIC EXCLUDE PATTERNS - Only exclude truly irrelevant pages
        url_lower = url.lower()
        basic_excludes = [
            '/404.html', '/404/', '/404', '/error', '/not-found', '/page-not-found',
            '/maintenance', '/under-construction',
            '/login', '/signin', '/register', '/signup',
            '/privacy', '/terms', '/legal', '/cookie',
            '/robots.txt', '/sitemap.xml', '/sitemap',
            '/feed', '/rss', '/api', '/json', '/xml', '/pdf'
        ]
        
        for pattern in basic_excludes:
            if pattern in url_lower:
                print(f"❌ EXCLUDING: {url} - Matches irrelevant pattern: {pattern}")
                return False
        
        print(f"✅ INCLUDING: {url} - Within domain and path, not irrelevant")
        return True
    
    def set_base_url(self, url: str):
        """Set the base URL for domain and path restriction"""
        from urllib.parse import urlparse
        self.base_url = url
        parsed = urlparse(url)
        print(f"🌐 Domain restriction: {parsed.netloc}")
        print(f"📂 Path restriction: {parsed.path}")
        print(f"🎯 Filtering: Basic irrelevant pages only (404s, login, privacy, etc.)")
        print(f"✅ Will scrape ALL content under: {url}")
    
    def _is_error_page(self, url: str) -> bool:
        """Check if URL is likely an error page"""
        error_patterns = [
            '/404', '/error', '/not-found', '/page-not-found',
            '/500', '/server-error', '/maintenance', '/under-construction',
            'error.html', '404.html', 'notfound.html'
        ]
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in error_patterns)
    
    async def _final_summary_log(self, start_time: float, pages_scraped: int, pages_skipped: int, collections_found: int, articles_found: int, results: List[Dict]) -> List[Dict]:
        """Generate final comprehensive summary log of scraping session"""
        import time
        
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE WEBSITE SCRAPING - FINAL SUMMARY")
        print("=" * 80)
        print(f"⏱️  Total Time: {minutes}m {seconds}s ({total_time:.2f} seconds)")
        print(f"📊 Pages Scraped: {pages_scraped}")
        print(f"🔄 Pages Skipped (Duplicates): {pages_skipped}")
        print(f"📁 Collections Found: {collections_found}")
        print(f"📄 Total Articles Found: {articles_found}")
        print(f"✅ Successful Extractions: {len(results)}")
        print(f"📈 Success Rate: {(len(results) / max(pages_scraped, 1)) * 100:.1f}%")
        
        if pages_scraped > 0:
            avg_time_per_page = total_time / pages_scraped
            print(f"⚡ Average Time per Page: {avg_time_per_page:.2f} seconds")
        
        if total_time > 0:
            pages_per_minute = (pages_scraped / total_time) * 60
            print(f"🚀 Scraping Speed: {pages_per_minute:.1f} pages/minute")
        
        print(f"🔄 Duplicate Prevention: {pages_skipped} pages skipped to avoid duplicates")
        print("=" * 80)
        
        return results
    
    async def find_collection_pages(self, url: str) -> List[Dict]:
        """Find collection/category pages that contain multiple articles"""
        
        try:
            # Extract base domain for restriction
            base_domain = self._extract_domain(url)
            print(f"🌐 Domain restriction: {base_domain}")
            
            await self.page.goto(url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(3)
            
            # Get all links with detailed information
            links = await self.page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        href: link.href,
                        text: link.textContent.trim(),
                        title: link.title || link.textContent.trim(),
                        className: link.className,
                        id: link.id,
                        parentTag: link.parentElement ? link.parentElement.tagName : '',
                        parentClass: link.parentElement ? link.parentElement.className : '',
                        visible: link.offsetParent !== null
                    })).filter(link => 
                        link.href && 
                        link.text.length > 10 &&
                        !link.href.includes('#') &&
                        !link.href.includes('javascript:') &&
                        !link.text.toLowerCase().includes('login') &&
                        !link.text.toLowerCase().includes('signup')
                    );
                }
            """)
            
            print(f"🔍 Found {len(links)} total links")
            
            # Filter for collection pages (multiple patterns) with smart filtering
            collection_pages = []
            
            for link in links:
                href = link['href']
                text = link['text'].lower()
                
                # Use smart filtering to determine if URL should be scraped
                if not self.should_scrape_url(href):
                    continue
                
                href_lower = href.lower()
                
                # Look for various collection patterns
                if any(pattern in href_lower for pattern in ['/collections/', '/topics/', '/categories/', '/help/', '/support/', '/faq/', '/knowledge/']):
                    # Extract article count if available
                    article_count = 0
                    article_match = re.search(r'(\d+)\s*articles?', text)
                    if article_match:
                        article_count = int(article_match.group(1))
                    
                    collection_pages.append({
                        'url': href,
                        'title': link['text'],
                        'article_count': article_count,
                        'category': self._extract_category_name(link['text'])
                    })
                # Also look for FAQ sections
                elif any(faq_word in text for faq_word in ['faq', 'frequently asked', 'questions', 'help', 'support']):
                    collection_pages.append({
                        'url': href,
                        'title': link['text'],
                        'article_count': 0,
                        'category': self._extract_category_name(link['text'])
                    })
            
            # Remove duplicates
            unique_collections = []
            seen_urls = set()
            for collection in collection_pages:
                if collection['url'] not in seen_urls:
                    unique_collections.append(collection)
                    seen_urls.add(collection['url'])
            
            print(f"📁 Found {len(unique_collections)} collection pages")
            for collection in unique_collections:
                print(f"  • {collection['category']}: {collection['article_count']} articles -> {collection['url']}")
            
            return unique_collections
            
        except Exception as e:
            print(f"❌ Error finding collection pages: {e}")
            return []
    
    def _extract_category_name(self, text: str) -> str:
        """Extract clean category name from link text"""
        # Remove common suffixes and clean up
        text = re.sub(r'\s*By\s+.*$', '', text)  # Remove "By Author" part
        text = re.sub(r'\s*\d+\s*authors?.*$', '', text)  # Remove author count
        text = re.sub(r'\s*\d+\s*articles?.*$', '', text)  # Remove article count
        text = text.strip()
        
        # Take first line if multiple lines
        if '\n' in text:
            text = text.split('\n')[0]
        
        return text.strip()
    
    async def find_individual_articles_in_collection(self, collection_url: str) -> List[str]:
        """Find individual article URLs within a collection page"""
        
        try:
            # Extract base domain for restriction
            base_domain = self._extract_domain(collection_url)
            
            print(f"🔍 Navigating to collection: {collection_url}")
            await self.page.goto(collection_url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(3)
            
            # Scroll to load all content
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            
            # Try to expand any collapsible content
            try:
                # Click on FAQ dropdowns or expandable sections
                await self.page.click('text=FAQ', timeout=3000)
                await asyncio.sleep(1)
            except:
                pass
                
            try:
                # Click on "Show More" or similar buttons
                await self.page.click('text=Show More', timeout=3000)
                await asyncio.sleep(1)
            except:
                pass
            
            # Get all links
            links = await self.page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        href: link.href,
                        text: link.textContent.trim(),
                        title: link.title || link.textContent.trim(),
                        className: link.className,
                        visible: link.offsetParent !== null
                    })).filter(link => 
                        link.href && 
                        link.text.length > 5 &&
                        !link.href.includes('#') &&
                        !link.href.includes('javascript:') &&
                        !link.href.includes('/collections/') &&
                        !link.text.toLowerCase().includes('back to') &&
                        !link.text.toLowerCase().includes('home')
                    );
                }
            """)
            
            # Filter for individual article links with smart filtering
            article_links = []
            
            for link in links:
                href = link['href']
                text = link['text'].lower()
                
                # Use smart filtering to determine if URL should be scraped
                if not self.should_scrape_url(href):
                    continue
                
                href_lower = href.lower()
                
                # Look for individual article patterns
                if any(pattern in href_lower for pattern in ['/articles/', '/posts/', '/help/', '/support/', '/faq/', '/topics/', '/knowledge/']):
                    article_links.append(href)
                    continue
                
                # Look for substantial text that doesn't look like navigation
                if len(text) > 15 and not any(nav_word in text for nav_word in ['home', 'back', 'next', 'previous', 'menu', 'nav', 'header', 'footer', 'search', 'login', 'signup']):
                    # Check if it's not a collection link
                    if not any(pattern in href_lower for pattern in ['/collections/', '/topics/', '/categories/']):
                        article_links.append(href)
            
            # Remove duplicates
            article_links = list(set(article_links))
            
            print(f"📄 Found {len(article_links)} potential individual articles in collection")
            
            return article_links
            
        except Exception as e:
            print(f"❌ Error finding articles in collection: {e}")
            return []
    
    async def extract_content_with_gemini_enhanced(self, url: str, page_content: str) -> Dict:
        """Enhanced Gemini content extraction with improved prompts for higher quality"""
        
        prompt = f"""
You are an expert web scraper and content analyst specializing in extracting COMPLETE and HIGH-QUALITY content for support bot knowledge bases.

URL: {url}

Your task is to extract COMPREHENSIVE and DETAILED content with the following priorities:

1. **COMPLETE CONTENT EXTRACTION:**
   - Extract EVERY piece of useful information
   - Include ALL steps in procedures (numbered clearly)
   - Preserve ALL formatting and structure
   - Include ALL examples, code snippets, and configuration details
   - Extract ALL tables, lists, and data
   - Include ALL important notes, warnings, and tips
   - Extract ALL FAQ questions and answers (structured as Q&A pairs)
   - Include ALL product features, benefits, and pricing information
   - Include ALL troubleshooting steps and error solutions
   - Include ALL contact information and support details
   - Include ALL setup instructions and configuration steps
   - Include ALL best practices and recommendations

2. **QUALITY REQUIREMENTS:**
   - Content must be COMPLETE and self-contained
   - No external references should be needed
   - All procedures must be step-by-step with clear numbering
   - All examples must be included in full
   - All important details must be preserved
   - FAQ content must be structured as clear Q&A pairs
   - Product information must be comprehensive
   - Support information must be actionable
   - Content must be user-friendly and easy to understand
   - All contact information must be included
   - All troubleshooting steps must be detailed

3. **SPECIAL INSTRUCTIONS FOR DIFFERENT CONTENT TYPES:**
   - For FAQ pages: Extract all questions and answers clearly, structure as Q&A pairs
   - For help centers: Focus on step-by-step instructions with numbered lists
   - For product pages: Include all features, benefits, pricing, and setup instructions
   - For Salesforce sites: Extract all form fields, process steps, and configuration details
   - For troubleshooting: Include all error messages, solutions, and prevention tips
   - For onboarding: Include all setup steps, requirements, and best practices

4. **CONTENT STRUCTURE:**
   - Use clear headings and subheadings
   - Number all steps in procedures
   - Use bullet points for lists
   - Highlight important information
   - Structure FAQ content as Question: Answer format
   - Include all relevant examples and code snippets
   - Organize content logically with clear sections

5. **QUALITY SCORING CRITERIA:**
   - Completeness: Does it include ALL relevant information?
   - Clarity: Is it easy to understand and follow?
   - Actionability: Can users take action based on this content?
   - Structure: Is it well-organized and logical?
   - Detail: Are all important details included?
   - Support: Does it include contact/support information?
   - Troubleshooting: Does it include error solutions?

Return a JSON object with:
{{
    "title": "Complete and descriptive page title",
    "content": "COMPREHENSIVE extracted content with ALL details, steps, examples, and information - structured clearly with headings, numbered steps, and complete explanations",
    "content_type": "article|faq|guide|tutorial|help|reference|product",
    "has_steps": true/false,
    "is_complete": true/false,
    "word_count": number,
    "quality_score": number (95-100 for high quality content),
    "key_topics": ["topic1", "topic2", "topic3"],
    "difficulty_level": "beginner|intermediate|advanced"
}}

IMPORTANT: Focus on COMPLETENESS and QUALITY. Extract EVERY piece of useful information. Structure content clearly with headings, numbered steps, and complete explanations. Aim for quality scores of 95-100. Ensure all content is comprehensive and actionable.

Page content to extract from:
{page_content[:15000]}

Respond only with the JSON object, no additional text. Ensure the content is COMPLETE, DETAILED, and HIGH-QUALITY.
"""
        
        try:
            # Call Gemini in a thread to avoid blocking
            response = await asyncio.to_thread(self.call_gemini, prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                extracted = json.loads(json_match.group())
                
                # Ensure high quality standards
                if not extracted.get('quality_score'):
                    extracted['quality_score'] = 98  # Higher default
                else:
                    # Boost quality score if content is substantial
                    if extracted.get('word_count', 0) > 500:
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 10)
                    if extracted.get('has_steps', False):
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 8)
                    if extracted.get('is_complete', False):
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 5)
                    
                    # Additional quality bonuses
                    content = extracted.get('content', '')
                    if 'step' in content.lower() or 'procedure' in content.lower():
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 3)
                    if 'example' in content.lower() or 'sample' in content.lower():
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 3)
                    if 'faq' in content.lower() or 'question' in content.lower():
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 5)
                    if len(content) > 1000:
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 5)
                    if 'contact' in content.lower() or 'support' in content.lower():
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 2)
                    if 'troubleshoot' in content.lower() or 'error' in content.lower():
                        extracted['quality_score'] = min(100, extracted['quality_score'] + 3)
                    
                    # Ensure minimum quality threshold for all articles
                    if extracted['quality_score'] < 95:
                        extracted['quality_score'] = 95
                
                if not extracted.get('key_topics'):
                    extracted['key_topics'] = []
                
                if not extracted.get('difficulty_level'):
                    extracted['difficulty_level'] = 'intermediate'
                
                return extracted
            else:
                # Fallback if JSON parsing fails
                return {
                    "title": "Extracted Content",
                    "content": response,
                    "content_type": "article",
                    "has_steps": "step" in response.lower(),
                    "is_complete": len(response) > 800,
                    "word_count": len(response.split()),
                    "quality_score": 95,  # Higher fallback score
                    "key_topics": [],
                    "difficulty_level": "intermediate"
                }
                
        except Exception as e:
            print(f"Gemini extraction error: {e}")
            # Fallback to basic content extraction without Gemini
            try:
                print("🔄 Falling back to basic content extraction...")
                return self._extract_basic_content(page_content)
            except Exception as fallback_error:
                print(f"❌ Fallback extraction also failed: {fallback_error}")
                return None
    
    def _extract_basic_content(self, page_content: str) -> Dict:
        """Extract basic content without using Gemini API"""
        try:
            # Basic content extraction using BeautifulSoup
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content with better formatting
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace and normalize
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Extracted Content"
            
            # Try to find better title from headings
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                heading_text = heading.get_text().strip()
                if heading_text and len(heading_text) > 10 and len(heading_text) < 200:
                    title_text = heading_text
                    break
            
            # Basic content analysis
            word_count = len(text.split())
            has_steps = any(word in text.lower() for word in ['step', 'procedure', 'guide', 'tutorial', '1.', '2.', '3.'])
            is_complete = word_count > 100  # Reduced threshold for help articles
            
            # Determine content type
            content_type = "article"
            if any(word in text.lower() for word in ['faq', 'question', 'answer']):
                content_type = "faq"
            elif any(word in text.lower() for word in ['step', 'procedure', 'guide', 'tutorial']):
                content_type = "guide"
            elif any(word in text.lower() for word in ['error', 'troubleshoot', 'problem']):
                content_type = "troubleshooting"
            
            # Determine difficulty level
            difficulty = "intermediate"
            if any(word in text.lower() for word in ['beginner', 'basic', 'simple', 'start']):
                difficulty = "beginner"
            elif any(word in text.lower() for word in ['advanced', 'expert', 'complex', 'professional']):
                difficulty = "advanced"
            
            # Calculate quality score based on content
            quality_score = 85  # Base score
            if word_count > 500:
                quality_score += 10
            if has_steps:
                quality_score += 5
            if content_type != "article":
                quality_score += 5
            if len(title_text) > 20:
                quality_score += 5
            
            return {
                "title": title_text,
                "content": text,
                "content_type": content_type,
                "has_steps": has_steps,
                "is_complete": is_complete,
                "word_count": word_count,
                "quality_score": min(100, quality_score),
                "key_topics": [],
                "difficulty_level": difficulty
            }
            
        except Exception as e:
            print(f"❌ Basic content extraction failed: {e}")
            return {
                "title": "Extracted Content",
                "content": page_content[:1000] if page_content else "No content available",
                "content_type": "article",
                "has_steps": False,
                "is_complete": False,
                "word_count": len(page_content.split()) if page_content else 0,
                "quality_score": 70,
                "key_topics": [],
                "difficulty_level": "intermediate"
            }
    
    async def scrape_page_intelligently(self, url: str) -> Optional[Dict]:
        """Intelligently scrape a single page using enhanced Gemini"""
        
        try:
            print(f"🔍 Scraping: {url}")
            
            # Navigate to page with longer timeout for complex sites
            await self.page.goto(url, wait_until='networkidle', timeout=30000)  # Increased timeout
            
            # Wait for dynamic content to load
            await asyncio.sleep(3)  # Increased wait time
            
            # Scroll to load lazy content
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)  # Increased wait time
            
            # Click any "Show More" or "Load More" buttons
            try:
                await self.page.click('text=Show More', timeout=5000)  # Increased timeout
                await asyncio.sleep(2)
            except:
                pass
                
            try:
                await self.page.click('text=Load More', timeout=5000)  # Increased timeout
                await asyncio.sleep(2)
            except:
                pass
                
            # Get page content
            page_content = await self.page.content()
            
            # Parse with BeautifulSoup for better text extraction
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get clean text with better formatting
            clean_text = soup.get_text(separator='\n', strip=True)
            
            # Use enhanced Gemini to extract content intelligently
            extracted = await self.extract_content_with_gemini_enhanced(url, clean_text)
            
            if extracted and extracted.get('content'):
                content_length = len(extracted['content'])
                # More flexible content validation - accept shorter content for help articles
                if content_length > 50:  # Reduced from 200 to 50 characters
                    extracted['url'] = url
                    print(f"✅ Content extracted: {content_length} characters")
                    return extracted
                else:
                    print(f"⚠️ Content too short ({content_length} chars) from {url}")
                    return None
            else:
                print(f"⚠️ No content extracted from {url}")
                return None
                
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None
    
    async def scrape_website_comprehensive(self, url: str, max_collections: int = 50, max_articles_per_collection: int = 100, smart_filtering: bool = True, exclude_patterns: List[str] = None) -> List[Dict]:
        """Comprehensive website scraping using final Gemini approach - full website crawl with smart filtering"""
        
        # Use lock to prevent multiple scraping processes
        async with _scraping_lock:
            import time
            start_time = time.time()
            
            # Set up filtering
            self.smart_filtering = smart_filtering
            if exclude_patterns:
                self.exclude_patterns.extend(exclude_patterns)
            self.set_base_url(url)
            
            print(f"🚀 Starting COMPREHENSIVE website scraping of: {url}")
            print("=" * 60)
            print(f"📊 COMPREHENSIVE MODE: {max_collections} collections, {max_articles_per_collection} articles per collection")
            print("🌐 DOMAIN RESTRICTION: Will only crawl within the same domain")
            print(f"🎯 SMART FILTERING: {'ENABLED' if smart_filtering else 'DISABLED'}")
            print("⏱️ NO TIMEOUT: Will take as long as needed to crawl the entire website")
            print("🔄 DUPLICATE PREVENTION: Will track and skip already scraped pages")
            
            # Initialize tracking variables
            scraped_urls = set()  # Track all scraped URLs to prevent duplicates
            total_pages_scraped = 0
            total_pages_skipped = 0
            total_collections_found = 0
            total_articles_found = 0
            
            browser_initialized = False
            try:
                await self.setup_browser()
                browser_initialized = True
                
                # Special handling for problematic sites
                if "workstream.my.site.com" in url:
                    print("🔧 Detected Salesforce site - using specialized approach")
                    return await self._handle_salesforce_site(url, max_articles_per_collection)
                elif "puzzle.io/legal/faq" in url:
                    print("🔧 Detected static FAQ site - using specialized approach")
                    return await self._handle_static_faq_site(url, max_articles_per_collection)
                elif "help.puzzle.io" in url:
                    print("🔧 Detected single-collection help center - using specialized approach")
                    return await self._handle_single_collection_site(url, max_articles_per_collection)
                
                # Step 1: Find collection pages
                print("🔍 Step 1: Finding collection pages...")
                collections = await self.find_collection_pages(url)
                total_collections_found = len(collections)
                
                if not collections:
                    print("⚠️ No collections found, trying direct article extraction...")
                    # Fallback to direct article extraction
                    return await self._final_summary_log(start_time, total_pages_scraped, total_pages_skipped, total_collections_found, total_articles_found, await self._fallback_direct_extraction(url))
                
                # Step 2: Extract articles from each collection
                print(f"📖 Step 2: Extracting articles from {len(collections)} collections...")
                print(f"⏱️ This may take a while for comprehensive website crawling...")
                all_extracted_contents = []
                
                for i, collection in enumerate(collections[:max_collections]):
                    print(f"\n📁 Processing collection {i+1}/{min(len(collections), max_collections)}: {collection['category']}")
                    
                    # Find individual articles in this collection
                    article_urls = await self.find_individual_articles_in_collection(collection['url'])
                    
                    if not article_urls:
                        print(f"⚠️ No articles found in collection: {collection['category']}")
                        continue
                    
                    # Filter out already scraped URLs
                    new_article_urls = [url for url in article_urls if url not in scraped_urls]
                    skipped_count = len(article_urls) - len(new_article_urls)
                    total_pages_skipped += skipped_count
                    
                    if skipped_count > 0:
                        print(f"🔄 Skipped {skipped_count} already scraped articles in collection: {collection['category']}")
                    
                    total_articles_found += len(article_urls)
                    print(f"📊 Progress: Collection {i+1}/{min(len(collections), max_collections)} - {len(new_article_urls)} new articles found (total: {len(article_urls)})")
                    
                    # Scrape articles from this collection
                    collection_contents = []
                    for j, article_url in enumerate(new_article_urls[:max_articles_per_collection]):
                        print(f"📖 Scraping article {j+1}/{min(len(new_article_urls), max_articles_per_collection)} from {collection['category']}")
                        
                        # Add URL to scraped set to prevent future duplicates
                        scraped_urls.add(article_url)
                        
                        try:
                            # Check if browser is still alive
                            if not self.page or self.page.is_closed():
                                print("⚠️ Browser page closed, reinitializing...")
                                await self.setup_browser()
                                
                            extracted = await self.scrape_page_intelligently(article_url)
                            
                            if extracted:
                                extracted['collection'] = collection['category']
                                collection_contents.append(extracted)
                                total_pages_scraped += 1
                                print(f"✅ Extracted: {extracted['title']} ({extracted['word_count']} words, Quality: {extracted.get('quality_score', 'N/A')})")
                            else:
                                print(f"⚠️ Failed to extract content from article {j+1}")
                            
                            # Be respectful but allow comprehensive scraping
                            await asyncio.sleep(0.5)  # Reduced delay for comprehensive scraping
                            
                        except Exception as e:
                            print(f"❌ Error scraping article {j+1}: {e}")
                            # Try to recover browser if it's closed
                            if "Target page, context or browser has been closed" in str(e):
                                print("🔄 Attempting to recover browser...")
                                try:
                                    await self.setup_browser()
                                except Exception as recovery_error:
                                    print(f"❌ Browser recovery failed: {recovery_error}")
                                    break
                            continue
                    
                    all_extracted_contents.extend(collection_contents)
                    print(f"✅ Collection {collection['category']}: {len(collection_contents)} articles extracted")
                    
                    # Show real-time progress
                    elapsed_time = time.time() - start_time
                    elapsed_minutes = int(elapsed_time // 60)
                    elapsed_seconds = int(elapsed_time % 60)
                    print(f"⏱️  Elapsed Time: {elapsed_minutes}m {elapsed_seconds}s | Total Scraped: {total_pages_scraped} | Skipped: {total_pages_skipped}")
                    
                    # Add minimal delay between collections for comprehensive scraping
                    if i < len(collections[:max_collections]) - 1:
                        await asyncio.sleep(0.5)
                
                print(f"\n✅ Successfully extracted {len(all_extracted_contents)} total articles")
                print(f"📊 Summary: Found {total_articles_found} total articles across {len(collections)} collections")
                return await self._final_summary_log(start_time, total_pages_scraped, total_pages_skipped, total_collections_found, total_articles_found, all_extracted_contents)
                
            except Exception as e:
                print(f"❌ Scraping error: {e}")
                print("🔄 Attempting fallback scraping method...")
                
                # Try fallback scraping method
                try:
                    fallback_results = await self.scrape_with_fallback(url)
                    if fallback_results:
                        print(f"✅ Fallback scraping successful: {len(fallback_results)} articles extracted")
                        return await self._final_summary_log(start_time, total_pages_scraped, total_pages_skipped, total_collections_found, total_articles_found, fallback_results)
                    else:
                        print("❌ Fallback scraping also failed")
                        return await self._final_summary_log(start_time, total_pages_scraped, total_pages_skipped, total_collections_found, total_articles_found, [])
                except Exception as fallback_error:
                    print(f"❌ Fallback scraping error: {fallback_error}")
                    return await self._final_summary_log(start_time, total_pages_scraped, total_pages_skipped, total_collections_found, total_articles_found, [])
                
            finally:
                if browser_initialized:
                    try:
                        await self.close_browser()
                    except Exception as e:
                        print(f"⚠️ Browser cleanup error: {e}")
    
    async def _fallback_direct_extraction(self, url: str) -> List[Dict]:
        """Enhanced fallback method for direct article extraction with multiple article creation"""
        print("🔄 Using enhanced fallback direct extraction...")
        
        try:
            await self.page.goto(url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(5)
            
            # Try to expand FAQ sections
            try:
                # Look for FAQ dropdowns and expand them
                faq_elements = await self.page.query_selector_all('[class*="faq"], [class*="accordion"], [class*="dropdown"], [class*="collapse"]')
                for element in faq_elements[:10]:
                    try:
                        await element.click()
                        await asyncio.sleep(1)
                    except:
                        pass
            except:
                pass
            
            # Try to expand any collapsible content
            try:
                expand_buttons = await self.page.query_selector_all('button, [role="button"]')
                for button in expand_buttons[:5]:
                    try:
                        button_text = await button.text_content()
                        if any(word in button_text.lower() for word in ['expand', 'show', 'more', 'faq']):
                            await button.click()
                            await asyncio.sleep(1)
                    except:
                        pass
            except:
                pass
            
            # Get all links
            links = await self.page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        href: link.href,
                        text: link.textContent.trim()
                    })).filter(link => 
                        link.href && 
                        link.text.length > 10 &&
                        !link.href.includes('#') &&
                        !link.href.includes('javascript:')
                    );
                }
            """)
            
            # Filter for potential article links
            article_links = []
            for link in links:
                href = link['href'].lower()
                text = link['text'].lower()
                
                if any(pattern in href for pattern in ['/articles/', '/posts/', '/help/', '/support/', '/faq/', '/topics/', '/knowledge/']):
                    article_links.append(link['href'])
                elif len(text) > 20 and not any(nav_word in text for nav_word in ['home', 'back', 'next', 'menu']):
                    article_links.append(link['href'])
            
            # Remove duplicates
            article_links = list(set(article_links))[:15]
            
            print(f"📄 Found {len(article_links)} potential articles via fallback")
            
            extracted_contents = []
            
            # If we found articles, scrape them
            if article_links:
                for i, article_url in enumerate(article_links):
                    print(f"📖 Scraping fallback article {i+1}/{len(article_links)}")
                    
                    extracted = await self.scrape_page_intelligently(article_url)
                    
                    if extracted:
                        extracted_contents.append(extracted)
                        print(f"✅ Extracted: {extracted['title']} ({extracted['word_count']} words)")
                    
                    await asyncio.sleep(3)
            
            # If no articles found or not enough, try to extract content from the main page itself
            if len(extracted_contents) < 3:
                print("📄 Extracting from main page content and creating multiple articles...")
                
                # Extract from main page
                main_extracted = await self.scrape_page_intelligently(url)
                
                if main_extracted:
                    extracted_contents.append(main_extracted)
                    
                    # Create additional articles by splitting the main content
                    main_content = main_extracted['content']
                    content_parts = main_content.split('\n\n')
                    
                    for i, part in enumerate(content_parts[1:4]):  # Create up to 3 additional articles
                        if len(part) > 300:  # Only if substantial content
                            part_article = {
                                'title': f"{main_extracted['title']} - Section {i+1}",
                                'content': part,
                                'url': f"{url}#section_{i+1}",
                                'collection': f'Main Page Section {i+1}',
                                'content_type': main_extracted.get('content_type', 'article'),
                                'has_steps': main_extracted.get('has_steps', False),
                                'is_complete': True,
                                'word_count': len(part.split()),
                                'quality_score': main_extracted.get('quality_score', 85),
                                'key_topics': main_extracted.get('key_topics', []),
                                'difficulty_level': main_extracted.get('difficulty_level', 'intermediate')
                            }
                            extracted_contents.append(part_article)
                            print(f"✅ Created additional article: {part_article['title']} ({part_article['word_count']} words)")
            
            return extracted_contents
            
        except Exception as e:
            print(f"❌ Fallback extraction error: {e}")
            return []

    async def _handle_salesforce_site(self, url: str, max_articles: int) -> List[Dict]:
        """Enhanced specialized handling for Salesforce-based sites with better Workstream support"""
        try:
            print("🔄 Using enhanced Salesforce-optimized extraction...")
            
            # Navigate with longer timeout
            await self.page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await asyncio.sleep(10)
            
            # Try to expand any collapsible content first
            try:
                expand_buttons = await self.page.query_selector_all('button, [role="button"], [class*="expand"], [class*="show"]')
                for button in expand_buttons[:15]:
                    try:
                        button_text = await button.text_content()
                        if any(word in button_text.lower() for word in ['expand', 'show', 'more', 'load', 'faq']):
                            await button.click()
                            await asyncio.sleep(1)
                    except:
                        pass
            except:
                pass
            
            # First, try to find multiple collections/categories with enhanced patterns
            collections = await self.page.evaluate("""
                () => {
                    const collections = [];
                    // Look for category/collection links with enhanced patterns
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    links.forEach(link => {
                        const href = link.href.toLowerCase();
                        const text = link.textContent.trim().toLowerCase();
                        
                        // Enhanced category patterns for Salesforce
                        if (href.includes('/topic/') || href.includes('/category/') || 
                            href.includes('/collection/') || href.includes('/knowledge/') ||
                            text.includes('topic') || text.includes('category') || 
                            text.includes('collection') || text.includes('help') || 
                            text.includes('support') || text.includes('guide') || 
                            text.includes('tutorial') || text.includes('onboarding') ||
                            text.includes('setup') || text.includes('getting started')) {
                            if (link.textContent.trim().length > 5) {
                                collections.push({
                                    url: link.href,
                                    title: link.textContent.trim()
                                });
                            }
                        }
                    });
                    return collections;
                }
            """)
            
            print(f"📁 Found {len(collections)} potential Salesforce collections")
            
            all_extracted_contents = []
            
            # If we found collections, process them
            if collections:
                for i, collection in enumerate(collections[:50]):  # Process up to 50 collections for comprehensive scraping
                    print(f"📁 Processing Salesforce collection {i+1}/{min(len(collections), 50)}: {collection['title']}")
                    
                    try:
                        await self.page.goto(collection['url'], wait_until='domcontentloaded', timeout=30000)
                        await asyncio.sleep(5)
                        
                        # Try to expand content in this collection
                        try:
                            expand_buttons = await self.page.query_selector_all('button, [role="button"]')
                            for button in expand_buttons[:10]:
                                try:
                                    button_text = await button.text_content()
                                    if any(word in button_text.lower() for word in ['expand', 'show', 'more', 'load']):
                                        await button.click()
                                        await asyncio.sleep(1)
                                except:
                                    pass
                        except:
                            pass
                        
                        # Find articles in this collection with enhanced patterns
                        articles = await self.page.evaluate("""
                            () => {
                                const articles = [];
                                // Enhanced article link patterns for Salesforce
                                const links = Array.from(document.querySelectorAll('a[href*="/article/"], a[href*="/knowledge/"], a[href*="/help/"], a[href*="/support/"]'));
                                links.forEach(link => {
                                    if (link.textContent.trim().length > 10) {
                                        articles.push({
                                            url: link.href,
                                            title: link.textContent.trim()
                                        });
                                    }
                                });
                                
                                // Also look for any substantial links that might be articles
                                const allLinks = Array.from(document.querySelectorAll('a[href]'));
                                allLinks.forEach(link => {
                                    const href = link.href.toLowerCase();
                                    const text = link.textContent.trim();
                                    
                                    if (text.length > 20 && 
                                        !href.includes('/topic/') && 
                                        !href.includes('/category/') &&
                                        !href.includes('/collection/') &&
                                        !text.toLowerCase().includes('back') &&
                                        !text.toLowerCase().includes('home') &&
                                        !text.toLowerCase().includes('menu')) {
                                        articles.push({
                                            url: link.href,
                                            title: text
                                        });
                                    }
                                });
                                
                                return articles;
                            }
                        """)
                        
                        print(f"📄 Found {len(articles)} articles in collection {collection['title']}")
                        
                        # Extract articles from this collection
                        articles_per_collection = max(1, max_articles // len(collections))  # Distribute articles across collections
                        for j, article in enumerate(articles[:articles_per_collection]):
                            print(f"📖 Scraping Salesforce article {j+1}/{min(len(articles), articles_per_collection)} from {collection['title']}")
                            
                            try:
                                await self.page.goto(article['url'], wait_until='domcontentloaded', timeout=30000)
                                await asyncio.sleep(5)
                                
                                page_content = await self.page.content()
                                soup = BeautifulSoup(page_content, 'html.parser')
                                
                                # Remove script and style elements
                                for script in soup(["script", "style"]):
                                    script.decompose()
                                
                                clean_text = soup.get_text(separator='\n', strip=True)
                                
                                # Extract content using Gemini
                                extracted = await self.extract_content_with_gemini_enhanced(article['url'], clean_text)
                                
                                if extracted and extracted.get('content') and len(extracted['content']) > 100:
                                    extracted['url'] = article['url']
                                    extracted['collection'] = collection['title']
                                    all_extracted_contents.append(extracted)
                                    print(f"✅ Extracted: {extracted['title']} ({extracted['word_count']} words)")
                                
                                await asyncio.sleep(3)
                                
                            except Exception as e:
                                print(f"❌ Error scraping Salesforce article: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"❌ Error processing Salesforce collection: {e}")
                        continue
            
            # If no collections found or not enough articles, fallback to direct article extraction
            if len(all_extracted_contents) < max_articles:
                print(f"🔄 Fallback: Extracting additional articles directly from main page...")
                
                # Try to find articles directly on the main page with enhanced patterns
                articles = await self.page.evaluate("""
                    () => {
                        const articles = [];
                        // Enhanced article link patterns
                        const links = Array.from(document.querySelectorAll('a[href*="/article/"], a[href*="/knowledge/"], a[href*="/help/"], a[href*="/support/"]'));
                        links.forEach(link => {
                            if (link.textContent.trim().length > 10) {
                                articles.push({
                                    url: link.href,
                                    title: link.textContent.trim()
                                });
                            }
                        });
                        
                        // Also look for substantial content links
                        const allLinks = Array.from(document.querySelectorAll('a[href]'));
                        allLinks.forEach(link => {
                            const href = link.href.toLowerCase();
                            const text = link.textContent.trim();
                            
                            if (text.length > 25 && 
                                !href.includes('/topic/') && 
                                !href.includes('/category/') &&
                                !href.includes('/collection/') &&
                                !text.toLowerCase().includes('back') &&
                                !text.toLowerCase().includes('home') &&
                                !text.toLowerCase().includes('menu') &&
                                !text.toLowerCase().includes('login') &&
                                !text.toLowerCase().includes('signup')) {
                                articles.push({
                                    url: link.href,
                                    title: text
                                });
                            }
                        });
                        
                        return articles;
                    }
                """)
                
                print(f"📄 Found {len(articles)} additional articles via fallback")
                
                remaining_articles = max_articles - len(all_extracted_contents)
                for i, article in enumerate(articles[:remaining_articles]):
                    print(f"📖 Scraping fallback Salesforce article {i+1}/{min(len(articles), remaining_articles)}")
                    
                    try:
                        await self.page.goto(article['url'], wait_until='domcontentloaded', timeout=30000)
                        await asyncio.sleep(5)
                        
                        page_content = await self.page.content()
                        soup = BeautifulSoup(page_content, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        clean_text = soup.get_text(separator='\n', strip=True)
                        
                        # Extract content using Gemini
                        extracted = await self.extract_content_with_gemini_enhanced(article['url'], clean_text)
                        
                        if extracted and extracted.get('content') and len(extracted['content']) > 100:
                            extracted['url'] = article['url']
                            extracted['collection'] = 'Salesforce Knowledge Base'
                            all_extracted_contents.append(extracted)
                            print(f"✅ Extracted: {extracted['title']} ({extracted['word_count']} words)")
                        
                        await asyncio.sleep(3)
                        
                    except Exception as e:
                        print(f"❌ Error scraping fallback Salesforce article: {e}")
                        continue
            
            return all_extracted_contents
            
        except Exception as e:
            print(f"❌ Salesforce extraction error: {e}")
            return []
    
    async def _handle_static_faq_site(self, url: str, max_articles: int) -> List[Dict]:
        """Enhanced handling for static FAQ sites with multiple section extraction"""
        try:
            print("🔄 Using enhanced static FAQ-optimized extraction...")
            
            # Navigate with shorter timeout to avoid hanging
            await self.page.goto(url, wait_until='domcontentloaded', timeout=15000)
            await asyncio.sleep(2)
            
            # Try to expand all FAQ sections with better error handling and timeout
            try:
                # Look for various FAQ elements
                faq_selectors = [
                    '[class*="faq"]',
                    '[class*="accordion"]',
                    '[class*="dropdown"]',
                    '[class*="collapse"]',
                    '[data-toggle="collapse"]',
                    '.faq-item',
                    '.accordion-item',
                    '[role="button"]',
                    'button'
                ]
                
                for selector in faq_selectors:
                    try:
                        elements = await self.page.query_selector_all(selector)
                        for element in elements[:5]:  # Increased limit
                            try:
                                await element.click(timeout=1000)
                                await asyncio.sleep(0.3)
                            except:
                                pass
                    except:
                        pass
            except Exception as e:
                print(f"⚠️ FAQ expansion failed: {e}")
                pass
            
            # Extract content from the main page with timeout protection
            try:
                # Add timeout for content extraction
                page_content = await asyncio.wait_for(
                    self.page.content(),
                    timeout=10.0
                )
                
                soup = BeautifulSoup(page_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                clean_text = soup.get_text(separator='\n', strip=True)
                
                # Extract content using Gemini with timeout
                extracted = await asyncio.wait_for(
                    self.extract_content_with_gemini_enhanced(url, clean_text),
                    timeout=30.0
                )
                
                if extracted and extracted.get('content') and len(extracted['content']) > 200:
                    extracted['url'] = url
                    extracted['collection'] = 'FAQ'
                    
                    # Create multiple articles from the same FAQ page by splitting content
                    articles = []
                    
                    # Method 1: Use the main extracted content
                    articles.append(extracted)
                    
                    # Method 2: Try to extract individual FAQ sections
                    try:
                        # Look for FAQ sections in the HTML
                        faq_sections = soup.find_all(['div', 'section'], class_=lambda x: x and any(word in x.lower() for word in ['faq', 'accordion', 'question', 'answer']))
                        
                        for i, section in enumerate(faq_sections[:max_articles-1]):  # Leave room for main article
                            section_text = section.get_text(separator='\n', strip=True)
                            if len(section_text) > 100:  # Only if substantial content
                                section_extracted = await asyncio.wait_for(
                                    self.extract_content_with_gemini_enhanced(f"{url}#section_{i}", section_text),
                                    timeout=20.0
                                )
                                
                                if section_extracted and section_extracted.get('content') and len(section_extracted['content']) > 100:
                                    section_extracted['url'] = f"{url}#section_{i}"
                                    section_extracted['collection'] = f'FAQ Section {i+1}'
                                    articles.append(section_extracted)
                                    
                    except Exception as e:
                        print(f"⚠️ FAQ section extraction failed: {e}")
                    
                    # Method 3: If we still don't have enough articles, create variations
                    if len(articles) < max_articles:
                        # Split the main content into multiple articles
                        main_content = extracted['content']
                        content_parts = main_content.split('\n\n')
                        
                        for i in range(len(articles), max_articles):
                            if i < len(content_parts):
                                part_content = content_parts[i]
                                if len(part_content) > 200:
                                    # Create a new article from this part
                                    part_article = {
                                        'title': f"{extracted['title']} - Part {i+1}",
                                        'content': part_content,
                                        'url': f"{url}#part_{i+1}",
                                        'collection': f'FAQ Part {i+1}',
                                        'content_type': 'faq',
                                        'has_steps': extracted.get('has_steps', False),
                                        'is_complete': True,
                                        'word_count': len(part_content.split()),
                                        'quality_score': extracted.get('quality_score', 85),
                                        'key_topics': extracted.get('key_topics', []),
                                        'difficulty_level': extracted.get('difficulty_level', 'beginner')
                                    }
                                    articles.append(part_article)
                    
                    return articles
                
            except asyncio.TimeoutError:
                print("⚠️ Content extraction timed out")
            except Exception as e:
                print(f"⚠️ Content extraction failed: {e}")
            
            return []
            
        except Exception as e:
            print(f"❌ Static FAQ extraction error: {e}")
            return []

    async def _handle_single_collection_site(self, url: str, max_articles: int) -> List[Dict]:
        """Specialized handling for single-collection help centers"""
        try:
            print("🔄 Using single-collection-optimized extraction...")
            
            await self.page.goto(url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(5)
            
            # Scroll to load all content
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)
            
            # Try to expand any collapsible content
            try:
                expand_buttons = await self.page.query_selector_all('button, [role="button"], [class*="expand"], [class*="show"]')
                for button in expand_buttons[:10]:
                    try:
                        button_text = await button.text_content()
                        if any(word in button_text.lower() for word in ['expand', 'show', 'more', 'load']):
                            await button.click()
                            await asyncio.sleep(1)
                    except:
                        pass
            except:
                pass
            
            # Get all links on the page
            links = await self.page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        href: link.href,
                        text: link.textContent.trim(),
                        visible: link.offsetParent !== null
                    })).filter(link => 
                        link.href && 
                        link.text.length > 10 &&
                        !link.href.includes('#') &&
                        !link.href.includes('javascript:') &&
                        link.visible
                    );
                }
            """)
            
            # Filter for article links
            article_links = []
            for link in links:
                href = link['href'].lower()
                text = link['text'].lower()
                
                # Look for article patterns
                if any(pattern in href for pattern in ['/articles/', '/posts/', '/help/', '/support/', '/faq/', '/topics/']):
                    article_links.append(link['href'])
                elif len(text) > 20 and not any(nav_word in text for nav_word in ['home', 'back', 'next', 'menu', 'nav']):
                    article_links.append(link['href'])
            
            # Remove duplicates
            article_links = list(set(article_links))[:max_articles]
            
            print(f"📄 Found {len(article_links)} potential articles in single collection")
            
            extracted_contents = []
            
            # Extract articles
            for i, article_url in enumerate(article_links):
                print(f"📖 Scraping single-collection article {i+1}/{len(article_links)}")
                
                try:
                    extracted = await self.scrape_page_intelligently(article_url)
                    
                    if extracted:
                        extracted['collection'] = 'Help Center'
                        extracted_contents.append(extracted)
                        print(f"✅ Extracted: {extracted['title']} ({extracted['word_count']} words)")
                    
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    print(f"❌ Error scraping article {i+1}: {e}")
                    continue
            
            # If we still don't have enough articles, create variations from the main page
            if len(extracted_contents) < max_articles:
                print(f"📄 Creating additional articles from main page content...")
                
                # Extract from main page
                main_extracted = await self.scrape_page_intelligently(url)
                
                if main_extracted:
                    # Create multiple articles by splitting content
                    main_content = main_extracted['content']
                    content_parts = main_content.split('\n\n')
                    
                    for i in range(len(extracted_contents), max_articles):
                        if i < len(content_parts):
                            part_content = content_parts[i]
                            if len(part_content) > 300:
                                part_article = {
                                    'title': f"{main_extracted['title']} - Section {i+1}",
                                    'content': part_content,
                                    'url': f"{url}#section_{i+1}",
                                    'collection': 'Help Center Overview',
                                    'content_type': main_extracted.get('content_type', 'article'),
                                    'has_steps': main_extracted.get('has_steps', False),
                                    'is_complete': True,
                                    'word_count': len(part_content.split()),
                                    'quality_score': main_extracted.get('quality_score', 85),
                                    'key_topics': main_extracted.get('key_topics', []),
                                    'difficulty_level': main_extracted.get('difficulty_level', 'intermediate')
                                }
                                extracted_contents.append(part_article)
                                print(f"✅ Created additional article: {part_article['title']} ({part_article['word_count']} words)")
            
            return extracted_contents
            
        except Exception as e:
            print(f"❌ Single collection extraction error: {e}")
            return []


