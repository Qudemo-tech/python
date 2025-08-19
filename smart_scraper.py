#!/usr/bin/env python3
"""
Smart Website Scraper - LLM-Powered Content Filtering & Adaptive Timeouts
Only scrapes demo-relevant content and adapts to website size
"""

import asyncio
import json
import re
import time
import psutil
import gc
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import logging
import openai
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartScraper:
    def __init__(self, openai_api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.scraped_urls = set()
        self.demo_relevant_urls = set()
        self.skipped_urls = set()
        self.start_time = None
        self.max_time_per_site = 3600  # 1 hour default
        self.max_articles_per_site = 200  # Limit articles per site
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
        logger.info("Smart Scraper Initialized with LLM filtering and memory management")
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        try:
            memory = psutil.virtual_memory()
            return {
                "percent": memory.percent,
                "available_mb": memory.available / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            return {"percent": 0, "available_mb": 0, "used_mb": 0, "total_mb": 0}
    
    def should_stop_for_memory(self) -> bool:
        """Check if we should stop due to high memory usage"""
        memory_info = self.check_memory_usage()
        if memory_info["percent"] > (self.memory_threshold * 100):
            logger.warning(f"High memory usage: {memory_info['percent']:.1f}% - stopping scraping")
            return True
        return False
    
    def cleanup_memory(self):
        """Force garbage collection to free memory"""
        try:
            gc.collect()
            memory_info = self.check_memory_usage()
            logger.info(f"Memory cleanup completed: {memory_info['percent']:.1f}% usage")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def is_demo_relevant_url(self, url: str, title: str = "", description: str = "") -> bool:
        """
        Simplified URL filtering - only skip obvious non-content URLs
        """
        try:
            # Only skip very obvious non-content URLs
            skip_patterns = [
                r'/privacy', r'/terms', r'/legal', r'/cookies', r'/contact',
                r'/about', r'/careers', r'/blog', r'/news', r'/press',
                r'/login', r'/signup', r'/account', r'/billing',
                r'/api/', r'/admin/', r'/internal/',
                r'\.pdf$', r'\.zip$', r'\.doc$', r'\.xls$'
            ]
            
            for pattern in skip_patterns:
                if re.search(pattern, url.lower()):
                    return False
            
            # Accept all other URLs for processing
            return True
            
        except Exception as e:
            logger.warning(f"URL filtering failed for {url}: {e}")
            # Accept URL if filtering fails
            return True
    
    def is_demo_relevant_content(self, content: str, title: str) -> Dict[str, Any]:
        """
        Use LLM to analyze content relevance and quality for demos
        """
        try:
            # Truncate content for LLM analysis
            content_preview = content[:1000] + "..." if len(content) > 1000 else content
            
            prompt = f"""
            Analyze this content for demo relevance:
            
            Title: {title}
            Content: {content_preview}
            
            Rate on a scale of 1-10:
            1. Relevance to product demo (how useful for showing product features)
            2. Content quality (clarity, completeness, helpfulness)
            3. Actionability (can users take action based on this)
            
            Also identify:
            - Content type (setup guide, troubleshooting, feature explanation, etc.)
            - Target audience (beginner, intermediate, advanced)
            - Key topics covered
            
            Respond in JSON format:
            {{
                "relevance_score": 1-10,
                "quality_score": 1-10,
                "actionability_score": 1-10,
                "content_type": "string",
                "audience_level": "beginner|intermediate|advanced",
                "key_topics": ["topic1", "topic2"],
                "is_demo_relevant": true/false
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.warning(f"LLM content analysis failed: {e}")
            # Fallback analysis
            word_count = len(content.split())
            has_steps = any(word in content.lower() for word in ['step', 'guide', 'tutorial', 'setup', 'how to'])
            
            return {
                "relevance_score": 7 if has_steps else 5,
                "quality_score": 8 if word_count > 100 else 5,
                "actionability_score": 7 if has_steps else 4,
                "content_type": "guide" if has_steps else "information",
                "audience_level": "beginner" if word_count < 200 else "intermediate",
                "key_topics": [],
                "is_demo_relevant": word_count > 50 and has_steps
            }
    
    async def estimate_website_size(self, base_url: str) -> Dict[str, Any]:
        """
        Quickly estimate website size to set appropriate timeouts
        """
        try:
            logger.info(f"Estimating website size for: {base_url}")
            
            # Get homepage and analyze structure
            response = self.session.get(base_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Count potential content pages - be more aggressive
            links = soup.find_all('a', href=True)
            content_links = []
            
            for link in links:
                href = link.get('href')
                if href and not href.startswith(('http', 'mailto', 'tel', '#')):
                    full_url = urljoin(base_url, href)
                    if self.is_same_domain(full_url, base_url):
                        content_links.append(full_url)
            
            # Also look for navigation menus and content areas
            nav_links = soup.find_all(['nav', 'menu', '.navigation', '.menu', '.sidebar'])
            for nav in nav_links:
                nav_links_found = nav.find_all('a', href=True)
                for link in nav_links_found:
                    href = link.get('href')
                    if href and not href.startswith(('http', 'mailto', 'tel', '#')):
                        full_url = urljoin(base_url, href)
                        if self.is_same_domain(full_url, base_url):
                            content_links.append(full_url)
            
            # For help centers, try to find article URLs in the HTML source
            if 'help' in base_url.lower():
                logger.info("Help center detected - searching for article patterns in HTML source")
                
                # Look for article URLs in the page source
                page_source = response.text
                
                # Common patterns for help center articles
                article_patterns = [
                    r'/articles/[^"\s]+',
                    r'/help/[^"\s]+', 
                    r'/support/[^"\s]+',
                    r'/docs/[^"\s]+',
                    r'/guides/[^"\s]+',
                    r'/tutorials/[^"\s]+',
                    r'/faq/[^"\s]+',
                    r'/knowledge/[^"\s]+',
                    r'/learn/[^"\s]+',
                    r'/en/articles/[^"\s]+',
                    r'/en/help/[^"\s]+',
                    r'/en/support/[^"\s]+'
                ]
                
                for pattern in article_patterns:
                    matches = re.findall(pattern, page_source)
                    for match in matches:
                        if match.startswith('/'):
                            full_url = urljoin(base_url, match)
                            if self.is_same_domain(full_url, base_url):
                                content_links.append(full_url)
                
                logger.info(f"Found {len(content_links)} potential article URLs from HTML source")
            
            # Analyze link patterns to estimate size
            unique_paths = set()
            for link in content_links:
                path = urlparse(link).path
                if path and len(path) > 1:
                    unique_paths.add(path)
            
            estimated_pages = len(unique_paths)
            
            # Special handling for help centers and documentation sites
            help_indicators = [
                'help', 'support', 'docs', 'documentation', 'guide', 'tutorial',
                'faq', 'knowledge', 'learn', 'how-to', 'articles'
            ]
            
            is_help_site = any(indicator in base_url.lower() for indicator in help_indicators)
            
            # If it's a help site and we found few links, it might be a large help center
            if is_help_site and estimated_pages < 10:
                logger.info(f"Detected help center site with few visible links - treating as large site")
                estimated_pages = 200  # Assume large help center
                size_category = "large"
                max_time = 5400  # 1.5 hours
                max_articles = 100
            else:
                # Categorize website size with Render-friendly limits
                if estimated_pages < 50:
                    size_category = "small"
                    max_time = 1800  # 30 minutes
                    max_articles = 30  # Reduced for Render
                elif estimated_pages < 200:
                    size_category = "medium"
                    max_time = 3600  # 1 hour
                    max_articles = 60  # Reduced for Render
                else:
                    size_category = "large"
                    max_time = 5400  # 1.5 hours (reduced from 2 hours)
                    max_articles = 100  # Reduced for Render
            
            logger.info(f"Website size estimate: {size_category} ({estimated_pages} pages)")
            logger.info(f"Max time: {max_time/60:.0f} minutes, Max articles: {max_articles}")
            
            return {
                "size_category": size_category,
                "estimated_pages": estimated_pages,
                "max_time": max_time,
                "max_articles": max_articles,
                "content_links": list(unique_paths)[:200]  # Increased limit for better discovery
            }
            
        except Exception as e:
            logger.warning(f"Website size estimation failed: {e}")
            return {
                "size_category": "unknown",
                "estimated_pages": 100,
                "max_time": 3600,
                "max_articles": 50,  # Conservative limit
                "content_links": []
            }
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain"""
        return urlparse(url1).netloc == urlparse(url2).netloc
    
    def should_continue_scraping(self) -> bool:
        """Check if we should continue scraping based on time, limits, and memory"""
        if not self.start_time:
            return True
        
        elapsed_time = time.time() - self.start_time
        articles_scraped = len(self.scraped_urls)
        
        # Check time limit
        if elapsed_time > self.max_time_per_site:
            logger.info(f"Time limit reached: {elapsed_time/60:.1f} minutes")
            return False
        
        # Check article limit
        if articles_scraped >= self.max_articles_per_site:
            logger.info(f"Article limit reached: {articles_scraped} articles")
            return False
        
        # Check memory usage
        if self.should_stop_for_memory():
            return False
        
        return True
    
    async def smart_scrape_website(self, base_url: str, company_name: str) -> Dict[str, Any]:
        """
        Smart website scraping with LLM filtering, adaptive timeouts, and memory management
        """
        self.start_time = time.time()
        self.scraped_urls.clear()
        self.demo_relevant_urls.clear()
        self.skipped_urls.clear()
        
        logger.info(f"Starting smart scraping for: {base_url}")
        
        # Initial memory check
        memory_info = self.check_memory_usage()
        logger.info(f"Initial memory usage: {memory_info['percent']:.1f}%")
        
        # Estimate website size and set limits
        size_estimate = await self.estimate_website_size(base_url)
        self.max_time_per_site = size_estimate["max_time"]
        self.max_articles_per_site = size_estimate["max_articles"]
        
        logger.info(f"Website category: {size_estimate['size_category']}")
        logger.info(f"Time limit: {self.max_time_per_site/60:.0f} minutes")
        logger.info(f"Article limit: {self.max_articles_per_site}")
        
        # Start scraping with smart filtering
        all_content = []
        skipped_count = 0
        
        try:
            # Process homepage first
            homepage_content = await self.scrape_page(base_url, company_name)
            if homepage_content:
                all_content.append(homepage_content)
            
            # For help centers, try to discover more content
            if size_estimate["size_category"] == "large" and "help" in base_url.lower():
                logger.info("Large help center detected - attempting comprehensive content discovery")
                
                # Try common help center patterns
                help_patterns = [
                    "/articles/", "/help/", "/support/", "/docs/", "/guides/",
                    "/tutorials/", "/faq/", "/knowledge/", "/learn/"
                ]
                
                discovered_urls = []
                for pattern in help_patterns:
                    try:
                        test_url = urljoin(base_url, pattern)
                        logger.info(f"Testing help center pattern: {test_url}")
                        response = self.session.get(test_url, timeout=5)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            links = soup.find_all('a', href=True)
                            for link in links:
                                href = link.get('href')
                                if href and not href.startswith(('http', 'mailto', 'tel', '#')):
                                    full_url = urljoin(test_url, href)
                                    if self.is_same_domain(full_url, base_url):
                                        discovered_urls.append(full_url)
                    except Exception as e:
                        logger.warning(f"Failed to test pattern {pattern}: {e}")
                        continue
                
                # Also try to find articles by testing common article URLs
                logger.info("Testing common article URL patterns")
                common_article_patterns = [
                    "/articles/", "/en/articles/", "/help/articles/", "/support/articles/",
                    "/docs/articles/", "/guides/", "/tutorials/", "/faq/"
                ]
                
                for pattern in common_article_patterns:
                    try:
                        test_url = urljoin(base_url, pattern)
                        response = self.session.get(test_url, timeout=5)
                        if response.status_code == 200:
                            # Look for article links in the response
                            page_source = response.text
                            article_matches = re.findall(r'/articles/[^"\s]+', page_source)
                            for match in article_matches:
                                if match.startswith('/'):
                                    full_url = urljoin(base_url, match)
                                    if self.is_same_domain(full_url, base_url):
                                        discovered_urls.append(full_url)
                    except Exception as e:
                        logger.warning(f"Failed to test article pattern {pattern}: {e}")
                        continue
                
                # Try different URL structures that Settle might use
                logger.info("Testing different URL structures")
                url_structures = [
                    "/en/articles/", "/articles/", "/help/", "/support/", "/docs/",
                    "/en/help/", "/en/support/", "/en/docs/", "/help/en/", "/support/en/"
                ]
                
                for structure in url_structures:
                    try:
                        test_url = urljoin(base_url, structure)
                        response = self.session.get(test_url, timeout=5)
                        if response.status_code == 200:
                            # Look for any links in the response
                            page_source = response.text
                            link_matches = re.findall(r'href="([^"]+)"', page_source)
                            for match in link_matches:
                                if match.startswith('/') and 'article' in match.lower():
                                    full_url = urljoin(base_url, match)
                                    if self.is_same_domain(full_url, base_url):
                                        discovered_urls.append(full_url)
                    except Exception as e:
                        logger.warning(f"Failed to test URL structure {structure}: {e}")
                        continue
                
                # Add discovered URLs to content links
                size_estimate["content_links"].extend(discovered_urls[:200])  # Increased limit
                logger.info(f"Discovered {len(discovered_urls)} additional URLs from help center patterns")
                
                # If still no URLs found, try to discover real URLs from Settle's help center
                if len(discovered_urls) == 0:
                    logger.info("No URLs discovered - attempting to find real Settle help center URLs")
                    
                    # First, try to get the actual help center structure
                    try:
                        # Get the main help center page
                        response = self.session.get(base_url, timeout=10)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look for actual article links in the page
                        all_links = soup.find_all('a', href=True)
                        real_article_urls = []
                        
                        for link in all_links:
                            href = link.get('href')
                            if href and not href.startswith(('http', 'mailto', 'tel', '#')):
                                full_url = urljoin(base_url, href)
                                if self.is_same_domain(full_url, base_url):
                                    # Check if it looks like an article URL
                                    if any(pattern in href.lower() for pattern in ['/articles/', '/help/', '/support/', '/docs/']):
                                        real_article_urls.append(full_url)
                        
                        logger.info(f"Found {len(real_article_urls)} real article URLs from homepage")
                        
                        # Also try to find category pages that might contain article lists
                        category_urls = []
                        for link in all_links:
                            href = link.get('href')
                            if href and not href.startswith(('http', 'mailto', 'tel', '#')):
                                full_url = urljoin(base_url, href)
                                if self.is_same_domain(full_url, base_url):
                                    # Look for category-like URLs
                                    if any(pattern in href.lower() for pattern in ['/categories/', '/topics/', '/sections/']):
                                        category_urls.append(full_url)
                        
                        logger.info(f"Found {len(category_urls)} category URLs from homepage")
                        
                        # Try to get articles from category pages
                        for category_url in category_urls[:5]:  # Limit to avoid too many requests
                            try:
                                logger.info(f"Checking category page: {category_url}")
                                cat_response = self.session.get(category_url, timeout=5)
                                if cat_response.status_code == 200:
                                    cat_soup = BeautifulSoup(cat_response.content, 'html.parser')
                                    cat_links = cat_soup.find_all('a', href=True)
                                    
                                    for link in cat_links:
                                        href = link.get('href')
                                        if href and not href.startswith(('http', 'mailto', 'tel', '#')):
                                            full_url = urljoin(category_url, href)
                                            if self.is_same_domain(full_url, base_url):
                                                if any(pattern in href.lower() for pattern in ['/articles/', '/help/', '/support/']):
                                                    real_article_urls.append(full_url)
                            except Exception as e:
                                logger.warning(f"Failed to check category page {category_url}: {e}")
                                continue
                        
                        # Add discovered real URLs
                        size_estimate["content_links"].extend(real_article_urls)
                        logger.info(f"Added {len(real_article_urls)} real article URLs from Settle help center")
                        
                    except Exception as e:
                        logger.warning(f"Failed to discover real URLs: {e}")
                        
                        # Fallback: try some common Settle help center patterns that might actually exist
                        logger.info("Trying fallback URL patterns")
                        fallback_urls = [
                            "/en/articles/getting-started-with-settle",
                            "/en/articles/how-to-settle",
                            "/en/articles/account-setup",
                            "/en/articles/payment-methods",
                            "/en/articles/transactions",
                            "/en/articles/security",
                            "/en/articles/troubleshooting",
                            "/en/articles/contact-support",
                            "/en/articles/faq",
                            "/en/articles/help"
                        ]
                        
                        for url_path in fallback_urls:
                            test_url = urljoin(base_url, url_path)
                            try:
                                response = self.session.get(test_url, timeout=5)
                                if response.status_code == 200:
                                    size_estimate["content_links"].append(url_path)
                                    logger.info(f"Found valid fallback URL: {test_url}")
                            except:
                                continue
            
            # Process other content links
            for i, link in enumerate(size_estimate["content_links"][:500]):  # Increased limit for comprehensive coverage
                if not self.should_continue_scraping():
                    break
                
                full_url = urljoin(base_url, link)
                
                # Scrape the page directly without URL filtering
                content = await self.scrape_page(full_url, company_name)
                if content:
                    all_content.append(content)
                
                # Memory cleanup every 5 articles
                if (i + 1) % 5 == 0:
                    self.cleanup_memory()
                
                # Small delay to be respectful
                await asyncio.sleep(0.5)
            
            # Calculate final metrics
            elapsed_time = time.time() - self.start_time
            total_articles = len(all_content)
            
            # Final memory cleanup
            self.cleanup_memory()
            
            logger.info(f"Smart scraping completed:")
            logger.info(f"    Articles scraped: {total_articles}")
            logger.info(f"    URLs skipped: {skipped_count}")
            logger.info(f"    Time taken: {elapsed_time/60:.1f} minutes")
            
            # Fix division by zero error
            total_processed = total_articles + skipped_count
            if total_processed > 0:
                success_rate = (total_articles / total_processed) * 100
                logger.info(f"    Success rate: {success_rate:.1f}%")
            else:
                logger.info(f"    Success rate: 0.0% (no URLs processed)")
            
            return {
                "success": True,
                "data": {
                    "chunks": all_content,
                    "summary": {
                        "total_items": total_articles,
                        "scraped_urls": len(self.scraped_urls),
                        "skipped_urls": len(self.skipped_urls),
                        "elapsed_time_minutes": elapsed_time / 60,
                        "website_size": size_estimate["size_category"]
                    }
                },
                "company_name": company_name,
                "website_url": base_url
            }
            
        except Exception as e:
            logger.error(f"Smart scraping failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def scrape_page(self, url: str, company_name: str) -> Optional[Dict[str, Any]]:
        """Scrape a single page with content relevance analysis"""
        try:
            if url in self.scraped_urls:
                return None
            
            response = self.session.get(url, timeout=10)
            
            # Check if page exists and has content
            if response.status_code != 200:
                logger.warning(f"Skipping {url} - status code {response.status_code}")
                return None
            
            # Check if content is substantial
            content_length = len(response.content)
            if content_length < 1000:  # Require at least 1KB of content
                logger.warning(f"Skipping {url} - too little content ({content_length} bytes)")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic content
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Untitled"
            
            # Check for obvious error pages
            if any(error_indicator in title_text.lower() for error_indicator in ['404', 'not found', 'error', 'page not found']):
                logger.warning(f"Skipping {url} - appears to be error page: {title_text}")
                return None
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '.main-content', '.post-content',
                '.entry-content', '.article-content', '.page-content', '.help-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if text and len(text) > 50:
                        content += text + "\n\n"
            
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text().strip()
            
            # Require substantial content
            if not content or len(content) < 200:  # Increased minimum content requirement
                logger.warning(f"Skipping {url} - insufficient content ({len(content)} chars)")
                return None
            
            # Analyze content relevance
            relevance_analysis = self.is_demo_relevant_content(content, title_text)
            
            # For help centers, accept all content without filtering
            is_help_site = any(indicator in url.lower() for indicator in ['help', 'support', 'docs', 'documentation'])
            
            if is_help_site:
                # For help sites, accept all content without filtering
                pass
            else:
                # Only include highly relevant content for other sites
                min_relevance_score = 6
                if not relevance_analysis.get("is_demo_relevant", False) and relevance_analysis.get("relevance_score", 0) < min_relevance_score:
                    self.skipped_urls.add(url)
                    return None
            
            self.scraped_urls.add(url)
            self.demo_relevant_urls.add(url)
            
            logger.info(f"Successfully scraped: {title_text} ({len(content)} chars)")
            
            return {
                "title": title_text,
                "content": content,
                "url": url,
                "company_name": company_name,
                "relevance_score": relevance_analysis["relevance_score"],
                "quality_score": relevance_analysis["quality_score"],
                "content_type": relevance_analysis["content_type"],
                "audience_level": relevance_analysis["audience_level"],
                "key_topics": relevance_analysis["key_topics"],
                "word_count": len(content.split())
            }
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None
