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
        
        logger.info("🧠 Smart Scraper Initialized with LLM filtering and memory management")
    
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
            logger.warning(f"⚠️ High memory usage: {memory_info['percent']:.1f}% - stopping scraping")
            return True
        return False
    
    def cleanup_memory(self):
        """Force garbage collection to free memory"""
        try:
            gc.collect()
            memory_info = self.check_memory_usage()
            logger.info(f"🧹 Memory cleanup completed: {memory_info['percent']:.1f}% usage")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def is_demo_relevant_url(self, url: str, title: str = "", description: str = "") -> bool:
        """
        Use LLM to determine if URL is relevant for demo purposes
        """
        try:
            # Quick heuristic filter first
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
            
            # LLM analysis for borderline cases
            context = f"URL: {url}\nTitle: {title}\nDescription: {description}"
            
            prompt = f"""
            Analyze if this webpage would be useful for a product demo or customer support:
            
            {context}
            
            Consider:
            - Does it contain product features, setup guides, or troubleshooting?
            - Is it customer-facing documentation or help content?
            - Would it help someone understand how to use the product?
            
            Respond with only: RELEVANT or NOT_RELEVANT
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "RELEVANT"
            
        except Exception as e:
            logger.warning(f"LLM filtering failed for {url}: {e}")
            # Fallback to heuristic
            return not any(pattern in url.lower() for pattern in ['/privacy', '/terms', '/legal'])
    
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
            logger.info(f"🔍 Estimating website size for: {base_url}")
            
            # Get homepage and analyze structure
            response = self.session.get(base_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Count potential content pages
            links = soup.find_all('a', href=True)
            content_links = []
            
            for link in links:
                href = link.get('href')
                if href and not href.startswith(('http', 'mailto', 'tel', '#')):
                    full_url = urljoin(base_url, href)
                    if self.is_same_domain(full_url, base_url):
                        content_links.append(full_url)
            
            # Analyze link patterns to estimate size
            unique_paths = set()
            for link in content_links:
                path = urlparse(link).path
                if path and len(path) > 1:
                    unique_paths.add(path)
            
            estimated_pages = len(unique_paths)
            
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
            
            logger.info(f"📊 Website size estimate: {size_category} ({estimated_pages} pages)")
            logger.info(f"⏱️ Max time: {max_time/60:.0f} minutes, Max articles: {max_articles}")
            
            return {
                "size_category": size_category,
                "estimated_pages": estimated_pages,
                "max_time": max_time,
                "max_articles": max_articles,
                "content_links": list(unique_paths)[:50]  # Reduced limit for Render
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
            logger.info(f"⏱️ Time limit reached: {elapsed_time/60:.1f} minutes")
            return False
        
        # Check article limit
        if articles_scraped >= self.max_articles_per_site:
            logger.info(f"📄 Article limit reached: {articles_scraped} articles")
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
        
        logger.info(f"🧠 Starting smart scraping for: {base_url}")
        
        # Initial memory check
        memory_info = self.check_memory_usage()
        logger.info(f"💾 Initial memory usage: {memory_info['percent']:.1f}%")
        
        # Estimate website size and set limits
        size_estimate = await self.estimate_website_size(base_url)
        self.max_time_per_site = size_estimate["max_time"]
        self.max_articles_per_site = size_estimate["max_articles"]
        
        logger.info(f"📊 Website category: {size_estimate['size_category']}")
        logger.info(f"⏱️ Time limit: {self.max_time_per_site/60:.0f} minutes")
        logger.info(f"📄 Article limit: {self.max_articles_per_site}")
        
        # Start scraping with smart filtering
        all_content = []
        skipped_count = 0
        
        try:
            # Process homepage first
            homepage_content = await self.scrape_page(base_url, company_name)
            if homepage_content:
                all_content.append(homepage_content)
            
            # Process other content links
            for i, link in enumerate(size_estimate["content_links"][:30]):  # Reduced limit for Render
                if not self.should_continue_scraping():
                    break
                
                full_url = urljoin(base_url, link)
                
                # Quick URL relevance check
                if not self.is_demo_relevant_url(full_url):
                    self.skipped_urls.add(full_url)
                    skipped_count += 1
                    continue
                
                # Scrape the page
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
            
            logger.info(f"✅ Smart scraping completed:")
            logger.info(f"   📄 Articles scraped: {total_articles}")
            logger.info(f"   ⏭️ URLs skipped: {skipped_count}")
            logger.info(f"   ⏱️ Time taken: {elapsed_time/60:.1f} minutes")
            logger.info(f"   🎯 Success rate: {total_articles/(total_articles+skipped_count)*100:.1f}%")
            
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
            logger.error(f"❌ Smart scraping failed: {e}")
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
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic content
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Untitled"
            
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
            
            if not content or len(content) < 50:
                return None
            
            # Analyze content relevance
            relevance_analysis = self.is_demo_relevant_content(content, title_text)
            
            # Only include highly relevant content
            if not relevance_analysis.get("is_demo_relevant", False):
                self.skipped_urls.add(url)
                return None
            
            self.scraped_urls.add(url)
            self.demo_relevant_urls.add(url)
            
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
