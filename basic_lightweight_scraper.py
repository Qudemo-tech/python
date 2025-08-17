#!/usr/bin/env python3
"""
Basic Lightweight Web Scraper with Intelligent Content Extraction
Uses basic NLP techniques and intelligent text processing
No API tokens required - works completely offline
"""

import os
import asyncio
import json
import re
import time
from typing import List, Dict, Optional, Any
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicLightweightScraper:
    """Basic lightweight web scraper with intelligent content extraction"""
    
    def __init__(self):
        """Initialize the basic lightweight scraper"""
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        logger.info("🚀 Initialized Basic Lightweight Scraper")
    
    def extract_content_intelligently(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract content intelligently using basic NLP techniques
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            
        Returns:
            Extracted content dictionary
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "link"]):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Untitled"
            
            # Extract headings
            headings = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                heading_text = heading.get_text().strip()
                if heading_text and len(heading_text) > 3:
                    headings.append(heading_text)
            
            # Extract main content areas
            main_content = []
            
            # Look for main content areas
            content_selectors = [
                'main', 'article', '.content', '.main-content', '.post-content',
                '.entry-content', '.article-content', '.page-content'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if text and len(text) > 50:
                        main_content.append(text)
            
            # If no main content found, get all paragraphs
            if not main_content:
                paragraphs = soup.find_all(['p', 'div'])
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 20 and not text.startswith('©') and not text.startswith('Privacy'):
                        main_content.append(text)
            
            # Combine and clean content
            if main_content:
                combined_content = ' '.join(main_content)
            else:
                combined_content = soup.get_text()
            
            # Clean text
            lines = (line.strip() for line in combined_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove excessive whitespace
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
            
            # Extract key information
            key_info = self.extract_key_information(cleaned_content, headings)
            
            # Word count
            word_count = len(cleaned_content.split())
            
            # Determine content type
            content_type = self.determine_content_type(cleaned_content, headings)
            
            # Determine difficulty level
            difficulty_level = self.determine_difficulty_level(cleaned_content, word_count)
            
            # Check for steps/instructions
            has_steps = self.has_step_by_step_instructions(cleaned_content)
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(cleaned_content, word_count, headings)
            
            return {
                'title': title_text,
                'content': cleaned_content[:2000],  # Limit content length
                'headings': headings[:5],
                'url': url,
                'word_count': word_count,
                'content_type': content_type,
                'collection': 'General',
                'quality_score': quality_score,
                'difficulty_level': difficulty_level,
                'has_steps': has_steps,
                'is_complete': word_count > 50,
                'key_topics': key_info['topics'],
                'key_points': key_info['points']
            }
            
        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            return {
                'title': 'Error',
                'content': f'Failed to extract content: {str(e)}',
                'url': url,
                'word_count': 0,
                'content_type': 'error',
                'collection': 'Error',
                'quality_score': 0,
                'difficulty_level': 'unknown',
                'has_steps': False,
                'is_complete': False,
                'key_topics': [],
                'key_points': []
            }
    
    def extract_key_information(self, content: str, headings: List[str]) -> Dict[str, Any]:
        """Extract key topics and points from content"""
        topics = []
        points = []
        
        # Extract topics from headings
        for heading in headings[:3]:
            if heading and len(heading) > 3:
                topics.append(heading)
        
        # Extract key points from content
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 20 and len(sentence) < 200:
                # Look for sentences with key words
                key_words = ['important', 'note', 'key', 'essential', 'must', 'should', 'need']
                if any(word in sentence.lower() for word in key_words):
                    points.append(sentence)
        
        # If no key points found, take first few meaningful sentences
        if not points:
            for sentence in sentences[:5]:
                sentence = sentence.strip()
                if sentence and len(sentence) > 20:
                    points.append(sentence)
        
        return {
            'topics': topics[:3],
            'points': points[:5]
        }
    
    def determine_content_type(self, content: str, headings: List[str]) -> str:
        """Determine the type of content"""
        content_lower = content.lower()
        headings_lower = [h.lower() for h in headings]
        
        # Check for FAQ
        if any('faq' in h for h in headings_lower) or 'frequently asked' in content_lower:
            return 'faq'
        
        # Check for tutorial/guide
        if any(word in content_lower for word in ['tutorial', 'guide', 'how to', 'step by step']):
            return 'tutorial'
        
        # Check for documentation
        if any(word in content_lower for word in ['documentation', 'api', 'reference']):
            return 'documentation'
        
        # Check for article/blog
        if any(word in content_lower for word in ['article', 'blog', 'post']):
            return 'article'
        
        return 'general'
    
    def determine_difficulty_level(self, content: str, word_count: int) -> str:
        """Determine the difficulty level of the content"""
        # Simple heuristic based on word count and complexity
        if word_count < 100:
            return 'beginner'
        elif word_count < 500:
            return 'intermediate'
        else:
            return 'advanced'
    
    def has_step_by_step_instructions(self, content: str) -> bool:
        """Check if content has step-by-step instructions"""
        content_lower = content.lower()
        step_indicators = [
            'step 1', 'step 2', 'step 3', 'first', 'second', 'third',
            '1.', '2.', '3.', 'step by step', 'instructions'
        ]
        return any(indicator in content_lower for indicator in step_indicators)
    
    def calculate_quality_score(self, content: str, word_count: int, headings: List[str]) -> int:
        """Calculate a quality score for the content"""
        score = 60  # Base score
        
        # Add points for word count
        if word_count > 100:
            score += 10
        if word_count > 300:
            score += 10
        
        # Add points for headings
        if headings:
            score += min(len(headings) * 5, 20)
        
        # Add points for content structure
        if self.has_step_by_step_instructions(content):
            score += 10
        
        # Add points for key information
        if any(word in content.lower() for word in ['important', 'note', 'key']):
            score += 5
        
        return min(score, 95)  # Cap at 95
    
    async def get_page_links(self, url: str) -> List[str]:
        """Get all links from a page using requests"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get all links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href and not href.startswith(('javascript:', 'mailto:', '#')):
                    # Make relative URLs absolute
                    absolute_url = urljoin(url, href)
                    links.append(absolute_url)
            
            # Filter links to same domain
            base_domain = urlparse(url).netloc
            filtered_links = []
            
            for link in links:
                try:
                    link_domain = urlparse(link).netloc
                    if link_domain == base_domain and link not in filtered_links:
                        filtered_links.append(link)
                except:
                    continue
            
            return filtered_links[:10]  # Limit to 10 links
            
        except Exception as e:
            logger.error(f"❌ Failed to get links from {url}: {e}")
            return []
    
    async def scrape_single_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page intelligently"""
        try:
            logger.info(f"🔍 Scraping: {url}")
            
            # Get page content
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract content intelligently
            extracted_content = self.extract_content_intelligently(response.text, url)
            
            logger.info(f"✅ Extracted {extracted_content['word_count']} words from {url}")
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape {url}: {e}")
            return {
                'title': 'Error',
                'content': f'Failed to scrape: {str(e)}',
                'url': url,
                'word_count': 0,
                'content_type': 'error',
                'collection': 'Error',
                'quality_score': 0,
                'difficulty_level': 'unknown',
                'has_steps': False,
                'is_complete': False,
                'key_topics': [],
                'key_points': []
            }
    
    async def scrape_website_comprehensive(self, base_url: str, max_pages: int = 3) -> List[Dict[str, Any]]:
        """Scrape website comprehensively with intelligent content extraction"""
        try:
            logger.info(f"🚀 Starting comprehensive scraping of: {base_url}")
            logger.info(f"📄 Max pages: {max_pages}")
            
            # Get initial page links
            initial_links = await self.get_page_links(base_url)
            
            # Add base URL to links
            all_links = [base_url] + initial_links
            unique_links = list(dict.fromkeys(all_links))  # Remove duplicates
            
            # Limit to max_pages
            links_to_scrape = unique_links[:max_pages]
            
            logger.info(f"🔗 Found {len(links_to_scrape)} pages to scrape")
            
            # Scrape each page
            extracted_contents = []
            
            for i, link in enumerate(links_to_scrape, 1):
                try:
                    logger.info(f"📄 Scraping page {i}/{len(links_to_scrape)}: {link}")
                    
                    content = await self.scrape_single_page(link)
                    
                    if content and content['word_count'] > 10:  # Only add if meaningful content
                        extracted_contents.append(content)
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to scrape {link}: {e}")
                    continue
            
            logger.info(f"✅ Scraping completed. Extracted {len(extracted_contents)} pages")
            
            return extracted_contents
            
        except Exception as e:
            logger.error(f"❌ Comprehensive scraping failed: {e}")
            return []



