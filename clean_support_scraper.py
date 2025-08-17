#!/usr/bin/env python3
"""
Clean Support Bot Scraper
Removes author details, redirections, and incomplete content
Provides complete, clean answers for support bot
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

class CleanSupportScraper:
    """Clean support bot scraper with complete content extraction"""
    
    def __init__(self):
        """Initialize the clean support scraper"""
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        logger.info("🚀 Initialized Clean Support Scraper")
    
    def extract_clean_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract clean content without author details or redirections
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            
        Returns:
            Clean extracted content dictionary
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
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5']):
                heading_text = heading.get_text().strip()
                if heading_text and len(heading_text) > 3:
                    headings.append(heading_text)
            
            # Extract main content with clean formatting
            main_content_sections = []
            
            # Look for main content areas first
            content_selectors = [
                'main', 'article', '.content', '.main-content', '.post-content',
                '.entry-content', '.article-content', '.page-content', '.help-content',
                '.faq-content', '.support-content', '.documentation-content'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = self.extract_clean_text(element)
                    if text and len(text) > 50:
                        main_content_sections.append(text)
            
            # If no main content found, extract from body
            if not main_content_sections:
                body = soup.find('body')
                if body:
                    main_content_sections.append(self.extract_clean_text(body))
            
            # Combine content
            if main_content_sections:
                combined_content = '\n\n'.join(main_content_sections)
            else:
                combined_content = self.extract_clean_text(soup)
            
            # Clean and format the content
            cleaned_content = self.clean_and_structure_content(combined_content)
            
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
                'content': cleaned_content,  # Complete content
                'headings': headings[:10],
                'url': url,
                'word_count': word_count,
                'content_type': content_type,
                'collection': 'Support',
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
    
    def extract_clean_text(self, element) -> str:
        """Extract clean text without author details or redirections"""
        if not element:
            return ""
        
        # Get text with proper spacing
        text = element.get_text(separator=' ', strip=True)
        
        # Remove author details and redirections
        text = self.remove_author_details(text)
        text = self.remove_redirections(text)
        text = self.remove_incomplete_content(text)
        
        # Clean up formatting
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Add space after punctuation
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        
        return text.strip()
    
    def remove_author_details(self, text: str) -> str:
        """Remove author details from text"""
        # Remove patterns like "By Author Name and X others"
        text = re.sub(r'By\s+[A-Za-z\s]+\s+and\s+\d+\s+others?\s*\d*\s*authors?\s*\d*\s*articles?', '', text)
        text = re.sub(r'By\s+[A-Za-z\s]+\s+\d*\s*authors?\s*\d*\s*articles?', '', text)
        text = re.sub(r'By\s+[A-Za-z\s]+', '', text)
        
        # Remove article counts
        text = re.sub(r'\d+\s*authors?\s*\d*\s*articles?', '', text)
        text = re.sub(r'\d+\s*articles?', '', text)
        
        # Remove author names followed by numbers
        text = re.sub(r'[A-Za-z\s]+\s+\d+\s*author', '', text)
        
        return text
    
    def remove_redirections(self, text: str) -> str:
        """Remove redirection phrases"""
        redirection_patterns = [
            r'click\s+here\s+to\s+[a-zA-Z\s]+',
            r'visit\s+[a-zA-Z\s]+',
            r'go\s+to\s+[a-zA-Z\s]+',
            r'check\s+[a-zA-Z\s]+',
            r'see\s+[a-zA-Z\s]+',
            r'refer\s+to\s+[a-zA-Z\s]+',
            r'for\s+more\s+information',
            r'additional\s+details',
            r'learn\s+more',
            r'read\s+more'
        ]
        
        for pattern in redirection_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def remove_incomplete_content(self, text: str) -> str:
        """Remove incomplete content patterns"""
        # Remove content ending with "..."
        text = re.sub(r'\.{3,}', '', text)
        
        # Remove incomplete sentences
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                # Check if sentence is complete
                if not sentence.endswith('...') and not sentence.endswith('..'):
                    complete_sentences.append(sentence)
        
        return '. '.join(complete_sentences)
    
    def clean_and_structure_content(self, content: str) -> str:
        """Clean and structure content for better readability"""
        if not content:
            return ""
        
        # Split into sentences and clean each one
        sentences = re.split(r'[.!?]+', content)
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                # Clean up formatting issues
                sentence = re.sub(r'([a-z])([A-Z])', r'\1 \2', sentence)  # Fix camelCase
                sentence = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', sentence)  # Fix numbers followed by letters
                sentence = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', sentence)  # Fix letters followed by numbers
                sentence = re.sub(r'\s+', ' ', sentence)  # Fix multiple spaces
                
                # Capitalize first letter
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                
                # Only add if it's a complete, meaningful sentence
                if len(sentence) > 15 and not self.is_redirection_sentence(sentence):
                    cleaned_sentences.append(sentence)
        
        # Join sentences with proper punctuation
        formatted_content = '. '.join(cleaned_sentences)
        if formatted_content and not formatted_content.endswith('.'):
            formatted_content += '.'
        
        return formatted_content
    
    def is_redirection_sentence(self, sentence: str) -> bool:
        """Check if a sentence is a redirection"""
        redirection_words = [
            'click', 'visit', 'go', 'check', 'see', 'refer', 'learn', 'read',
            'more', 'additional', 'details', 'information'
        ]
        
        sentence_lower = sentence.lower()
        redirection_count = sum(1 for word in redirection_words if word in sentence_lower)
        
        # If sentence contains multiple redirection words, it's likely a redirection
        return redirection_count >= 2
    
    def extract_key_information(self, content: str, headings: List[str]) -> Dict[str, Any]:
        """Extract key topics and points from content"""
        topics = []
        points = []
        
        # Extract topics from headings
        for heading in headings[:5]:
            if heading and len(heading) > 3:
                topics.append(heading)
        
        # Extract key points from content
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 20 and len(sentence) < 300:
                # Look for sentences with key words
                key_words = ['important', 'note', 'key', 'essential', 'must', 'should', 'need', 'setup', 'configure', 'manage', 'how to', 'step', 'guide']
                if any(word in sentence.lower() for word in key_words):
                    points.append(sentence)
        
        # If no key points found, take first few meaningful sentences
        if not points:
            for sentence in sentences[:6]:
                sentence = sentence.strip()
                if sentence and len(sentence) > 20 and not self.is_redirection_sentence(sentence):
                    points.append(sentence)
        
        return {
            'topics': topics[:5],
            'points': points[:6]
        }
    
    def determine_content_type(self, content: str, headings: List[str]) -> str:
        """Determine the type of content"""
        content_lower = content.lower()
        headings_lower = [h.lower() for h in headings]
        
        # Check for FAQ
        if any('faq' in h for h in headings_lower) or 'frequently asked' in content_lower:
            return 'faq'
        
        # Check for tutorial/guide
        if any(word in content_lower for word in ['tutorial', 'guide', 'how to', 'step by step', 'setup', 'instructions']):
            return 'tutorial'
        
        # Check for documentation
        if any(word in content_lower for word in ['documentation', 'api', 'reference']):
            return 'documentation'
        
        # Check for help center
        if any(word in content_lower for word in ['help', 'support', 'assistance']):
            return 'help'
        
        # Check for article/blog
        if any(word in content_lower for word in ['article', 'blog', 'post']):
            return 'article'
        
        return 'support'
    
    def determine_difficulty_level(self, content: str, word_count: int) -> str:
        """Determine the difficulty level of the content"""
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
            '1.', '2.', '3.', 'step by step', 'instructions', 'setup', 'guide'
        ]
        return any(indicator in content_lower for indicator in step_indicators)
    
    def calculate_quality_score(self, content: str, word_count: int, headings: List[str]) -> int:
        """Calculate a quality score for the content"""
        score = 70  # Higher base score for clean content
        
        # Add points for word count
        if word_count > 100:
            score += 10
        if word_count > 300:
            score += 10
        if word_count > 500:
            score += 5
        
        # Add points for headings
        if headings:
            score += min(len(headings) * 5, 20)
        
        # Add points for content structure
        if self.has_step_by_step_instructions(content):
            score += 10
        
        # Add points for key information
        if any(word in content.lower() for word in ['important', 'note', 'key', 'setup', 'configure']):
            score += 5
        
        # Add points for proper formatting
        if '.' in content and len(content.split('.')) > 5:
            score += 5
        
        # Deduct points for redirections
        if any(word in content.lower() for word in ['click here', 'visit', 'go to', 'check']):
            score -= 10
        
        return max(min(score, 95), 60)  # Cap between 60-95
    
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
            
            return filtered_links[:12]  # Reasonable number of links
            
        except Exception as e:
            logger.error(f"❌ Failed to get links from {url}: {e}")
            return []
    
    async def scrape_single_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page with clean content extraction"""
        try:
            logger.info(f"🔍 Scraping: {url}")
            
            # Get page content
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract clean content
            extracted_content = self.extract_clean_content(response.text, url)
            
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
        """Scrape website comprehensively with clean content extraction"""
        try:
            logger.info(f"🚀 Starting clean scraping of: {base_url}")
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
                    
                    if content and content['word_count'] > 20:  # Only add if meaningful content
                        extracted_contents.append(content)
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to scrape {link}: {e}")
                    continue
            
            logger.info(f"✅ Clean scraping completed. Extracted {len(extracted_contents)} pages")
            
            return extracted_contents
            
        except Exception as e:
            logger.error(f"❌ Clean scraping failed: {e}")
            return []



