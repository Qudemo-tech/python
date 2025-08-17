#!/usr/bin/env python3
"""
Enhanced Salesforce & Authentication Scraper
Handles Salesforce, authentication-required sites, and complex web applications
Multiple access strategies for different website types
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSalesforceScraper:
    def __init__(self):
        """Initialize enhanced scraper with multiple access strategies"""
        # Standard requests session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Selenium driver for JavaScript-heavy sites
        self.driver = None
        self.setup_selenium()
        
        logger.info("🚀 Enhanced Salesforce Scraper Initialized")
    
    def setup_selenium(self):
        """Setup Selenium WebDriver for JavaScript-heavy sites"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Disable images and CSS for faster loading
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("✅ Selenium WebDriver initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Selenium setup failed: {e}")
            self.driver = None
    
    def detect_site_type(self, url: str) -> str:
        """Detect the type of website to determine access strategy"""
        url_lower = url.lower()
        
        # Salesforce detection
        if any(domain in url_lower for domain in ['salesforce.com', 'force.com', 'my.salesforce.com', '.site.com']):
            return 'salesforce'
        
        # Other authentication-required sites
        if any(domain in url_lower for domain in ['sharepoint.com', 'office365.com', 'teams.microsoft.com']):
            return 'microsoft'
        
        # JavaScript-heavy sites
        if any(domain in url_lower for domain in ['react', 'vue', 'angular', 'spa']):
            return 'javascript_heavy'
        
        # Standard sites
        return 'standard'
    
    async def extract_with_requests(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content using standard requests"""
        try:
            logger.info(f"🔍 Trying requests extraction: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if we got a login page or error
            if self.is_login_page(response.text):
                logger.warning(f"⚠️ Login page detected for {url}")
                return None
            
            # Extract content
            extracted_content = self.extract_perfect_content(response.text, url)
            
            if extracted_content and extracted_content['word_count'] > 10:
                logger.info(f"✅ Requests extraction successful: {extracted_content['word_count']} words")
                return extracted_content
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Requests extraction failed: {e}")
            return None
    
    async def extract_with_selenium(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content using Selenium for JavaScript-heavy sites"""
        if not self.driver:
            logger.warning("⚠️ Selenium driver not available")
            return None
        
        try:
            logger.info(f"🔍 Trying Selenium extraction: {url}")
            
            # Navigate to page
            self.driver.get(url)
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Check for login page
            if self.is_login_page_selenium():
                logger.warning(f"⚠️ Login page detected for {url}")
                return None
            
            # Wait for content to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.warning("⚠️ Page load timeout")
            
            # Get page source
            page_source = self.driver.page_source
            
            # Extract content
            extracted_content = self.extract_perfect_content(page_source, url)
            
            if extracted_content and extracted_content['word_count'] > 10:
                logger.info(f"✅ Selenium extraction successful: {extracted_content['word_count']} words")
                return extracted_content
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Selenium extraction failed: {e}")
            return None
    
    def is_login_page(self, html_content: str) -> bool:
        """Check if the page is a login page"""
        login_indicators = [
            'login', 'sign in', 'signin', 'authentication', 'auth',
            'username', 'password', 'email', 'credentials',
            'access denied', 'unauthorized', 'forbidden',
            'please log in', 'please sign in', 'login required'
        ]
        
        content_lower = html_content.lower()
        login_count = sum(1 for indicator in login_indicators if indicator in content_lower)
        
        return login_count >= 3
    
    def is_login_page_selenium(self) -> bool:
        """Check if current page is a login page using Selenium"""
        try:
            # Look for login-related elements
            login_elements = self.driver.find_elements(By.XPATH, 
                "//*[contains(text(), 'login') or contains(text(), 'sign in') or contains(text(), 'username') or contains(text(), 'password')]"
            )
            
            return len(login_elements) > 2
            
        except Exception:
            return False
    
    def extract_perfect_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract content with enhanced selectors for all website types"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "link"]):
                element.decompose()
            
            # Extract title and headings
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Untitled"
            
            headings = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5']):
                heading_text = heading.get_text().strip()
                if heading_text and len(heading_text) > 2:
                    headings.append(heading_text)
            
            # Enhanced content selectors for different website types
            content_selectors = [
                # General content areas
                'main', 'article', '.content', '.main-content', '.post-content',
                '.entry-content', '.article-content', '.page-content', '.help-content',
                '.faq-content', '.support-content', '.documentation-content',
                
                # FAQ specific selectors
                '.faq', '.faq-item', '.faq-question', '.faq-answer', '.accordion',
                '.faq-section', '.faq-container', '.faq-list',
                
                # Salesforce specific selectors
                '.slds-card', '.slds-card__body', '.slds-card__header',
                '.slds-tabs__content', '.slds-tabs__panel',
                '.slds-accordion__content', '.slds-accordion__section',
                '.slds-page-header', '.slds-page-header__content',
                '.slds-form', '.slds-form-element',
                '.slds-text-heading', '.slds-text-body',
                '.slds-panel', '.slds-panel__body',
                '.slds-modal', '.slds-modal__content',
                
                # Knowledge base selectors
                '.knowledge-base', '.kb-article', '.kb-content',
                '.help-article', '.help-section', '.help-docs',
                '.documentation', '.docs-content', '.api-docs',
                
                # JavaScript-heavy site selectors
                '.js-content', '.dynamic-content', '.ajax-content',
                '.react-content', '.vue-content', '.angular-content',
                '[data-reactroot]', '[data-vue]', '[ng-app]',
                
                # Microsoft/SharePoint selectors
                '.ms-', '.sp-', '.sharepoint-', '.office365-',
                '.ms-webpart', '.sp-webpart', '.ms-list',
                
                # Generic content selectors
                '.text-content', '.article-text', '.content-text',
                '.main-text', '.body-text', '.page-text'
            ]
            
            main_content_sections = []
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = self.clean_text_perfectly(element)
                    if text and len(text) > 30:
                        main_content_sections.append(text)
            
            # If no main content found, extract from body
            if not main_content_sections:
                body = soup.find('body')
                if body:
                    main_content_sections.append(self.clean_text_perfectly(body))
            
            # Combine and clean content - NO TRUNCATION
            combined_content = '\n\n'.join(main_content_sections)
            cleaned_content = self.structure_content_perfectly(combined_content)
            
            # Extract key information
            key_info = self.extract_key_info(cleaned_content, headings)
            word_count = len(cleaned_content.split())
            
            # Determine content type and quality
            content_type = self.determine_content_type(cleaned_content, headings, url)
            difficulty_level = 'beginner' if word_count < 100 else 'intermediate' if word_count < 300 else 'advanced'
            has_steps = any(word in cleaned_content.lower() for word in ['step', 'guide', 'tutorial', 'setup', 'how to'])
            quality_score = self.calculate_perfect_score(cleaned_content, word_count, headings, url)
            
            return {
                'title': title_text,
                'content': cleaned_content,  # Full content, no truncation
                'headings': headings[:10],
                'url': url,
                'word_count': word_count,
                'content_type': content_type,
                'collection': 'Enhanced Support',
                'quality_score': quality_score,
                'difficulty_level': difficulty_level,
                'has_steps': has_steps,
                'is_complete': word_count > 30,
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
    
    def clean_text_perfectly(self, element) -> str:
        """Clean text to perfection with enhanced patterns - NO TRUNCATION"""
        if not element:
            return ""
        
        text = element.get_text(separator=' ', strip=True)
        
        # Remove author details
        text = re.sub(r'By\s+[A-Za-z\s]+\s+and\s+\d+\s+others?\s*\d*\s*authors?\s*\d*\s*articles?', '', text)
        text = re.sub(r'By\s+[A-Za-z\s]+\s+\d*\s*authors?\s*\d*\s*articles?', '', text)
        text = re.sub(r'By\s+[A-Za-z\s]+', '', text)
        text = re.sub(r'\d+\s*authors?\s*\d*\s*articles?', '', text)
        text = re.sub(r'\d+\s*articles?', '', text)
        text = re.sub(r'[A-Za-z\s]+\s+\d+\s*author', '', text)
        text = re.sub(r'[A-Za-z\s]+avatar', '', text)
        
        # Remove date information
        text = re.sub(r'\d+\s*Views?\s*•\s*\w+\s+\d+,\s*\d+\s*•\s*\w+', '', text)
        text = re.sub(r'\d+\s*Views?\s*•\s*\w+\s+\d+,\s*\d+', '', text)
        text = re.sub(r'\w+\s+\d+,\s*\d+\s*•\s*\w+', '', text)
        text = re.sub(r'\d+\s*Views?', '', text)
        text = re.sub(r'\d+\s*•\s*\w+\s+\d+,\s*\d+', '', text)
        text = re.sub(r'\w+\s+\d+,\s*\d+', '', text)
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
        text = re.sub(r'\d{1,2}-\d{1,2}-\d{2,4}', '', text)
        text = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', '', text)
        
        # Remove redirections
        redirection_patterns = [
            r'click\s+here\s+to\s+[a-zA-Z\s]+', r'visit\s+[a-zA-Z\s]+', r'go\s+to\s+[a-zA-Z\s]+',
            r'check\s+[a-zA-Z\s]+', r'see\s+[a-zA-Z\s]+', r'refer\s+to\s+[a-zA-Z\s]+',
            r'for\s+more\s+information', r'additional\s+details', r'learn\s+more',
            r'read\s+more', r'get\s+in\s+touch', r'contact\s+us', r'get\s+started',
            r'book\s+a\s+demo', r'request\s+access', r'get\s+help', r'find\s+out\s+more'
        ]
        
        for pattern in redirection_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove incomplete content indicators but keep actual content
        text = re.sub(r'\.{3,}$', '', text)  # Only remove trailing ...
        text = re.sub(r'…$', '', text)  # Remove trailing ellipsis
        
        # Fix formatting
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)
        
        return text.strip()
    
    def structure_content_perfectly(self, content: str) -> str:
        """Structure content perfectly - NO TRUNCATION"""
        if not content:
            return ""
        
        sentences = re.split(r'[.!?]+', content)
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                # Clean up formatting
                sentence = re.sub(r'([a-z])([A-Z])', r'\1 \2', sentence)
                sentence = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', sentence)
                sentence = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', sentence)
                sentence = re.sub(r'\s+', ' ', sentence)
                
                # Capitalize first letter
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                
                # Check if not a redirection sentence
                redirection_words = ['click', 'visit', 'go', 'check', 'see', 'refer', 'learn', 'read', 'more', 'additional', 'details', 'information', 'contact', 'get', 'book', 'request', 'find', 'start', 'demo', 'help', 'support']
                redirection_count = sum(1 for word in redirection_words if word in sentence.lower())
                
                if len(sentence) > 15 and redirection_count < 2:
                    cleaned_sentences.append(sentence)
        
        formatted_content = '. '.join(cleaned_sentences)
        if formatted_content and not formatted_content.endswith('.'):
            formatted_content += '.'
        
        # Ensure no truncation
        return formatted_content
    
    def extract_key_info(self, content: str, headings: List[str]) -> Dict[str, Any]:
        """Extract key information"""
        topics = [h for h in headings[:5] if h and len(h) > 2]
        
        sentences = re.split(r'[.!?]+', content)
        points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 15 and len(sentence) < 400:
                key_words = ['important', 'note', 'key', 'essential', 'must', 'should', 'need', 'setup', 'configure', 'manage', 'how to', 'step', 'guide', 'tutorial', 'example', 'tip', 'best practice', 'recommendation', 'requirement']
                if any(word in sentence.lower() for word in key_words):
                    points.append(sentence)
        
        if not points:
            for sentence in sentences[:8]:
                sentence = sentence.strip()
                if sentence and len(sentence) > 15:
                    points.append(sentence)
        
        return {'topics': topics[:5], 'points': points[:8]}
    
    def determine_content_type(self, content: str, headings: List[str], url: str) -> str:
        """Determine content type"""
        content_lower = content.lower()
        headings_lower = [h.lower() for h in headings]
        url_lower = url.lower()
        
        if any('faq' in h for h in headings_lower) or 'frequently asked' in content_lower or 'faq' in url_lower:
            return 'faq'
        elif any(word in content_lower for word in ['tutorial', 'guide', 'how to', 'step by step', 'setup', 'instructions', 'onboarding']):
            return 'tutorial'
        elif any(word in content_lower for word in ['documentation', 'api', 'reference', 'docs']):
            return 'documentation'
        elif any(word in content_lower for word in ['help', 'support', 'assistance', 'knowledge']):
            return 'help'
        elif any(word in content_lower for word in ['legal', 'privacy', 'terms', 'policy']):
            return 'legal'
        elif any(word in content_lower for word in ['article', 'blog', 'post']):
            return 'article'
        else:
            return 'support'
    
    def calculate_perfect_score(self, content: str, word_count: int, headings: List[str], url: str) -> int:
        """Calculate perfect quality score"""
        score = 85  # High base score
        
        # Word count points
        if word_count > 50: score += 5
        if word_count > 100: score += 5
        if word_count > 200: score += 5
        
        # Heading points
        if headings: score += min(len(headings) * 3, 15)
        
        # Content structure points
        if any(word in content.lower() for word in ['step', 'guide', 'tutorial', 'setup', 'how to']):
            score += 10
        
        # Key information points
        if any(word in content.lower() for word in ['important', 'note', 'key', 'setup', 'configure', 'example']):
            score += 5
        
        # Formatting points
        if '.' in content and len(content.split('.')) > 3: score += 5
        
        # Content type points
        content_type = self.determine_content_type(content, headings, url)
        if content_type in ['faq', 'tutorial', 'help']: score += 5
        
        # Deduct for redirections
        if any(word in content.lower() for word in ['click here', 'visit', 'go to', 'check']):
            score -= 5
        
        return max(min(score, 100), 75)
    
    async def get_page_links(self, url: str) -> List[str]:
        """Get page links with enhanced detection"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href and not href.startswith(('javascript:', 'mailto:', '#', 'tel:')):
                    absolute_url = urljoin(url, href)
                    links.append(absolute_url)
            
            base_domain = urlparse(url).netloc
            filtered_links = []
            
            for link in links:
                try:
                    link_domain = urlparse(link).netloc
                    if link_domain == base_domain and link not in filtered_links:
                        filtered_links.append(link)
                except:
                    continue
            
            return filtered_links[:15]
            
        except Exception as e:
            logger.error(f"❌ Failed to get links: {e}")
            return []
    
    async def scrape_single_page_enhanced(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape single page with multiple strategies"""
        site_type = self.detect_site_type(url)
        logger.info(f"🌐 Detected site type: {site_type}")
        
        # Try different extraction strategies based on site type
        if site_type == 'salesforce':
            # Try Selenium first for Salesforce sites
            result = await self.extract_with_selenium(url)
            if result:
                return result
            
            # Fallback to requests
            result = await self.extract_with_requests(url)
            return result
        
        elif site_type == 'javascript_heavy':
            # Try Selenium for JavaScript-heavy sites
            result = await self.extract_with_selenium(url)
            if result:
                return result
            
            # Fallback to requests
            result = await self.extract_with_requests(url)
            return result
        
        else:
            # Standard sites - try requests first
            result = await self.extract_with_requests(url)
            if result:
                return result
            
            # Fallback to Selenium
            result = await self.extract_with_selenium(url)
            return result
    
    async def scrape_website_comprehensive(self, base_url: str, max_pages: int = 3) -> List[Dict[str, Any]]:
        """Scrape website comprehensively with enhanced strategies"""
        try:
            logger.info(f"🚀 Starting enhanced scraping: {base_url}")
            
            # Try to scrape the base URL first
            base_content = await self.scrape_single_page_enhanced(base_url)
            
            extracted_contents = []
            if base_content and base_content['word_count'] > 15:
                extracted_contents.append(base_content)
            
            # Try to get additional links
            try:
                initial_links = await self.get_page_links(base_url)
                all_links = initial_links
                unique_links = list(dict.fromkeys(all_links))
                links_to_scrape = unique_links[:max_pages-1]  # -1 because we already scraped base
                
                logger.info(f"🔗 Found {len(links_to_scrape)} additional pages")
                
                for i, link in enumerate(links_to_scrape, 1):
                    try:
                        logger.info(f"📄 Scraping additional page {i}/{len(links_to_scrape)}")
                        
                        content = await self.scrape_single_page_enhanced(link)
                        
                        if content and content['word_count'] > 15:
                            extracted_contents.append(content)
                        
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to scrape {link}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not get additional links: {e}")
            
            logger.info(f"✅ Enhanced scraping completed: {len(extracted_contents)} pages")
            return extracted_contents
            
        except Exception as e:
            logger.error(f"❌ Enhanced scraping failed: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("✅ Selenium driver closed")
            except Exception as e:
                logger.error(f"❌ Error closing Selenium driver: {e}")


