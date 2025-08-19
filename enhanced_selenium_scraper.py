#!/usr/bin/env python3
"""
Enhanced Selenium scraper for dynamic websites like Settle's help center
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import openai
import os

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSeleniumScraper:
    def __init__(self, openai_api_key: str = None):
        self.openai_client = None
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        self.scraped_urls = set()
        self.start_time = None
        self.max_time_per_site = 1800  # 30 minutes
        self.max_articles_per_site = 100
        
        logger.info("Enhanced Selenium Scraper Initialized")
    
    def setup_driver(self):
        """Setup headless Chrome driver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            return None
    
    def should_continue_scraping(self) -> bool:
        """Check if we should continue scraping"""
        if not self.start_time:
            return True
        
        elapsed_time = time.time() - self.start_time
        articles_scraped = len(self.scraped_urls)
        
        if elapsed_time > self.max_time_per_site:
            logger.info(f"Time limit reached: {elapsed_time/60:.1f} minutes")
            return False
        
        if articles_scraped >= self.max_articles_per_site:
            logger.info(f"Article limit reached: {articles_scraped} articles")
            return False
        
        return True
    
    def get_collection_links(self, driver, base_url: str) -> List[tuple]:
        """Get all collection links from the main page"""
        collection_links = []
        
        try:
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content
            time.sleep(3)
            
            # Find collection links
            links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/collections/']")
            
            for link in links:
                href = link.get_attribute('href')
                text = link.text.strip()
                
                if href and text and href not in [url[0] for url in collection_links]:
                    # Extract article count from text
                    article_count = 0
                    if 'articles' in text.lower():
                        try:
                            # Look for number before "articles"
                            import re
                            match = re.search(r'(\d+)\s*articles', text.lower())
                            if match:
                                article_count = int(match.group(1))
                        except:
                            pass
                    
                    collection_links.append((href, text, article_count))
                    logger.info(f"Found collection: {text} ({article_count} articles) -> {href}")
            
            return collection_links
            
        except Exception as e:
            logger.error(f"Failed to get collection links: {e}")
            return []
    
    def get_article_links_from_collection(self, driver, collection_url: str) -> List[tuple]:
        """Get article links from a collection page"""
        article_links = []
        
        try:
            logger.info(f"Loading collection page: {collection_url}")
            driver.get(collection_url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content
            time.sleep(3)
            
            # Try different selectors for article links
            selectors = [
                "a[href*='/articles/']",
                "a[href*='/help/']",
                ".article-link",
                ".help-article",
                "[data-testid*='article']",
                "[class*='article']",
                "a[href*='/en/']"  # General Settle links
            ]
            
            for selector in selectors:
                try:
                    links = driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for link in links:
                        href = link.get_attribute('href')
                        text = link.text.strip()
                        
                        if (href and text and 
                            href not in [url[0] for url in article_links] and
                            '/collections/' not in href and  # Skip collection links
                            href != collection_url):  # Skip self-reference
                            
                            article_links.append((href, text))
                            logger.info(f"Found article: {text} -> {href}")
                            
                except Exception as e:
                    logger.warning(f"Failed with selector {selector}: {e}")
            
            return article_links
            
        except Exception as e:
            logger.error(f"Failed to get article links from collection: {e}")
            return []
    
    def extract_article_content(self, driver, article_url: str, title: str) -> Dict[str, Any]:
        """Extract content from an article page"""
        try:
            logger.info(f"Extracting content from: {title}")
            driver.get(article_url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content
            time.sleep(2)
            
            # Remove unwanted elements
            unwanted_selectors = [
                "script", "style", "nav", "footer", "header", "aside",
                ".navigation", ".sidebar", ".menu", ".footer", ".header"
            ]
            
            for selector in unwanted_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        driver.execute_script("arguments[0].remove();", element)
                except:
                    continue
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '.main-content', '.post-content',
                '.entry-content', '.article-content', '.page-content', '.help-content',
                '.article-body', '.post-body', '.content-body', '.article'
            ]
            
            content = ""
            for selector in content_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        text = element.text.strip()
                        if text and len(text) > 50:
                            content += text + "\n\n"
                except:
                    continue
            
            if not content:
                # Fallback to body content
                try:
                    body = driver.find_element(By.TAG_NAME, "body")
                    content = body.text.strip()
                except:
                    content = ""
            
            if content and len(content) > 200:
                return {
                    "title": title,
                    "content": content,
                    "url": article_url,
                    "word_count": len(content.split()),
                    "success": True
                }
            else:
                logger.warning(f"Insufficient content from {article_url}")
                return {"success": False, "error": "Insufficient content"}
                
        except Exception as e:
            logger.error(f"Failed to extract content from {article_url}: {e}")
            return {"success": False, "error": str(e)}
    
    async def scrape_settle_help_center(self, base_url: str, company_name: str) -> Dict[str, Any]:
        """Scrape Settle's help center using Selenium"""
        self.start_time = time.time()
        self.scraped_urls.clear()
        
        logger.info(f"Starting enhanced Selenium scraping for: {base_url}")
        
        driver = self.setup_driver()
        if not driver:
            return {"success": False, "error": "Failed to setup browser driver"}
        
        all_articles = []
        
        try:
            # Step 1: Load main page and get collection links
            logger.info("Step 1: Loading main page and finding collections...")
            driver.get(base_url)
            
            collection_links = self.get_collection_links(driver, base_url)
            logger.info(f"Found {len(collection_links)} collections")
            
            if not collection_links:
                return {"success": False, "error": "No collections found"}
            
            # Step 2: Process each collection
            for i, (collection_url, collection_name, article_count) in enumerate(collection_links):
                if not self.should_continue_scraping():
                    break
                
                logger.info(f"Processing collection {i+1}/{len(collection_links)}: {collection_name}")
                
                # Get article links from this collection
                article_links = self.get_article_links_from_collection(driver, collection_url)
                logger.info(f"Found {len(article_links)} articles in collection: {collection_name}")
                
                # Step 3: Extract content from each article
                for j, (article_url, article_title) in enumerate(article_links):
                    if not self.should_continue_scraping():
                        break
                    
                    if article_url in self.scraped_urls:
                        continue
                    
                    logger.info(f"Scraping article {j+1}/{len(article_links)}: {article_title}")
                    
                    # Extract article content
                    article_data = self.extract_article_content(driver, article_url, article_title)
                    
                    if article_data.get("success"):
                        self.scraped_urls.add(article_url)
                        
                        # Add collection info
                        article_data["collection_name"] = collection_name
                        article_data["company_name"] = company_name
                        
                        all_articles.append(article_data)
                        logger.info(f"Successfully scraped: {article_title} ({article_data['word_count']} words)")
                    
                    # Small delay between articles
                    time.sleep(1)
            
            # Calculate final metrics
            elapsed_time = time.time() - self.start_time
            total_articles = len(all_articles)
            
            logger.info(f"Enhanced Selenium scraping completed:")
            logger.info(f"    Articles scraped: {total_articles}")
            logger.info(f"    Collections processed: {len(collection_links)}")
            logger.info(f"    Time taken: {elapsed_time/60:.1f} minutes")
            
            return {
                "success": True,
                "data": {
                    "chunks": all_articles,
                    "summary": {
                        "total_items": total_articles,
                        "collections_processed": len(collection_links),
                        "scraped_urls": len(self.scraped_urls),
                        "elapsed_time_minutes": elapsed_time / 60,
                        "website_size": "dynamic"
                    }
                },
                "company_name": company_name,
                "website_url": base_url
            }
            
        except Exception as e:
            logger.error(f"Enhanced Selenium scraping failed: {e}")
            return {"success": False, "error": str(e)}
        
        finally:
            driver.quit()

if __name__ == "__main__":
    # Test the enhanced Selenium scraper
    scraper = EnhancedSeleniumScraper()
    result = asyncio.run(scraper.scrape_settle_help_center("https://help.settle.com/en", "seattle"))
    print(json.dumps(result, indent=2))

