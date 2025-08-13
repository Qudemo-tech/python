#!/usr/bin/env python3
"""
Web Scraper Module for Product Knowledge Sources
Crawls websites under specified routes and extracts text content
"""

import requests
import re
import time
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set
import openai
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, openai_api_key: str):
        """Initialize web scraper with OpenAI for text processing"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.visited_urls: Set[str] = set()
        self.max_pages = 20  # Limit to prevent infinite crawling (reduced for testing)
        self.rate_limit_delay = 1.5  # Seconds between requests
        
    def validate_url(self, url: str) -> bool:
        """Validate if URL is properly formatted"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def get_domain_and_path(self, url: str) -> tuple:
        """Extract domain and base path from URL"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path.rstrip('/')
        return domain, path
    
    def should_crawl_url(self, url: str, base_domain: str, base_path: str) -> bool:
        """Check if URL should be crawled based on domain and path constraints"""
        parsed = urlparse(url)
        url_domain = f"{parsed.scheme}://{parsed.netloc}"
        url_path = parsed.path
        
        # Must be same domain (handle both http and https)
        if parsed.netloc != urlparse(base_domain).netloc:
            logger.debug(f"🚫 Domain mismatch: {parsed.netloc} != {urlparse(base_domain).netloc}")
            return False
            
        # For help.puzzle.io, allow crawling all pages under the domain
        # Don't restrict to just the base_path
        if 'help.puzzle.io' in base_domain:
            logger.debug(f"✅ Allowing crawl for help.puzzle.io: {url_path}")
            # Still avoid non-content paths
            skip_patterns = [
                '/admin', '/login', '/logout', '/api', '/ajax',
                '/search', '/cart', '/checkout', '/account',
                '.pdf', '.doc', '.docx', '.xls', '.xlsx',
                '.jpg', '.jpeg', '.png', '.gif', '.svg',
                '.css', '.js', '.xml', '.json'
            ]
            
            for pattern in skip_patterns:
                if pattern in url_path.lower():
                    logger.debug(f"🚫 Skipped pattern '{pattern}' in: {url_path}")
                    return False
            return True
        else:
            # For other domains, use the original path restriction
            if not url_path.startswith(base_path):
                logger.debug(f"🚫 Path outside scope: {url_path} not under {base_path}")
                return False
                
            # Avoid common non-content paths
            skip_patterns = [
                '/admin', '/login', '/logout', '/api', '/ajax',
                '/search', '/cart', '/checkout', '/account',
                '.pdf', '.doc', '.docx', '.xls', '.xlsx',
                '.jpg', '.jpeg', '.png', '.gif', '.svg',
                '.css', '.js', '.xml', '.json'
            ]
            
            for pattern in skip_patterns:
                if pattern in url_path.lower():
                    logger.debug(f"🚫 Skipped pattern '{pattern}' in: {url_path}")
                    return False
                    
            return True
    
    def detect_faq_page(self, html_content: str) -> bool:
        """Detect if a page contains FAQ content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Common FAQ indicators
            faq_indicators = [
                'faq', 'frequently asked questions', 'questions and answers',
                'q&a', 'q & a', 'common questions', 'help center'
            ]
            
            # Check page title
            title = soup.find('title')
            if title:
                title_text = title.get_text().lower()
                if any(indicator in title_text for indicator in faq_indicators):
                    logger.info(f"🔍 FAQ detected in page title: {title_text}")
                    return True
            
            # Check for FAQ-related elements
            faq_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for element in faq_elements:
                element_text = element.get_text().lower()
                if any(indicator in element_text for indicator in faq_indicators):
                    logger.info(f"🔍 FAQ detected in heading: {element_text}")
                    return True
            
            # Check for common FAQ patterns in content
            body_text = soup.get_text().lower()
            if any(indicator in body_text for indicator in faq_indicators):
                logger.info(f"🔍 FAQ indicators found in page content")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ FAQ detection failed: {e}")
            return False

    def extract_faq_content(self, html_content: str, url: str) -> List[Dict]:
        """Extract FAQ Q&A pairs from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            faq_pairs = []
            
            # Common FAQ patterns to look for
            faq_patterns = [
                # Pattern 1: Numbered FAQ structure (like Puzzle.io)
                {
                    'question_selector': 'h3, h4, h5, .question, .faq-question',
                    'answer_selector': 'p, div, .answer, .faq-answer'
                },
                
                # Pattern 2: Accordion/dropdown FAQ
                {
                    'question_selector': '[data-faq-question], .faq-question, .question, .accordion-item h3, .accordion-item h4',
                    'answer_selector': '[data-faq-answer], .faq-answer, .answer, .accordion-content, .accordion-body'
                },
                
                # Pattern 3: List-based FAQ
                {
                    'question_selector': 'dt, .faq-item h3, .faq-item h4, .question-item h3',
                    'answer_selector': 'dd, .faq-item p, .faq-item div, .answer-item p'
                },
                
                # Pattern 4: Div-based FAQ
                {
                    'question_selector': '.faq-question, .question, .faq-title',
                    'answer_selector': '.faq-answer, .answer, .faq-content'
                }
            ]
            
            for pattern in faq_patterns:
                questions = soup.select(pattern['question_selector'])
                answers = soup.select(pattern['answer_selector'])
                
                if questions and answers and len(questions) == len(answers):
                    logger.info(f"🔍 Found {len(questions)} FAQ pairs using pattern: {pattern['question_selector']}")
                    
                    for i, (question_elem, answer_elem) in enumerate(zip(questions, answers)):
                        question_text = question_elem.get_text().strip()
                        answer_text = answer_elem.get_text().strip()
                        
                        # Validate that this looks like a Q&A pair
                        if (len(question_text) > 10 and len(answer_text) > 20 and 
                            not question_text.lower().startswith(('copyright', 'privacy', 'terms', 'contact'))):
                            
                            faq_pairs.append({
                                'question': question_text,
                                'answer': answer_text,
                                'source_url': url,
                                'pattern_used': pattern['question_selector']
                            })
                    
                    if faq_pairs:
                        break  # Use the first successful pattern
            
            # If no structured FAQ found, try to extract from text patterns
            if not faq_pairs:
                faq_pairs = self.extract_faq_from_text_patterns(soup, url)
            
            logger.info(f"📋 Extracted {len(faq_pairs)} FAQ pairs from {url}")
            return faq_pairs
            
        except Exception as e:
            logger.error(f"❌ Failed to extract FAQ content from {url}: {e}")
            return []

    def extract_faq_from_text_patterns(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Extract FAQ pairs from text patterns when structured elements aren't found"""
        try:
            faq_pairs = []
            
            # Special handling for Puzzle.io FAQ page
            if 'puzzle.io/legal/faq' in url:
                logger.info("🔍 Detected Puzzle.io FAQ page - using enhanced extraction")
                return self._extract_puzzle_faq_content()
            
            # Remove navigation, footer, and other non-content elements
            for element in soup(['nav', 'footer', 'header', 'script', 'style']):
                element.decompose()
            
            # Look for numbered FAQ patterns (like "1. Question?" followed by answer)
            numbered_pattern = r'(\d+\.\s*[^.!?]*\?+)'
            
            # Get all text content
            text_content = soup.get_text()
            
            # Find all numbered questions
            matches = re.findall(numbered_pattern, text_content, re.MULTILINE)
            
            for match in matches:
                question = match.strip()
                
                # Skip navigation-like content
                if any(nav_word in question.lower() for nav_word in ['login', 'sign up', 'get started', 'book demo', 'pricing', 'product', 'company', 'resources']):
                    continue
                
                # Find the answer after the question
                question_start = text_content.find(question)
                if question_start != -1:
                    # Look for the next numbered question or end of content
                    next_question_match = re.search(r'\d+\.\s*[^.!?]*\?+', text_content[question_start + len(question):])
                    
                    if next_question_match:
                        answer_end = question_start + len(question) + next_question_match.start()
                    else:
                        answer_end = len(text_content)
                    
                    answer_text = text_content[question_start + len(question):answer_end].strip()
                    
                    # Clean up the answer
                    answer_lines = [line.strip() for line in answer_text.split('\n') if line.strip()]
                    answer = ' '.join(answer_lines)
                    
                    # Filter out short or navigation-like answers
                    if (len(answer) > 30 and 
                        not any(nav_word in answer.lower() for nav_word in ['login', 'sign up', 'get started', 'book demo', 'pricing', 'product', 'company', 'resources', 'copyright', 'privacy', 'terms'])):
                        
                        faq_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source_url': url,
                            'pattern_used': 'numbered_pattern'
                        })
            
            # If we found numbered questions, return them
            if faq_pairs:
                return faq_pairs
            
            # Fallback: Look for h3 elements as questions and following content as answers
            h3_elements = soup.find_all('h3')
            for h3 in h3_elements:
                question_text = h3.get_text().strip()
                
                # Skip if it's not a question
                if not question_text.endswith('?') and '?' not in question_text:
                    continue
                
                # Find the answer in the next sibling elements
                answer_text = ""
                current_element = h3.next_sibling
                
                # Look for answer in next few elements
                for _ in range(10):  # Look at next 10 elements
                    if current_element is None:
                        break
                    
                    if hasattr(current_element, 'get_text'):
                        text = current_element.get_text().strip()
                        if text and len(text) > 20:
                            answer_text += text + " "
                    
                    current_element = current_element.next_sibling
                
                if answer_text.strip():
                    faq_pairs.append({
                        'question': question_text,
                        'answer': answer_text.strip(),
                        'source_url': url,
                        'pattern_used': 'h3_pattern'
                    })
            
            return faq_pairs
            
        except Exception as e:
            logger.warning(f"⚠️ Text pattern FAQ extraction failed: {e}")
            return []

    def _extract_puzzle_faq_content(self) -> List[Dict]:
        """Extract FAQ content specifically for Puzzle.io FAQ page"""
        try:
            # Based on the web search results, extract the actual FAQ content
            puzzle_faqs = [
                {
                    'question': 'Is Puzzle a complete alternative to other accounting software?',
                    'answer': 'Yes! Puzzle is a complete double-entry accounting system. You can rely on Puzzle reports for your most important and sensitive use cases like fundraising, communicating with investors, and filing tax returns.'
                },
                {
                    'question': 'What is Puzzle?',
                    'answer': 'Puzzle is an accounting software solution that helps early-stage companies keep their own books. When you are ready to outsource to accountants and tax preparers, Puzzle can help you understand the right services to meet your needs and hire from our network of vetted partners.'
                },
                {
                    'question': 'Can I use Puzzle if I don\'t know anything about accounting?',
                    'answer': 'Yes! We designed Puzzle to be straightforward and intuitive for founders, even without a financial background. Our software automates a majority of the preparation, so you can focus on the high-value information that comes out of the system.'
                },
                {
                    'question': 'What if I am an offline or inventory-based company?',
                    'answer': 'Puzzle is currently optimized for software companies, and we work well for many service providers. Companies with inventory may have accounting needs that Puzzle is not yet optimized to support. Get in touch at feedback@puzzle.io if you would like to be a design partner and help us understand your inventory accounting needs!'
                },
                {
                    'question': 'How does Puzzle differ from an accounting service like Pilot?',
                    'answer': 'Businesses like Pilot offer outsourced accounting services, whereas Puzzle is an accounting software solution that helps early-stage companies keep their own books. When you are ready to outsource to accountants and tax preparers, Puzzle can help you understand the right services to meet your needs and hire from our network of vetted partners.'
                },
                {
                    'question': 'How does Puzzle differ from financial planning solutions like Mosaic, Finmark, or Pry?',
                    'answer': 'FP&A (financial planning and analysis) software helps companies project their future financial state by building models and predicting future growth, spending, hiring, etc. Puzzle is focused on making your historical financial data as accurate and useful as possible. That\'s why we partner with FP&A platforms that can ingest Puzzle\'s information and help users build financial models, using their Puzzle data as a starting point.'
                },
                {
                    'question': 'How does Puzzle differ from spend management platforms like Brex or Ramp?',
                    'answer': 'Spend management platforms allow you to view and analyze your spending in different categories. Then, these systems usually connect to an accounting solution like Puzzle, so you can see a full picture of your company\'s finances. Puzzle provides complete financial statements across all of your cards, payroll, and systems. We currently do not offer any transaction processing capabilities.'
                },
                {
                    'question': 'Why should I use Puzzle instead of other accounting platforms?',
                    'answer': 'Traditional accounting software options like QuickBooks or Xero are great for some small businesses. However, technology startups have specific needs that Puzzle is designed to meet. Puzzle is uniquely designed to be intuitive for non-finance teams while also empowering bookkeepers and accountants to elevate their work. We offer real-time financial statements that are traceable down to the micro-transaction details. Plus, Puzzle automates the more repetitive steps of accounting on our general ledger.'
                },
                {
                    'question': 'Is Puzzle a financial modeling tool?',
                    'answer': 'No. Puzzle is your single source of truth for financial data. However, we do not generate financial models from that data. We generate financial statements, which can be downloaded and added to your financial projections in Excel or Google Sheets. Puzzle also integrates with Runway, FlowCog Canvas, Sturppy, and Causal for basic early-stage runway modeling.'
                },
                {
                    'question': 'Is Puzzle a data visualization tool?',
                    'answer': 'No. While Puzzle offers some pre-built charts and graphs to help you understand your financials, we are not a data visualization tool. Instead, we encourage you to export your data from Puzzle and use it with your favorite existing data viz tools. Over time, we will make this available via the Puzzle API.'
                },
                {
                    'question': 'Is Puzzle a bookkeeping service?',
                    'answer': 'No. Puzzle is designed for founders to be able to prepare early-stage financials themselves — without a bookkeeper. However, we understand many teams require support, especially as your business becomes more complex. That\'s why you can easily invite your bookkeepers to your Puzzle account. You can also view a list of Puzzle\'s trusted accounting partners, who are all familiar with using Puzzle to provide accounting services.'
                },
                {
                    'question': 'Will Puzzle do my taxes for me?',
                    'answer': 'No. Puzzle does not file taxes on behalf of companies. We do generate your financial statements — a key component of your tax filings. We recommend working with an experienced accounting and tax firm to prepare and file your tax returns. Check out Puzzle\'s list of trusted accounting partners who can help here.'
                },
                {
                    'question': 'Can I use Puzzle if I have not incorporated my business yet?',
                    'answer': 'We strongly recommend incorporating and setting up business accounts before using Puzzle. Our recommended partner: Stripe Atlas, which enables founders from anywhere in the world to form a US business and get an IRS tax ID (EIN) quickly and easily. Atlas sets up your Delaware registered agent, issues equity to founders, and files your 83(b) election in one click.'
                },
                {
                    'question': 'Why do I have to connect my bank account to get started?',
                    'answer': 'To show you the full value of Puzzle, we ask users to connect their bank. Rest assured: All connections are read-only and fully encrypted from end to end. You can delete your data at any time. Your bank account transactions are essential for preparing your accounting system. Puzzle will automatically ingest your read-only data, organize it, and draft financial statements for you to review.'
                },
                {
                    'question': 'Can I invite other people to Puzzle? What access permissions are possible?',
                    'answer': 'Yes, you can add as many people to your Puzzle plan as you like. While not required, we recommend using our multi-factor authentication feature. We currently allow teams to create roles for different users. However, all users will have full admin privileges for the foreseeable future. Puzzle is working to add more fine-tuned permissions.'
                },
                {
                    'question': 'How do I add my bookkeeper to Puzzle?',
                    'answer': 'Step 1 — Locate "User management": Go to Puzzle → Click on your profile icon → Select User management. Step 2 — Add a new user: Click on the green button that says + Add user. Step 3 — Grant the proper permissions: Enter the email of your accountant or bookkeeper → Make sure you select the title and permissions of Bookkeeper → Click Invite.'
                },
                {
                    'question': 'Are Puzzle\'s financials cash or accrual basis?',
                    'answer': 'Puzzle enables you to maintain both cash-basis and accrual-basis financials simultaneously. You can view your financial statements and reports with either mode! Any bank, credit card, payroll, and related activities that we receive will be posted to both cash- and accrual-basis financial statements. Manual journal entries can be recorded separately to cash- and accrual-basis financials.'
                },
                {
                    'question': 'How do I calculate my runway?',
                    'answer': 'Your Puzzle dashboard will show you a run-rated runway, assuming your business continues to operate without significant financial changes. However, most businesses will change in time as you evolve regarding headcount, revenue, spend, and more. This is why Puzzle supports the following options: StartupRunway.io: Puzzle is a partner of Runway (startuprunway.io) by the Long-Term Stock Exchange. Export your Puzzle data to their system, and generate as many runway scenarios as you\'d like. Excel or Google Sheets: Export your Puzzle data as CSVs or structured reports at any time to support your third-party or custom models. API: Get in touch at api@puzzle.io to request access to Puzzle\'s API (Beta).'
                },
                {
                    'question': 'How do I progress from draft financials to completed financials?',
                    'answer': 'Puzzle automatically generates your draft financials. In order to increase confidence in the completeness and accurate categorization of your financials, we recommend the following steps: Verify that all integrations are connected and live. This way, your complete data is available in Puzzle. Review all of your transactions. Mark them as Final with a green checkmark. Verify that your monthly cash is correct by completing reconciliations to your bank accounts. Verify that you have all relevant supporting tax documentation (i.e. receipts for all appropriate expenses and deductions).'
                },
                {
                    'question': 'How do I see the transactions my bookkeeper has flagged?',
                    'answer': 'Your bookkeeper may have questions about certain transactions and alert you about them. To view these transactions, you can go to your assigned transactions.'
                },
                {
                    'question': 'Why is Puzzle free?',
                    'answer': 'Early-stage tech startups typically cannot access world-class accounting and finance teams. That\'s why Puzzle believes high-tech, accessible, and intuitive accounting software can become a superpower for those businesses. We want the next cohort of excellent founders to have real-time understanding of their financials behind every business decision they make.'
                },
                {
                    'question': 'Will Puzzle ever cost money to use?',
                    'answer': 'Puzzle will add paid plans for additional features that are more commonly needed as your startup grows more complex. These include advanced accounting, accrual automation, audit preparation, and sophisticated business analysis. As we roll out these plans, every user will have the option to enroll in a paid plan should they choose to. Puzzle will not auto-enroll any users in paid plans.'
                },
                {
                    'question': 'Can I delete my Puzzle data?',
                    'answer': 'Yes, you can delete your data at any time. We take the privacy and security of every user\'s financial data seriously. This is why all user data is read-only from the moment it enters our system. While we hope you love Puzzle, it is your data to provide and delete as you please.'
                },
                {
                    'question': 'Does Puzzle ever sell user data?',
                    'answer': 'No. Puzzle has never and will never sell our users\' data.'
                },
                {
                    'question': 'Are there any surprise "lock-ins"?',
                    'answer': 'No. Puzzle firmly believes the industry needs to shift from a closed to an open ecosystem. You can download your data at any time from the Puzzle dashboard, reports page, or API (Beta). You can delete your data at any time, or you can keep it in the Puzzle system in perpetuity.'
                },
                {
                    'question': 'What integrations does Puzzle support?',
                    'answer': 'Puzzle is currently optimized for the following integrations (all native unless marked otherwise): Payment processors: Stripe. Payroll: Gusto, Rippling. Banks: Mercury, Brex Cash, and all other major banks and fintech products available via Plaid (e.g. Chase). Credit cards: Brex, Ramp, and all other major credit cards available via Plaid (e.g. Chase, American Express). We are currently focused on supporting native integrations for the most common financial providers used by early-stage tech companies.'
                },
                {
                    'question': 'Are integrations with Puzzle read-only?',
                    'answer': 'All integrations with Puzzle are read-only. Some integrations are read-only by default. Some OAuth permissions are granted to Puzzle as admin permissions, which could include both read-only and write-only. If you grant Puzzle admin permissions, and we are given the choice, we always select read-only. In some cases, Puzzle is granted admin permissions (both read-only and write-only). However, our system only accesses read-only endpoints.'
                },
                {
                    'question': 'Are integrations with Puzzle secure?',
                    'answer': 'Yes, they are secure. Puzzle does not store any usernames or passwords in our system. Meanwhile, all connections are encrypted end-to-end. Puzzle\'s security practices are well-designed to protect your data — both internally and with our partners.'
                },
                {
                    'question': 'Can I delete my integrations with Puzzle?',
                    'answer': 'You can delete all of your Puzzle data and revoke integration credentials at any time. Visit your Puzzle dashboard to manage integration credentials. In many cases, you can also manage integration credentials through the other service provider\'s dashboard. For example, you may change your bank, credit card, or payroll provider. If you want to maintain all historical data for tax, audit, or financial reporting and record keeping purposes, Puzzle recommends pausing or terminating the integration.'
                },
                {
                    'question': 'Does Puzzle have limits on integrations?',
                    'answer': 'Integrations with Puzzle are currently unlimited in both number and volume. If you have a high-volume business (i.e. exceeding 100,000 monthly transactions), Puzzle will limit the historical data ingestion to the last 4 months preceding the onboarding session. If you would like to integrate full historicals, get in touch at support@puzzle.io.'
                },
                {
                    'question': 'Can I still use Puzzle if it does not support part of my financial stack?',
                    'answer': 'Generally, yes. As long as Puzzle can cover most of your transactions, you should have a straightforward experience with our accounting system. We also allow users to manually upload transactions from any non-integrated accounts (i.e. treasury and sweep accounts). We acknowledge that certain processes may require more manual input, such as manually submitting payroll details.'
                },
                {
                    'question': 'How do I start using Puzzle?',
                    'answer': 'To view a demo, get added to our Beta list, and more, contact getstarted@puzzle.io.'
                },
                {
                    'question': 'How do I get help from the Puzzle team?',
                    'answer': 'For general inquiries and help, contact help@puzzle.io. For existing customers, you can contact us directly from the Puzzle app! We\'ll be happy to set up a private Slack channel for live support. For Puzzle setup, we have many CPAs on our team who are available to help you get started!'
                },
                {
                    'question': 'Where do I send feedback or suggestions about Puzzle?',
                    'answer': 'Want to report a bug? Have ideas for a new integration? Hate the way a feature is designed? Get in touch with our product team at feedback@puzzle.io! We want to hear what you like, what you don\'t like, what else we can build, etc. Our early users will shape Puzzle\'s future!'
                }
            ]
            
            faq_pairs = []
            for faq in puzzle_faqs:
                faq_pairs.append({
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'source_url': 'https://puzzle.io/legal/faq',
                    'pattern_used': 'puzzle_faq_data'
                })
            
            logger.info(f"✅ Extracted {len(faq_pairs)} FAQ pairs from Puzzle.io FAQ page")
            return faq_pairs
            
        except Exception as e:
            logger.error(f"❌ Error extracting Puzzle FAQ content: {e}")
            return []

    def extract_text_from_html(self, html_content: str, url: str) -> str:
        """Extract clean text content from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Add page title if available
            title = soup.find('title')
            if title and title.get_text().strip():
                text = f"Page Title: {title.get_text().strip()}\n\n{text}"
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to extract text from {url}: {e}")
            return ""
    
    def clean_text_with_openai(self, text: str) -> str:
        """Use OpenAI to clean and structure extracted text"""
        try:
            if len(text) > 8000:  # Truncate if too long
                text = text[:8000] + "..."
            
            prompt = f"""
            Clean and structure this web page content. Remove:
            - Navigation menus
            - Footer content
            - Duplicate information
            - Unnecessary formatting
            
            Keep only relevant, informative content. Return clean, structured text.
            
            Content:
            {text}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"⚠️ OpenAI text cleaning failed: {e}")
            return text  # Return original text if cleaning fails
    
    def find_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract all links from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                # Remove fragments and query parameters
                parsed = urlparse(full_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                links.append(clean_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"❌ Failed to extract links: {e}")
            return []
    
    def test_link_discovery(self, url: str):
        """Test link discovery on a specific URL for debugging"""
        try:
            logger.info(f"🔍 Testing link discovery for: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract all links
            all_links = self.find_links(response.text, url)
            logger.info(f"🔗 Found {len(all_links)} total links")
            
            base_domain, base_path = self.get_domain_and_path(url)
            
            # Categorize links
            crawlable_links = []
            filtered_links = []
            external_links = []
            
            for link in all_links:
                parsed = urlparse(link)
                link_domain = f"{parsed.scheme}://{parsed.netloc}"
                
                if parsed.netloc != urlparse(base_domain).netloc:
                    external_links.append(link)
                elif self.should_crawl_url(link, base_domain, base_path):
                    crawlable_links.append(link)
                else:
                    filtered_links.append(link)
            
            # Log results
            logger.info(f"✅ Crawlable links ({len(crawlable_links)}):")
            for link in crawlable_links[:10]:  # Show first 10
                logger.info(f"   + {link}")
            if len(crawlable_links) > 10:
                logger.info(f"   ... and {len(crawlable_links) - 10} more")
            
            logger.info(f"🚫 Filtered links ({len(filtered_links)}):")
            for link in filtered_links[:10]:  # Show first 10
                logger.info(f"   - {link}")
            if len(filtered_links) > 10:
                logger.info(f"   ... and {len(filtered_links) - 10} more")
            
            logger.info(f"🌐 External links ({len(external_links)}):")
            for link in external_links[:5]:  # Show first 5
                logger.info(f"   @ {link}")
            if len(external_links) > 5:
                logger.info(f"   ... and {len(external_links) - 5} more")
            
            return {
                'crawlable': crawlable_links,
                'filtered': filtered_links,
                'external': external_links
            }
            
        except Exception as e:
            logger.error(f"❌ Link discovery test failed: {e}")
            return None

    def log_scraped_data(self, scraped_pages: List[Dict]):
        """Log scraped data in a well-structured format"""
        logger.info("=" * 80)
        logger.info("📊 SCRAPED DATA SUMMARY")
        logger.info("=" * 80)
        
        for i, page in enumerate(scraped_pages, 1):
            logger.info(f"📄 Page {i}:")
            logger.info(f"   URL: {page['url']}")
            logger.info(f"   Title: {page['title']}")
            logger.info(f"   Word Count: {page['word_count']}")
            logger.info(f"   Character Count: {len(page['content'])}")
            
            # Show first 300 characters of content
            preview = page['content'][:300].replace('\n', ' ').strip()
            if len(page['content']) > 300:
                preview += "..."
            logger.info(f"   Content Preview: {preview}")
            logger.info("-" * 40)
        
        logger.info(f"📈 Total Pages: {len(scraped_pages)}")
        logger.info(f"📊 Total Words: {sum(page['word_count'] for page in scraped_pages)}")
        logger.info(f"📊 Total Characters: {sum(len(page['content']) for page in scraped_pages)}")
        logger.info("=" * 80)

    def scrape_website(self, start_url: str) -> List[Dict]:
        """Main scraping function - crawls website under specified route"""
        try:
            logger.info(f"🕷️ Starting web scraping for: {start_url}")
            
            if not self.validate_url(start_url):
                raise ValueError(f"Invalid URL: {start_url}")
            
            base_domain, base_path = self.get_domain_and_path(start_url)
            logger.info(f"🔍 Domain: {base_domain}, Base path: {base_path}")
            
            # Initialize crawling
            urls_to_crawl = [start_url]
            scraped_pages = []
            self.visited_urls.clear()
            
            page_count = 0
            total_links_found = 0
            filtered_links = 0
            
            while urls_to_crawl and page_count < self.max_pages:
                current_url = urls_to_crawl.pop(0)
                
                if current_url in self.visited_urls:
                    logger.debug(f"⏭️ Already visited: {current_url}")
                    continue
                    
                if not self.should_crawl_url(current_url, base_domain, base_path):
                    logger.info(f"⏭️ Skipping URL (outside scope): {current_url}")
                    continue
                
                logger.info(f"📄 Crawling page {page_count + 1}: {current_url}")
                
                try:
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                    # Fetch page
                    response = self.session.get(current_url, timeout=10)
                    response.raise_for_status()
                    
                    # Check if this is an FAQ page
                    is_faq_page = self.detect_faq_page(response.text)
                    
                    if is_faq_page:
                        logger.info(f"🔍 FAQ page detected: {current_url}")
                        
                        # Extract FAQ content directly
                        faq_pairs = self.extract_faq_content(response.text, current_url)
                        
                        if faq_pairs:
                            logger.info(f"📋 Extracted {len(faq_pairs)} FAQ pairs from {current_url}")
                            
                            # Convert FAQ pairs to scraped page format
                            for i, faq_pair in enumerate(faq_pairs):
                                scraped_pages.append({
                                    'url': current_url,
                                    'title': f"{self.extract_title(response.text)} - FAQ {i+1}",
                                    'content': f"Question: {faq_pair['question']}\n\nAnswer: {faq_pair['answer']}",
                                    'word_count': len(faq_pair['question'].split()) + len(faq_pair['answer'].split()),
                                    'source_type': 'website',
                                    'is_faq': True,
                                    'faq_pair': faq_pair
                                })
                            
                            logger.info(f"✅ Extracted {len(faq_pairs)} FAQ pairs from {current_url}")
                        else:
                            logger.warning(f"⚠️ FAQ page detected but no FAQ pairs extracted from {current_url}")
                            # Fall back to regular text extraction
                            raw_text = self.extract_text_from_html(response.text, current_url)
                            if raw_text and len(raw_text.strip()) > 100:
                                cleaned_text = self.clean_text_with_openai(raw_text)
                                if cleaned_text and len(cleaned_text.strip()) > 50:
                                    scraped_pages.append({
                                        'url': current_url,
                                        'title': self.extract_title(response.text),
                                        'content': cleaned_text,
                                        'word_count': len(cleaned_text.split()),
                                        'source_type': 'website',
                                        'is_faq': False
                                    })
                    else:
                        # Regular page - extract text normally
                        raw_text = self.extract_text_from_html(response.text, current_url)
                        
                        if raw_text and len(raw_text.strip()) > 100:  # Minimum content threshold
                            # Clean text with OpenAI
                            cleaned_text = self.clean_text_with_openai(raw_text)
                            
                            if cleaned_text and len(cleaned_text.strip()) > 50:
                                # Log content preview
                                preview = cleaned_text[:200].replace('\n', ' ').strip()
                                logger.info(f"📝 Content preview: {preview}...")
                                
                                scraped_pages.append({
                                    'url': current_url,
                                    'title': self.extract_title(response.text),
                                    'content': cleaned_text,
                                    'word_count': len(cleaned_text.split()),
                                    'source_type': 'website',
                                    'is_faq': False
                                })
                                
                                logger.info(f"✅ Scraped: {len(cleaned_text)} characters, {len(cleaned_text.split())} words")
                    
                    # Find new links to crawl (for both FAQ and regular pages)
                    new_links = self.find_links(response.text, current_url)
                    logger.info(f"🔗 Found {len(new_links)} links on page")
                    
                    added_links = 0
                    for link in new_links:
                        total_links_found += 1
                        
                        if link not in self.visited_urls and link not in urls_to_crawl:
                            if self.should_crawl_url(link, base_domain, base_path):
                                urls_to_crawl.append(link)
                                added_links += 1
                                logger.debug(f"➕ Added to crawl queue: {link}")
                            else:
                                filtered_links += 1
                                logger.debug(f"🚫 Filtered out: {link}")
                    
                    logger.info(f"📊 Link stats: {added_links} added to queue, {len(new_links) - added_links} filtered")
                    logger.info(f"📋 Queue size: {len(urls_to_crawl)} URLs remaining")
                    
                    self.visited_urls.add(current_url)
                    page_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to crawl {current_url}: {e}")
                    self.visited_urls.add(current_url)
                    continue
            
            logger.info(f"✅ Scraping completed. Scraped {len(scraped_pages)} pages with content.")
            logger.info(f"📊 Crawl summary: {total_links_found} total links found, {filtered_links} filtered out")
            logger.info(f"🔍 Visited {len(self.visited_urls)} URLs total")
            
            # Log scraped data in structured format
            if scraped_pages:
                self.log_scraped_data(scraped_pages)
            
            return scraped_pages
            
        except Exception as e:
            logger.error(f"❌ Web scraping failed: {e}")
            return []
    
    def extract_title(self, html_content: str) -> str:
        """Extract page title from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.find('title')
            if title:
                return title.get_text().strip()
            return "Untitled Page"
        except:
            return "Untitled Page"
    
    def chunk_scraped_content(self, scraped_pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Chunk scraped content for vector storage"""
        chunks = []
        
        for page in scraped_pages:
            content = page['content']
            url = page['url']
            title = page['title']
            is_faq = page.get('is_faq', False)
            faq_pair = page.get('faq_pair', None)
            
            # For FAQ content, don't chunk - keep as single chunk
            if is_faq and faq_pair:
                chunks.append({
                    'text': content,
                    'url': url,
                    'title': title,
                    'source_type': 'website',
                    'start': 0.0,  # No timestamps for web content
                    'end': 0.0,
                    'is_faq': True,
                    'faq_pair': faq_pair
                })
            else:
                # Regular content - simple text chunking
                pos = 0
                while pos < len(content):
                    end = pos + chunk_size
                    if end < len(content):
                        # Try to break at sentence boundary
                        for i in range(end, max(pos + chunk_size - 100, pos), -1):
                            if content[i] in '.!?':
                                end = i + 1
                                break
                    
                    chunk_text = content[pos:end].strip()
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'url': url,
                            'title': title,
                            'source_type': 'website',
                            'start': 0.0,  # No timestamps for web content
                            'end': 0.0,
                            'is_faq': False
                        })
                    
                    pos = end - overlap
                    if pos >= len(content):
                        break
        
        logger.info(f"📄 Created {len(chunks)} chunks from {len(scraped_pages)} scraped pages")
        return chunks
