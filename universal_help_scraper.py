import asyncio, re, time, json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from selectolax.parser import HTMLParser
from uritools import urijoin, uridefrag

# Content extractors
import trafilatura
from readability import Document as ReadDoc
import extruct

# Robots
import urllib.robotparser as robotparser

# Optional JS fallback (load only if used)
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False


# ---------------------------
# Config
# ---------------------------
@dataclass
class CrawlConfig:
    seed_url: str
    allowed_path_prefix: str  # e.g. "/helpcenter" (must start with "/")
    max_pages: int = 200
    concurrency: int = 8
    request_timeout: float = 20.0
    delay_ms: int = 250           # polite crawl
    use_js_fallback: bool = True  # render only if needed & allowed
    min_text_chars: int = 300     # store only substantive pages
    user_agent: str = "QuDemoBot/1.0 (+contact@example.com)"
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "/login", "/signin", "/signup", "/account", "/privacy", "/terms",
        "/support/tickets", "/cart", "/checkout", "/admin", "/search"
    ])


@dataclass
class ExtractedDoc:
    url: str
    title: str
    text: str
    html_len: int
    word_count: int
    cms: Optional[str] = None
    schema_org: Optional[Dict] = None


# ---------------------------
# Utility: URL gating
# ---------------------------
def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = uridefrag(href)[0]  # drop #fragments
    try:
        abs_url = urijoin(base, href)
        # strip default ports
        parsed = urlparse(abs_url)
        netloc = parsed.hostname
        if parsed.port in (80, 443, None):
            netloc = parsed.hostname
        parsed = parsed._replace(netloc=netloc or parsed.netloc)
        return urlunparse(parsed)
    except Exception:
        return None


def same_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.scheme == pb.scheme and pa.hostname == pb.hostname


def path_allowed(url: str, seed: str, prefix: str) -> bool:
    p, s = urlparse(url), urlparse(seed)
    if not same_domain(url, seed):
        return False
    # Require the URL path to start with the provided prefix relative to domain root
    path = p.path or "/"
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if not path.startswith(prefix.rstrip("/")):
        return False
    return True


def excluded(url: str, patterns: List[str]) -> bool:
    u = url.lower()
    return any(pat in u for pat in patterns)


# ---------------------------
# CMS detection (heuristics)
# ---------------------------
def detect_cms(html: str) -> Optional[str]:
    h = html.lower()
    if "zendesk" in h or "zdassets.com" in h:
        return "zendesk"
    if "helpscout" in h or "beacon-v2" in h:
        return "helpscout"
    if "salesforce" in h or "slds-" in h or "lightning" in h or "my.site.com" in h:
        return "salesforce"
    if "intercom" in h and "help center" in h:
        return "intercom"
    if "freshdesk" in h:
        return "freshdesk"
    if "document360" in h:
        return "document360"
    if "helpjuice" in h:
        return "helpjuice"
    if "kb" in h and "knowledge base" in h:
        return "generic_kb"
    return None


# ---------------------------
# Content extraction strategies
# ---------------------------
def extract_with_trafilatura(html: str, url: str) -> Tuple[str, str]:
    """
    Returns (title, text) or ("","") if not enough content.
    """
    try:
        downloaded = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True,
            include_formatting=True, favor_recall=True
        )
        if downloaded:
            # Trafilatura returns plain text with some structure
            try:
                bare_extraction = trafilatura.bare_extraction(html, url=url)
                title = bare_extraction.get("title") if isinstance(bare_extraction, dict) else ""
            except:
                title = ""
            text = downloaded.strip()
            return title, text
        return "", ""
    except Exception as e:
        print(f"❌ Trafilatura extraction failed: {e}")
        return "", ""


def extract_with_readability(html: str) -> Tuple[str, str]:
    try:
        doc = ReadDoc(html)
        title = (doc.short_title() or "").strip()
        content_html = doc.summary(html_partial=True)
        tree = HTMLParser(content_html)
        text = tree.text(separator="\n").strip()
        return title, text
    except Exception as e:
        print(f"❌ Readability extraction failed: {e}")
        return "", ""


def pick_best_title(fallbacks: List[str]) -> str:
    for t in fallbacks:
        if t and len(t.strip()) >= 5:
            return t.strip()
    return "Untitled"


def parse_schema_org(html: str, url: str) -> Optional[Dict]:
    try:
        data = extruct.extract(html, base_url=url, syntaxes=["json-ld", "microdata", "opengraph"])
        return data or None
    except Exception:
        return None


# ---------------------------
# Heuristic: does this page need JS?
# ---------------------------
def needs_js(html: str) -> bool:
    # Few signals: no meaningful text, heavy client-side frameworks, placeholder shells
    text = HTMLParser(html).text().strip()
    if len(text) < 200:
        return True
    if re.search(r"(ng\-|data-reactroot|__NEXT_DATA__|<template|hydration|skeleton)", html, re.I):
        return True
    return False


# ---------------------------
# Core crawler
# ---------------------------
class UniversalHelpCrawler:
    def __init__(self, cfg: CrawlConfig):
        self.cfg = cfg
        self.seed = cfg.seed_url.rstrip("/")
        self.allowed_prefix = cfg.allowed_path_prefix.rstrip("/")
        self.base_domain = f"{urlparse(self.seed).scheme}://{urlparse(self.seed).hostname}"
        self.seen: Set[str] = set()
        self.to_visit: asyncio.Queue[str] = asyncio.Queue()
        self.docs: List[ExtractedDoc] = []
        self.robots = robotparser.RobotFileParser()
        self._robots_loaded = False

    async def _load_robots(self, client: httpx.AsyncClient):
        robots_url = urljoin(self.base_domain, "/robots.txt")
        try:
            r = await client.get(robots_url, timeout=self.cfg.request_timeout, headers={"User-Agent": self.cfg.user_agent})
            if r.status_code == 200:
                self.robots.parse(r.text.splitlines())
                self._robots_loaded = True
        except Exception:
            # If robots cannot be fetched, default to allow (you may change to deny)
            self._robots_loaded = False

    def _robots_allowed(self, url: str) -> bool:
        if not self._robots_loaded:
            return True
        try:
            return self.robots.can_fetch(self.cfg.user_agent, url)
        except Exception:
            return True

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        if not self._robots_allowed(url):
            return None
        try:
            r = await client.get(url, timeout=self.cfg.request_timeout, headers={"User-Agent": self.cfg.user_agent})
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("text/html"):
                return r.text
        except Exception:
            return None
        return None

    async def _render(self, url: str) -> Optional[str]:
        if not PLAYWRIGHT_AVAILABLE or not self.cfg.use_js_fallback:
            return None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
                page = await browser.new_page(user_agent=self.cfg.user_agent)
                await page.goto(url, wait_until="networkidle", timeout=45000)
                html = await page.content()
                await browser.close()
                return html
        except Exception:
            return None

    def _extract_links(self, base_url: str, html: str) -> List[str]:
        tree = HTMLParser(html)
        links = []
        for a in tree.css("a[href]"):
            href = a.attributes.get("href")
            candidate = normalize_url(base_url, href)
            if not candidate:
                continue
            if not path_allowed(candidate, self.seed, self.allowed_prefix):
                continue
            if excluded(candidate, self.cfg.exclude_patterns):
                continue
            links.append(candidate)
        return links

    def _extract_content(self, html: str, url: str) -> Optional[ExtractedDoc]:
        cms = detect_cms(html)
        
        # Try Trafilatura first
        t_title, t_text = extract_with_trafilatura(html, url)
        r_title, r_text = extract_with_readability(html)

        # Pick stronger text
        text = max([t_text, r_text], key=lambda x: len(x or ""))
        
        # If content is too short, try basic HTML parsing as fallback
        if not text or len(text) < self.cfg.min_text_chars:
            # Try basic HTML parsing
            tree = HTMLParser(html)
            basic_text = tree.text(separator="\n").strip()
            if len(basic_text) > len(text or ""):
                text = basic_text
        
        if not text or len(text) < self.cfg.min_text_chars:
            return None

        # Title selection
        tree = HTMLParser(html)
        h1 = tree.css_first("h1")
        raw_title = h1.text(strip=True) if h1 else ""
        doc_title = pick_best_title([raw_title, t_title, r_title])

        schema = parse_schema_org(html, url)

        return ExtractedDoc(
            url=url,
            title=doc_title,
            text=text,
            html_len=len(html),
            word_count=len(text.split()),
            cms=cms,
            schema_org=schema
        )

    async def _handle_page(self, client: httpx.AsyncClient, url: str):
        if url in self.seen:
            return
        self.seen.add(url)

        html = await self._fetch(client, url)
        if html is None:
            return

        # If the page looks empty/JSy, try a single render pass
        if needs_js(html) and self.cfg.use_js_fallback:
            rendered = await self._render(url)
            if rendered:
                html = rendered

        doc = self._extract_content(html, url)
        if doc:
            self.docs.append(doc)

        links = self._extract_links(url, html)
        for link in links:
            if link not in self.seen:
                await self.to_visit.put(link)

        await asyncio.sleep(self.cfg.delay_ms / 1000.0)  # politeness

    async def crawl(self) -> List[ExtractedDoc]:
        async with httpx.AsyncClient(http2=True, headers={"User-Agent": self.cfg.user_agent}) as client:
            await self._load_robots(client)

            # Seed only if path-allowed
            if path_allowed(self.seed, self.seed, self.allowed_prefix):
                await self.to_visit.put(self.seed)

            workers = [asyncio.create_task(self._worker(client)) for _ in range(self.cfg.concurrency)]
            
            # Add timeout to prevent hanging
            try:
                await asyncio.wait_for(self.to_visit.join(), timeout=300)  # 5 minute timeout
            except asyncio.TimeoutError:
                print("⚠️ Crawling timed out after 5 minutes")
            
            for w in workers:
                w.cancel()
            return self.docs

    async def _worker(self, client: httpx.AsyncClient):
        pages = 0
        while pages < self.cfg.max_pages:
            try:
                url = await asyncio.wait_for(self.to_visit.get(), timeout=10.0)  # Increased timeout
            except asyncio.TimeoutError:
                break
            try:
                await self._handle_page(client, url)
            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
            finally:
                self.to_visit.task_done()
                pages += 1


# ---------------------------
# Integration with existing system
# ---------------------------
class UniversalScraperIntegration:
    """Integration layer to convert ExtractedDoc to existing chunk format"""
    
    def __init__(self):
        self.crawler = None
    
    async def scrape_website_universal(self, website_url: str, company_name: str, qudemo_id: str) -> List[Dict]:
        """
        Scrape website using universal scraper and convert to existing chunk format
        """
        try:
            print(f"🚀 Starting universal scraper for: {website_url}")
            
            # Simple approach: just process the single URL without crawling
            async with httpx.AsyncClient(http2=True, headers={"User-Agent": "QuDemoBot/1.0 (Help Center Crawler)"}) as client:
                # Fetch the page
                response = await client.get(website_url, timeout=15.0)
                if response.status_code != 200:
                    print(f"❌ Failed to fetch {website_url}: {response.status_code}")
                    return []
                
                html = response.text
                print(f"✅ Fetched HTML: {len(html)} characters")
                
                # Extract content using our methods
                cms = detect_cms(html)
                print(f"📊 CMS detected: {cms}")
                
                # Try content extraction
                t_title, t_text = extract_with_trafilatura(html, website_url)
                r_title, r_text = extract_with_readability(html)
                
                # Pick best content
                text = max([t_text, r_text], key=lambda x: len(x or ""))
                
                # If content is too short, try basic HTML parsing
                if not text or len(text) < 100:
                    tree = HTMLParser(html)
                    basic_text = tree.text(separator="\n").strip()
                    if len(basic_text) > len(text or ""):
                        text = basic_text
                        print(f"🔄 Using basic HTML parsing: {len(text)} chars")
                
                if not text or len(text) < 100:
                    print(f"❌ Content too short: {len(text or '')} chars")
                    return []
                
                # Get title
                tree = HTMLParser(html)
                h1 = tree.css_first("h1")
                raw_title = h1.text(strip=True) if h1 else ""
                title = pick_best_title([raw_title, t_title, r_title])
                
                print(f"✅ Extracted content: {title} ({len(text)} chars)")
                
                # Split into chunks
                content_chunks = self._split_content_into_chunks(text, max_chunk_size=2000)
                
                # Convert to existing chunk format
                chunks = []
                for i, chunk_text in enumerate(content_chunks):
                    chunk = {
                        'content': chunk_text,
                        'title': title,
                        'url': website_url,
                        'source_type': 'web_scraping',
                        'content_has_text': True,
                        'company_name': company_name,
                        'qudemo_id': qudemo_id,
                        'cms_platform': cms,
                        'word_count': len(chunk_text.split()),
                        'chunk_index': i,
                        'total_chunks': len(content_chunks)
                    }
                    chunks.append(chunk)
                
                print(f"✅ Created {len(chunks)} chunks")
                return chunks
            
        except Exception as e:
            print(f"❌ Universal scraper error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _split_content_into_chunks(self, text: str, max_chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk = text[start:end]
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            
            if last_period > max_chunk_size * 0.7:  # If period is in last 30%
                end = start + last_period + 1
            elif last_newline > max_chunk_size * 0.7:  # If newline is in last 30%
                end = start + last_newline + 1
            
            chunks.append(text[start:end])
            start = end - overlap  # Overlap for context
        
        return chunks


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Example: only crawl /helpcenter on puzzle.io, ignore /contact etc.
    seed = "https://puzzle.io/helpcenter"
    cfg = CrawlConfig(
        seed_url=seed,
        allowed_path_prefix="/helpcenter",   # strict path gating
        max_pages=120,
        concurrency=10,
        use_js_fallback=True,                # render only when content looks JS-only
        min_text_chars=250
    )

    async def run():
        crawler = UniversalHelpCrawler(cfg)
        docs = await crawler.crawl()
        # Persist or pass downstream (Pinecone, etc.)
        for d in docs:
            print(json.dumps({
                "url": d.url,
                "title": d.title,
                "words": d.word_count,
                "cms": d.cms
            }, ensure_ascii=False))

    asyncio.run(run())
