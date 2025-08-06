#!/usr/bin/env python3
"""
Proxy Configuration for YouTube Download VM
Helps bypass anti-bot measures by rotating IP addresses
"""

import random
import requests
from typing import Optional, Dict, Any

class ProxyManager:
    def __init__(self):
        # Free proxy list (you can replace with paid proxies for better reliability)
        self.proxy_list = [
            # Add your proxy servers here
            # Format: 'http://username:password@ip:port'
            # Example: 'http://user:pass@proxy1.example.com:8080'
        ]
        
        # Rotate user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        ]
    
    def get_random_proxy(self) -> Optional[str]:
        """Get a random proxy from the list"""
        if self.proxy_list:
            return random.choice(self.proxy_list)
        return None
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent"""
        return random.choice(self.user_agents)
    
    def get_yt_dlp_options(self, use_proxy: bool = True) -> Dict[str, Any]:
        """Get yt-dlp options with anti-bot measures"""
        options = {
            'http_headers': {
                'User-Agent': self.get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip,deflate',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                    'player_skip': ['webpage'],
                }
            },
            'sleep_interval': random.randint(1, 3),
            'max_sleep_interval': random.randint(3, 7),
            'retries': 3,
            'fragment_retries': 3,
            'cookiesfrombrowser': None,
        }
        
        if use_proxy:
            proxy = self.get_random_proxy()
            if proxy:
                options['proxy'] = proxy
        
        return options

# Global proxy manager instance
proxy_manager = ProxyManager() 