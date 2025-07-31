#!/usr/bin/env python3
"""
Fetch live logs from Render using the API
"""

import os
import time
import requests
import json
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class RenderLogsFetcher:
    """Fetch live logs from Render service"""
    
    def __init__(self):
        self.api_token = os.getenv('RENDER_API_TOKEN')
        self.service_id = os.getenv('RENDER_SERVICE_ID', 'python-gwir')
        
        if not self.api_token:
            logger.error("❌ RENDER_API_TOKEN environment variable not set")
            logger.info("💡 Please set your Render API token:")
            logger.info("   export RENDER_API_TOKEN=your_token_here")
            raise ValueError("RENDER_API_TOKEN not set")
        
        self.base_url = "https://api.render.com/v1"
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
    
    def get_service_id(self):
        """Get the service ID if not provided"""
        try:
            url = f"{self.base_url}/services"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            services = response.json()
            for service in services:
                if service.get('name') == self.service_id or service.get('serviceId') == self.service_id:
                    return service['id']
            
            logger.error(f"❌ Service '{self.service_id}' not found")
            logger.info("Available services:")
            for service in services:
                logger.info(f"  - {service.get('name')} (ID: {service.get('id')})")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get service ID: {e}")
            return None
    
    def get_logs(self, service_id=None, limit=100, since=None):
        """Get logs from Render service"""
        
        if not service_id:
            service_id = self.get_service_id()
            if not service_id:
                return None
        
        try:
            url = f"{self.base_url}/services/{service_id}/logs"
            params = {
                'limit': limit
            }
            
            if since:
                params['since'] = since
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Failed to get logs: {e}")
            return None
    
    def stream_logs(self, service_id=None, interval=5):
        """Stream logs continuously"""
        
        if not service_id:
            service_id = self.get_service_id()
            if not service_id:
                return
        
        logger.info(f"🔄 Starting log stream for service: {service_id}")
        logger.info(f"⏱️ Polling interval: {interval} seconds")
        logger.info("Press Ctrl+C to stop")
        
        last_timestamp = None
        
        try:
            while True:
                logs = self.get_logs(service_id, limit=50, since=last_timestamp)
                
                if logs and logs.get('logs'):
                    for log_entry in logs['logs']:
                        timestamp = log_entry.get('timestamp', '')
                        message = log_entry.get('message', '')
                        
                        # Format timestamp
                        if timestamp:
                            try:
                                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                formatted_time = timestamp
                        else:
                            formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Print log entry
                        print(f"[{formatted_time}] {message}")
                        
                        # Update last timestamp
                        if timestamp:
                            last_timestamp = timestamp
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Log streaming stopped")
        except Exception as e:
            logger.error(f"❌ Log streaming failed: {e}")

def main():
    """Main function"""
    
    try:
        fetcher = RenderLogsFetcher()
        
        # Get service ID
        service_id = fetcher.get_service_id()
        if not service_id:
            logger.error("❌ Could not find service ID")
            return
        
        logger.info(f"✅ Found service ID: {service_id}")
        
        # Start streaming logs
        fetcher.stream_logs(service_id)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize log fetcher: {e}")

if __name__ == "__main__":
    main() 