#!/usr/bin/env python3
"""
Simple log monitor for Render service
"""

import requests
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class SimpleLogMonitor:
    """Simple monitor for Render service"""
    
    def __init__(self):
        self.service_url = "https://python-gwir.onrender.com"
        self.test_video_url = "https://youtu.be/ZAGxqOT2l2U?si=uB03UNTGKGzgIJ7L"
    
    def check_health(self):
        """Check if the service is healthy"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ Service is healthy: {response.status_code}")
                return True
            else:
                logger.warning(f"⚠️ Service returned status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Service health check failed: {e}")
            return False
    
    def trigger_video_test(self):
        """Trigger a video processing test"""
        try:
            logger.info(f"🧪 Triggering video test with: {self.test_video_url}")
            
            payload = {
                "video_url": self.test_video_url,
                "company_name": "qudemo",
                "is_loom": False,
                "source": "youtube",
                "meeting_link": None
            }
            
            response = requests.post(
                f"{self.service_url}/process-video/qudemo",
                json=payload,
                timeout=60
            )
            
            logger.info(f"📊 Test response status: {response.status_code}")
            
            if response.status_code == 200:
                logger.info("✅ Video test successful!")
                return True
            else:
                logger.error(f"❌ Video test failed: {response.status_code}")
                try:
                    error_detail = response.json()
                    logger.error(f"📝 Error details: {error_detail}")
                except:
                    logger.error(f"📝 Response text: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Video test request failed: {e}")
            return False
    
    def monitor_and_test(self, interval=30):
        """Monitor service and run tests"""
        logger.info(f"🔄 Starting service monitor")
        logger.info(f"⏱️ Check interval: {interval} seconds")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                logger.info("=" * 60)
                logger.info(f"🕐 Check at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check health
                if self.check_health():
                    # If healthy, trigger a test
                    self.trigger_video_test()
                else:
                    logger.warning("⚠️ Service not healthy, skipping test")
                
                logger.info(f"⏳ Waiting {interval} seconds until next check...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped")
        except Exception as e:
            logger.error(f"❌ Monitoring failed: {e}")

def main():
    """Main function"""
    monitor = SimpleLogMonitor()
    monitor.monitor_and_test()

if __name__ == "__main__":
    main() 