#!/usr/bin/env python3
"""
Check deployment status and diagnose issues
"""

import requests
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def check_service_status():
    """Check if the service is responding"""
    
    service_url = "https://python-gwir.onrender.com"
    
    logger.info(f"🔍 Checking service status: {service_url}")
    
    # Try different endpoints
    endpoints = [
        "/health",
        "/",
        "/docs"
    ]
    
    for endpoint in endpoints:
        try:
            logger.info(f"📡 Testing endpoint: {endpoint}")
            response = requests.get(f"{service_url}{endpoint}", timeout=15)
            logger.info(f"✅ {endpoint} - Status: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"📝 Response: {response.text[:200]}...")
                return True
        except requests.exceptions.Timeout:
            logger.error(f"⏰ {endpoint} - Timeout (15s)")
        except requests.exceptions.ConnectionError:
            logger.error(f"🔌 {endpoint} - Connection Error")
        except Exception as e:
            logger.error(f"❌ {endpoint} - Error: {e}")
    
    return False

def check_github_deployment():
    """Check if the latest deployment was successful"""
    
    logger.info("🔍 Checking GitHub deployment status...")
    
    # Check if the latest commit was pushed
    try:
        import subprocess
        result = subprocess.run(['git', 'log', '--oneline', '-1'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"📝 Latest commit: {result.stdout.strip()}")
        else:
            logger.error(f"❌ Git log failed: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Git check failed: {e}")

def check_branch_status():
    """Check current branch and status"""
    
    try:
        import subprocess
        
        # Check current branch
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            current_branch = result.stdout.strip()
            logger.info(f"🌿 Current branch: {current_branch}")
        
        # Check if there are uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            if result.stdout.strip():
                logger.warning("⚠️ There are uncommitted changes:")
                logger.warning(result.stdout)
            else:
                logger.info("✅ No uncommitted changes")
                
    except Exception as e:
        logger.error(f"❌ Git status check failed: {e}")

def main():
    """Main diagnostic function"""
    
    logger.info("🚀 Starting deployment diagnostics...")
    logger.info("=" * 60)
    
    # Check git status
    check_branch_status()
    check_github_deployment()
    
    logger.info("=" * 60)
    
    # Check service status
    if check_service_status():
        logger.info("✅ Service is responding!")
    else:
        logger.error("❌ Service is not responding")
        logger.info("💡 Possible issues:")
        logger.info("   1. Service crashed during deployment")
        logger.info("   2. Environment variables missing")
        logger.info("   3. Dependencies failed to install")
        logger.info("   4. Service is still starting up")
        logger.info("   5. Render service is down")
        
        logger.info("🔧 Recommended actions:")
        logger.info("   1. Check Render dashboard for deployment logs")
        logger.info("   2. Verify environment variables are set")
        logger.info("   3. Check if all dependencies are in requirements.txt")
        logger.info("   4. Try redeploying the service")

if __name__ == "__main__":
    main() 