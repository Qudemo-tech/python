#!/usr/bin/env python3
"""
Check Render deployment logs
"""

import requests
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def get_render_logs():
    """Fetch deployment logs from Render"""
    
    # Render API details
    RENDER_API_KEY = os.getenv("RENDER_API_KEY")
    SERVICE_ID = os.getenv("RENDER_SERVICE_ID")
    
    if not RENDER_API_KEY:
        logger.error("❌ RENDER_API_KEY environment variable not set")
        return None
    
    if not SERVICE_ID:
        logger.error("❌ RENDER_SERVICE_ID environment variable not set")
        return None
    
    headers = {
        "Authorization": f"Bearer {RENDER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Get service details
        service_url = f"https://api.render.com/v1/services/{SERVICE_ID}"
        logger.info(f"🔍 Fetching service details from: {service_url}")
        
        response = requests.get(service_url, headers=headers, timeout=30)
        logger.info(f"📡 Service API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"❌ Service API error: {response.text}")
            return None
            
        service_data = response.json()
        logger.info(f"✅ Service found: {service_data.get('service', {}).get('name', 'Unknown')}")
        logger.info(f"📊 Service status: {service_data.get('service', {}).get('status', 'Unknown')}")
        
        # Get recent deployments
        deployments_url = f"https://api.render.com/v1/services/{SERVICE_ID}/deploys"
        logger.info(f"🔍 Fetching recent deployments...")
        
        response = requests.get(deployments_url, headers=headers, timeout=30)
        logger.info(f"📡 Deployments API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"❌ Deployments API error: {response.text}")
            return None
            
        deployments = response.json()
        logger.info(f"📊 Found {len(deployments)} deployments")
        
        if not deployments:
            logger.warning("⚠️ No deployments found")
            return None
        
        # Get the latest deployment
        latest_deployment = deployments[0]
        logger.info(f"🔍 Latest deployment data: {latest_deployment}")
        
        # Extract deployment info from the nested structure
        deploy_data = latest_deployment.get('deploy', {})
        deployment_id = deploy_data.get('id')
        deployment_status = deploy_data.get('status')
        created_at = deploy_data.get('createdAt')
        
        logger.info(f"📝 Latest deployment ID: {deployment_id}")
        logger.info(f"📊 Deployment status: {deployment_status}")
        logger.info(f"🕐 Created at: {created_at}")
        
        # Get deployment logs
        if deployment_id:
            logs_url = f"https://api.render.com/v1/services/{SERVICE_ID}/deploys/{deployment_id}/log"
            logger.info(f"🔍 Fetching deployment logs...")
            
            response = requests.get(logs_url, headers=headers, timeout=60)
            response.raise_for_status()
            
            logs_data = response.json()
            logs = logs_data.get('log', '')
            
            if logs:
                logger.info("📋 Deployment logs:")
                logger.info("=" * 80)
                print(logs)
                logger.info("=" * 80)
                
                # Save logs to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"render_logs_{timestamp}.txt"
                
                with open(log_filename, 'w', encoding='utf-8') as f:
                    f.write(f"Deployment ID: {deployment_id}\n")
                    f.write(f"Status: {deployment_status}\n")
                    f.write(f"Created: {created_at}\n")
                    f.write("=" * 80 + "\n")
                    f.write(logs)
                
                logger.info(f"💾 Logs saved to: {log_filename}")
                return {
                    'deployment_id': deployment_id,
                    'status': deployment_status,
                    'logs': logs,
                    'log_file': log_filename
                }
            else:
                logger.warning("⚠️ No logs found for deployment")
                return None
        else:
            logger.error("❌ No deployment ID found")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to fetch Render logs: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return None

def check_service_status():
    """Check if the service is responding"""
    service_url = "https://python-gwir.onrender.com"
    
    logger.info(f"🔍 Checking service status: {service_url}")
    
    try:
        response = requests.get(f"{service_url}/health", timeout=10)
        logger.info(f"✅ Service is responding: {response.status_code}")
        return True
    except requests.exceptions.Timeout:
        logger.error("⏰ Service timeout")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Service error: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Checking Render deployment status...")
    
    # Check if service is responding
    service_ok = check_service_status()
    
    if not service_ok:
        logger.warning("⚠️ Service is not responding, checking deployment logs...")
    
    # Get deployment logs
    logs_data = get_render_logs()
    
    if logs_data:
        logger.info("✅ Successfully retrieved deployment logs")
        
        # Check for common deployment issues
        logs = logs_data['logs'].lower()
        
        if 'error' in logs:
            logger.warning("⚠️ Errors found in deployment logs")
        
        if 'timeout' in logs:
            logger.warning("⚠️ Timeout issues found in deployment logs")
        
        if 'memory' in logs:
            logger.warning("⚠️ Memory issues found in deployment logs")
        
        if 'import' in logs:
            logger.warning("⚠️ Import issues found in deployment logs")
        
        if 'requirements' in logs:
            logger.warning("⚠️ Requirements/dependency issues found in deployment logs")
    else:
        logger.error("❌ Failed to retrieve deployment logs")

if __name__ == "__main__":
    main() 