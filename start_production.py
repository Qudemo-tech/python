#!/usr/bin/env python3
"""
Production startup script for QuDemo Python Backend
Optimized for cloud deployment (Render, Heroku, AWS, etc.)
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        'OPENAI_API_KEY',
        'PINECONE_API_KEY',
        'GEMINI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ All required environment variables are set")
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import fastapi
        import uvicorn
        import openai
        import pinecone
        import whisper
        import yt_dlp
        import google.generativeai
        logger.info("✅ All required dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def main():
    """Main production startup function"""
    logger.info("🚀 Starting QuDemo Python Backend (Production Mode)")
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependencies check failed")
        sys.exit(1)
    
    # Set production environment variables
    os.environ['PYTHON_ENV'] = 'production'
    
    # Import and start the application
    try:
        from main import app
        import uvicorn
        
        # Get port from environment (Render, Heroku, etc.)
        port = int(os.getenv('PORT', 5001))
        host = os.getenv('HOST', '0.0.0.0')
        
        logger.info(f"🌐 Starting server on {host}:{port}")
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
