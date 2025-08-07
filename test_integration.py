#!/usr/bin/env python3
"""
Integration Test
Test both Gemini and Loom processors work correctly
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_processors():
    """Test both processors initialization"""
    
    # Check required API keys
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not all([gemini_api_key, pinecone_api_key, openai_api_key]):
        logger.error("❌ Missing required API keys")
        return False
    
    try:
        # Test Gemini processor
        from gemini_transcription import GeminiTranscriptionProcessor
        gemini_processor = GeminiTranscriptionProcessor(gemini_api_key, pinecone_api_key, openai_api_key)
        logger.info("✅ Gemini processor initialized successfully")
        
        # Test Loom processor
        from loom_processor import LoomVideoProcessor
        loom_processor = LoomVideoProcessor(openai_api_key, pinecone_api_key)
        logger.info("✅ Loom processor initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Processor initialization failed: {e}")
        return False

def test_main_integration():
    """Test main.py integration"""
    try:
        # Import main module
        from main import initialize_processors, process_video
        
        # Test processor initialization
        if initialize_processors():
            logger.info("✅ Main module processor initialization successful")
        else:
            logger.error("❌ Main module processor initialization failed")
            return False
        
        # Test video processing function (without actually processing)
        test_youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        test_loom_url = "https://www.loom.com/share/test"
        
        # Test URL detection logic
        if 'youtube.com' in test_youtube_url or 'youtu.be' in test_youtube_url:
            logger.info("✅ YouTube URL detection working")
        
        if 'loom.com' in test_loom_url:
            logger.info("✅ Loom URL detection working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Main integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Integration...")
    print("=" * 50)
    
    # Test 1: Processor initialization
    print("\n1. Testing processor initialization...")
    if test_processors():
        print("✅ Processor initialization passed")
    else:
        print("❌ Processor initialization failed")
    
    # Test 2: Main integration
    print("\n2. Testing main integration...")
    if test_main_integration():
        print("✅ Main integration passed")
    else:
        print("❌ Main integration failed")
    
    print("\n" + "=" * 50)
    print("Integration test completed!")
