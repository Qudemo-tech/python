#!/usr/bin/env python3
"""
Memory usage test script for video processing
"""
import os
import psutil
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_memory_limits():
    """Check current memory usage and limits"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    total_mb = system_memory.total / 1024 / 1024
    available_mb = system_memory.available / 1024 / 1024
    
    logger.info(f"💾 Current memory usage: {memory_mb:.1f}MB")
    logger.info(f"💾 System total memory: {total_mb:.1f}MB")
    logger.info(f"💾 System available memory: {available_mb:.1f}MB")
    logger.info(f"💾 Memory usage percentage: {(memory_mb/total_mb)*100:.1f}%")
    
    # Check if we're approaching limits
    if memory_mb > 400:  # 400MB warning threshold
        logger.warning(f"⚠️ High memory usage: {memory_mb:.1f}MB")
    
    if available_mb < 100:  # Less than 100MB available
        logger.error(f"❌ Low available memory: {available_mb:.1f}MB")
    
    return memory_mb, total_mb, available_mb

def test_whisper_memory():
    """Test Whisper model memory usage"""
    try:
        import whisper
        logger.info("🔍 Testing Whisper model memory usage...")
        
        # Test tiny model
        logger.info("Loading tiny model...")
        check_memory_limits()
        model_tiny = whisper.load_model("tiny")
        memory_after_tiny = check_memory_limits()
        
        # Test base model
        logger.info("Loading base model...")
        model_base = whisper.load_model("base")
        memory_after_base = check_memory_limits()
        
        # Test small model
        logger.info("Loading small model...")
        model_small = whisper.load_model("small")
        memory_after_small = check_memory_limits()
        
        logger.info("✅ Whisper memory test completed")
        
    except Exception as e:
        logger.error(f"❌ Whisper test failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting memory usage test...")
    check_memory_limits()
    test_whisper_memory()
    logger.info("✅ Memory test completed") 