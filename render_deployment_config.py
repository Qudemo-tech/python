#!/usr/bin/env python3
"""
Render Deployment Configuration
Optimized settings for Render's 512MB memory limit
"""

import os
import logging

logger = logging.getLogger(__name__)

def get_render_optimized_settings():
    """
    Get Render-optimized settings based on environment
    """
    # Check if we're running on Render
    is_render = os.getenv('RENDER', 'false').lower() == 'true'
    
    if is_render:
        logger.info("🚀 Running on Render - using optimized settings for 2GB RAM plan")
        return {
            # Memory management - optimized for 2GB RAM
            'max_memory_mb': 1800,  # Conservative limit for 2GB plan
            'memory_cleanup_threshold': 1400,  # Start cleanup at 1.4GB
            'memory_fail_threshold': 1900,  # Fail at 1.9GB
            
            # Video processing - optimized for production
            'ytdlp_format': 'worst[height<=240]',  # Force 240p for memory efficiency
            'ytdlp_max_filesize': '100M',  # Smaller files for faster processing
            'whisper_model': 'tiny',  # Fastest processing, lowest memory
            'embedding_batch_size': 8,  # Larger batches with more RAM
            'prefer_ytdlp': True,
            
            # Loom-specific settings
            'LOOM_FALLBACK_TO_API': False,  # Disable API fallback for stability
            'MAX_VIDEO_SIZE_MB': 30,  # Conservative video size limit
            
            # Processing optimizations
            'enable_aggressive_cleanup': True,  # Aggressive cleanup for production
            'cleanup_frequency': 3,  # Cleanup every 3 operations
            'retry_attempts': 2,
            
            # Audio conversion settings
            'audio_sample_rate': 16000,  # Whisper standard
            'audio_channels': 1,  # Mono for efficiency
            'audio_codec': 'pcm_s16le',  # 16-bit PCM
        }
    else:
        logger.info("💻 Running locally - using development settings")
        return {
            # Memory management
            'max_memory_mb': 800,  # Higher limit for local development
            'memory_cleanup_threshold': 600,
            'memory_fail_threshold': 750,
            
            # Video processing
            'ytdlp_format': 'best[height<=720]',
            'ytdlp_max_filesize': '200M',
            'whisper_model': 'tiny',  # Use tiny for better performance
            'embedding_batch_size': 5,
            'prefer_ytdlp': True,
            
            # Loom-specific settings
            'LOOM_FALLBACK_TO_API': True,
            'MAX_VIDEO_SIZE_MB': 50,
            
            # Processing optimizations
            'enable_aggressive_cleanup': False,
            'cleanup_frequency': 10,
            'retry_attempts': 3,
            
            # Audio conversion settings
            'audio_sample_rate': 16000,
            'audio_channels': 1,
            'audio_codec': 'pcm_s16le',
        }

def get_memory_settings():
    """Get memory-specific settings"""
    settings = get_render_optimized_settings()
    return {
        'max_memory_mb': settings['max_memory_mb'],
        'cleanup_threshold': settings['memory_cleanup_threshold'],
        'fail_threshold': settings['memory_fail_threshold'],
        'enable_aggressive_cleanup': settings['enable_aggressive_cleanup'],
    }

def get_video_settings():
    """Get video processing settings"""
    settings = get_render_optimized_settings()
    return {
        'ytdlp_format': settings['ytdlp_format'],
        'ytdlp_max_filesize': settings['ytdlp_max_filesize'],
        'whisper_model': settings['whisper_model'],
        'max_video_size_mb': settings['MAX_VIDEO_SIZE_MB'],
        'retry_attempts': settings['retry_attempts'],
    }

def get_embedding_settings():
    """Get embedding processing settings"""
    settings = get_render_optimized_settings()
    return {
        'batch_size': settings['embedding_batch_size'],
        'model': 'text-embedding-3-small',  # Always use small model for memory
    } 