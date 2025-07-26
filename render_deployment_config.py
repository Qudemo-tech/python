#!/usr/bin/env python3
"""
Render Deployment Configuration
Optimized settings for Render's 512MB memory limit
"""

import os
import logging

logger = logging.getLogger(__name__)

def get_render_optimized_settings():
    """Get optimized settings for Render deployment with 2GB RAM"""
    if os.getenv("RENDER"):
        # Production settings for 2GB RAM plan
        return {
            # Memory settings
            'max_memory_mb': 1800,
            'memory_cleanup_threshold': 1400,
            'memory_fail_threshold': 1900,
            
            # Video download settings - relaxed limits for 2GB RAM
            'ytdlp_format': 'worst[height<=480]',  # Allow higher quality
            'ytdlp_max_filesize': '500M',  # Increased to 500MB
            'MAX_VIDEO_SIZE_MB': 300,  # Increased to 300MB for 2GB RAM
            
            # Whisper settings
            'whisper_model': 'tiny',
            
            # Embedding settings
            'embedding_batch_size': 8,
            
            # Processing settings
            'prefer_ytdlp': True,
            'LOOM_FALLBACK_TO_API': False,
            'enable_aggressive_cleanup': True,
            'cleanup_frequency': 3,
            'retry_attempts': 2,
            
            # Audio conversion settings
            'audio_sample_rate': 16000,
            'audio_channels': 1,
            'audio_codec': 'pcm_s16le'
        }
    else:
        # Development settings
        return {
            'max_memory_mb': 1500,
            'memory_cleanup_threshold': 1200,
            'memory_fail_threshold': 1800,
            'ytdlp_format': 'best[height<=720]',
            'ytdlp_max_filesize': '500M',
            'whisper_model': 'tiny',
            'embedding_batch_size': 5,
            'prefer_ytdlp': True,
            'LOOM_FALLBACK_TO_API': True,
            'MAX_VIDEO_SIZE_MB': 200,  # Higher for development
            'enable_aggressive_cleanup': False,
            'cleanup_frequency': 10,
            'retry_attempts': 3,
            'audio_sample_rate': 16000,
            'audio_channels': 1,
            'audio_codec': 'pcm_s16le'
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