#!/usr/bin/env python3
"""
Configuration for Hybrid Video Processing System
Centralized configuration for all hybrid processing components
"""

import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HybridConfig:
    """Configuration class for hybrid video processing system"""
    
    # Google Cloud Configuration
    GOOGLE_CLOUD_CONFIG = {
        'project_id': os.getenv('GOOGLE_CLOUD_PROJECT_ID'),
        'service_account_path': os.getenv('GOOGLE_SERVICE_ACCOUNT_PATH', 'service-account-key.json'),
        'gcs_bucket_name': os.getenv('GCS_BUCKET_NAME', 'qudemo-video-processing'),
        'region': os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
    }
    
    # Speech-to-Text Configuration
    SPEECH_TO_TEXT_CONFIG = {
        'language_code': 'en-US',
        'sample_rate_hertz': 16000,
        'enable_word_time_offsets': True,
        'enable_automatic_punctuation': True,
        'model': 'video',  # Optimized for video content
        'use_enhanced': True,
        'timeout': 300,  # 5 minutes
        'max_retries': 3
    }
    
    # Video Intelligence Configuration
    VIDEO_INTELLIGENCE_CONFIG = {
        'features': [
            'SHOT_CHANGE_DETECTION',
            'LABEL_DETECTION',
            'OBJECT_TRACKING'
        ],
        'label_detection_mode': 'SHOT_AND_FRAME_MODE',
        'stationary_camera': True,
        'timeout': 300,  # 5 minutes
        'max_retries': 3
    }
    
    # Chunking Configuration
    CHUNKING_CONFIG = {
        'min_chunk_size': 100,
        'max_chunk_size': 1000,
        'chunk_overlap': 200,
        'min_segment_duration': 10,
        'max_segment_duration': 120,
        'confidence_threshold': 0.5,
        'min_content_length': 20,
        'chars_per_second': 15,  # Average speaking rate
        'words_per_second': 3,   # Average speaking rate
        'max_video_duration': 36000  # 10 hours max
    }
    
    # Hybrid Search Configuration
    HYBRID_SEARCH_CONFIG = {
        'semantic_weight': 0.7,  # Weight for semantic similarity
        'label_weight': 0.3,     # Weight for label matching
        'min_confidence': 0.1,   # Minimum confidence threshold
        'max_results': 20,       # Maximum results to consider
        'label_boost': 1.2,      # Boost factor for label matches
        'keyword_weight': 0.1    # Weight for keyword matching
    }
    
    # Validation Configuration
    VALIDATION_CONFIG = {
        'duration_parity_threshold': 0.95,  # 95% accuracy for duration parity
        'anchor_coverage_threshold': 0.8,   # 80% of boundaries should be anchored
        'label_utility_threshold': 0.3,     # 30% of chunks should have labels
        'quality_score_threshold': 70,      # Minimum quality score
        'max_processing_time': 600          # 10 minutes max processing time
    }
    
    # Pinecone Configuration
    PINECONE_CONFIG = {
        'indexes': {
            'video': 'qudemo-video-index',
            'knowledge': 'qudemo-knowledge-index',
            'web': 'qudemo-web-index',
            'legacy': 'qudemo-index'
        },
        'embedding_model': 'text-embedding-3-large',
        'embedding_dimensions': 3072,
        'namespace_separator': '-'
    }
    
    # Fallback Configuration
    FALLBACK_CONFIG = {
        'enable_youtube_captions': True,
        'enable_existing_processor': True,
        'fallback_timeout': 30,
        'max_fallback_retries': 2
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'enable_structured_logging': True,
        'log_file': 'hybrid_processing.log'
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get complete configuration dictionary"""
        return {
            'google_cloud': cls.GOOGLE_CLOUD_CONFIG,
            'speech_to_text': cls.SPEECH_TO_TEXT_CONFIG,
            'video_intelligence': cls.VIDEO_INTELLIGENCE_CONFIG,
            'chunking': cls.CHUNKING_CONFIG,
            'hybrid_search': cls.HYBRID_SEARCH_CONFIG,
            'validation': cls.VALIDATION_CONFIG,
            'pinecone': cls.PINECONE_CONFIG,
            'fallback': cls.FALLBACK_CONFIG,
            'logging': cls.LOGGING_CONFIG
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration and check for required environment variables"""
        required_vars = [
            'GOOGLE_CLOUD_PROJECT_ID',
            'PINECONE_API_KEY',
            'OPENAI_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {missing_vars}")
            return False
        
        # Check if service account file exists
        service_account_path = cls.GOOGLE_CLOUD_CONFIG['service_account_path']
        if not os.path.exists(service_account_path):
            print(f"⚠️ Service account file not found: {service_account_path}")
            print("   Using default credentials (for production environments)")
        
        print("✅ Configuration validation passed")
        return True
    
    @classmethod
    def get_namespace(cls, company_name: str, qudemo_id: str) -> str:
        """Generate namespace for company and qudemo"""
        clean_company = company_name.lower().replace(' ', cls.PINECONE_CONFIG['namespace_separator'])
        return f"{clean_company}{cls.PINECONE_CONFIG['namespace_separator']}{qudemo_id}"
    
    @classmethod
    def get_processing_timeout(cls) -> int:
        """Get processing timeout in seconds"""
        return cls.VALIDATION_CONFIG['max_processing_time']
    
    @classmethod
    def is_hybrid_enabled(cls) -> bool:
        """Check if hybrid processing is enabled"""
        return os.getenv('ENABLE_HYBRID_PROCESSING', 'true').lower() == 'true'
    
    @classmethod
    def get_debug_mode(cls) -> bool:
        """Check if debug mode is enabled"""
        return os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Global configuration instance
config = HybridConfig()

def get_hybrid_config() -> HybridConfig:
    """Get global hybrid configuration instance"""
    return config
