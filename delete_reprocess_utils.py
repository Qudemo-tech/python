#!/usr/bin/env python3
"""
Delete/Reprocess Utilities for QuDemo Video Processing
Implements targeted delete by namespace + version for idempotent re-upserts
"""

import os
import logging
from typing import Dict, List, Optional
from pinecone import Pinecone

logger = logging.getLogger(__name__)

class DeleteReprocessManager:
    """Manages targeted deletion and reprocessing of video content"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.indexes = {
            'video': 'qudemo-video-index',
            'knowledge': 'qudemo-knowledge-index',
            'analytics': 'qudemo-analytics-index',
            'legacy': 'qudemo-index'
        }
    
    def delete_by_namespace_and_version(self, 
                                      company_name: str, 
                                      qudemo_id: str,
                                      transcript_version: str = "v1",
                                      chunking_version: str = "v2-seg-safe") -> Dict:
        """
        Delete all vectors for a specific namespace and version combination
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            transcript_version: Transcript version to delete
            chunking_version: Chunking version to delete
            
        Returns:
            Dict with deletion results
        """
        try:
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Delete from video index
            video_index = self.pc.Index(self.indexes['video'])
            
            # Query to find vectors with specific version
            filter_criteria = {
                "company_name": {"$eq": company_name},
                "qudemo_id": {"$eq": qudemo_id},
                "transcript_version": {"$eq": transcript_version},
                "chunking_version": {"$eq": chunking_version}
            }
            
            # Get all vectors matching criteria
            results = video_index.query(
                vector=[0.0] * 3072,  # Dummy vector
                top_k=10000,  # Large number to get all
                namespace=namespace,
                include_metadata=True,
                filter=filter_criteria
            )
            
            vector_ids = [match.id for match in results.matches]
            
            if vector_ids:
                # Delete vectors
                video_index.delete(ids=vector_ids, namespace=namespace)
                logger.info(f"✅ Deleted {len(vector_ids)} vectors from namespace '{namespace}' with versions {transcript_version}/{chunking_version}")
                
                return {
                    'success': True,
                    'deleted_count': len(vector_ids),
                    'namespace': namespace,
                    'transcript_version': transcript_version,
                    'chunking_version': chunking_version,
                    'deleted_ids': vector_ids[:10]  # First 10 for logging
                }
            else:
                logger.info(f"ℹ️ No vectors found to delete in namespace '{namespace}' with versions {transcript_version}/{chunking_version}")
                return {
                    'success': True,
                    'deleted_count': 0,
                    'namespace': namespace,
                    'transcript_version': transcript_version,
                    'chunking_version': chunking_version,
                    'message': 'No vectors found to delete'
                }
                
        except Exception as e:
            logger.error(f"❌ Delete operation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'namespace': namespace,
                'transcript_version': transcript_version,
                'chunking_version': chunking_version
            }
    
    def get_namespace_stats(self, company_name: str, qudemo_id: str) -> Dict:
        """
        Get statistics for a namespace
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            
        Returns:
            Dict with namespace statistics
        """
        try:
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            video_index = self.pc.Index(self.indexes['video'])
            
            # Get index stats
            stats = video_index.describe_index_stats()
            
            if namespace in stats.namespaces:
                ns_stats = stats.namespaces[namespace]
                
                # Query to get version breakdown
                results = video_index.query(
                    vector=[0.0] * 3072,  # Dummy vector
                    top_k=1000,  # Sample
                    namespace=namespace,
                    include_metadata=True
                )
                
                version_counts = {}
                for match in results.matches:
                    metadata = match.metadata
                    transcript_ver = metadata.get('transcript_version', 'unknown')
                    chunking_ver = metadata.get('chunking_version', 'unknown')
                    version_key = f"{transcript_ver}/{chunking_ver}"
                    version_counts[version_key] = version_counts.get(version_key, 0) + 1
                
                return {
                    'success': True,
                    'namespace': namespace,
                    'total_vectors': ns_stats.vector_count,
                    'version_breakdown': version_counts,
                    'sample_size': len(results.matches)
                }
            else:
                return {
                    'success': True,
                    'namespace': namespace,
                    'total_vectors': 0,
                    'version_breakdown': {},
                    'message': 'Namespace not found'
                }
                
        except Exception as e:
            logger.error(f"❌ Get namespace stats failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'namespace': namespace
            }

# Global instance
_delete_reprocess_manager = None

def initialize_delete_reprocess_manager() -> bool:
    """Initialize the delete/reprocess manager"""
    global _delete_reprocess_manager
    try:
        _delete_reprocess_manager = DeleteReprocessManager()
        logger.info("✅ Delete/Reprocess Manager initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Delete/Reprocess Manager: {e}")
        return False

def get_delete_reprocess_manager() -> Optional[DeleteReprocessManager]:
    """Get the delete/reprocess manager instance"""
    return _delete_reprocess_manager
