#!/usr/bin/env python3
"""
Knowledge Content Backfill Script
Adds content_has_text field to existing knowledge documents for proper filtering
"""

import os
import logging
from typing import Dict, List
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeContentBackfill:
    """Backfill script for knowledge content validation"""
    
    def __init__(self):
        """Initialize backfill script"""
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.min_content_length = 300  # Minimum characters for substantial content
        
    def backfill_all_namespaces(self) -> Dict:
        """Backfill all namespaces in knowledge index"""
        try:
            logger.info("🔧 Starting knowledge content backfill...")
            
            # Get knowledge index
            knowledge_index = self.pc.Index('qudemo-knowledge-index')
            
            # Get all namespaces
            namespaces = self._get_all_namespaces(knowledge_index)
            
            total_updated = 0
            total_processed = 0
            
            for namespace in namespaces:
                logger.info(f"📁 Processing namespace: {namespace}")
                namespace_stats = self._backfill_namespace(knowledge_index, namespace)
                total_updated += namespace_stats['updated']
                total_processed += namespace_stats['processed']
            
            logger.info(f"✅ Backfill complete: {total_updated}/{total_processed} documents updated")
            
            return {
                'success': True,
                'total_updated': total_updated,
                'total_processed': total_processed,
                'namespaces_processed': len(namespaces)
            }
            
        except Exception as e:
            logger.error(f"❌ Backfill failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_all_namespaces(self, index) -> List[str]:
        """Get all namespaces from the index"""
        try:
            # Query with a dummy vector to get namespace info
            # This is a workaround since Pinecone doesn't have a direct namespace list API
            namespaces = set()
            
            # Try to get namespaces by querying with a zero vector
            try:
                zero_vector = [0.0] * 3072  # text-embedding-3-large dimension
                results = index.query(
                    vector=zero_vector,
                    top_k=1,
                    include_metadata=True
                )
                
                # Extract namespace from results
                if results.matches:
                    # This is a workaround - we'll need to scan through the data
                    pass
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not get namespaces automatically: {e}")
            
            # For now, return empty list - this would need to be implemented
            # based on your specific namespace naming convention
            logger.info("ℹ️ Namespace detection not implemented - please specify namespaces manually")
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting namespaces: {e}")
            return []
    
    def _backfill_namespace(self, index, namespace: str) -> Dict:
        """Backfill a specific namespace"""
        try:
            logger.info(f"🔍 Scanning namespace: {namespace}")
            
            # Get all vectors in namespace (this is a simplified approach)
            # In practice, you might need to paginate through results
            zero_vector = [0.0] * 3072
            results = index.query(
                vector=zero_vector,
                top_k=10000,  # Large number to get all results
                include_metadata=True,
                namespace=namespace
            )
            
            updated_count = 0
            processed_count = 0
            
            for match in results.matches:
                processed_count += 1
                
                # Check if update is needed
                metadata = match.metadata
                current_content_has_text = metadata.get('content_has_text')
                
                # Determine if content has substantial text
                text = metadata.get('text', '') or metadata.get('content', '')
                has_substantial_text = len(text.strip()) >= self.min_content_length
                
                # Update if needed
                if current_content_has_text != has_substantial_text:
                    # Update the vector with new metadata
                    updated_metadata = metadata.copy()
                    updated_metadata['content_has_text'] = has_substantial_text
                    
                    # Upsert the updated vector
                    index.upsert(
                        vectors=[{
                            'id': match.id,
                            'values': match.values,  # Keep existing embedding
                            'metadata': updated_metadata
                        }],
                        namespace=namespace
                    )
                    
                    updated_count += 1
                    
                    if updated_count % 100 == 0:
                        logger.info(f"📝 Updated {updated_count} documents in {namespace}")
            
            logger.info(f"✅ Namespace {namespace}: {updated_count}/{processed_count} documents updated")
            
            return {
                'updated': updated_count,
                'processed': processed_count
            }
            
        except Exception as e:
            logger.error(f"❌ Error backfilling namespace {namespace}: {e}")
            return {
                'updated': 0,
                'processed': 0
            }
    
    def backfill_specific_namespace(self, namespace: str) -> Dict:
        """Backfill a specific namespace"""
        try:
            logger.info(f"🔧 Backfilling specific namespace: {namespace}")
            
            knowledge_index = self.pc.Index('qudemo-knowledge-index')
            stats = self._backfill_namespace(knowledge_index, namespace)
            
            return {
                'success': True,
                'namespace': namespace,
                'updated': stats['updated'],
                'processed': stats['processed']
            }
            
        except Exception as e:
            logger.error(f"❌ Error backfilling namespace {namespace}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_content_quality(self, namespace: str) -> Dict:
        """Validate content quality in a namespace"""
        try:
            logger.info(f"🔍 Validating content quality in: {namespace}")
            
            knowledge_index = self.pc.Index('qudemo-knowledge-index')
            
            # Get sample of documents
            zero_vector = [0.0] * 3072
            results = knowledge_index.query(
                vector=zero_vector,
                top_k=1000,
                include_metadata=True,
                namespace=namespace
            )
            
            stats = {
                'total_documents': len(results.matches),
                'has_content_has_text': 0,
                'content_has_text_true': 0,
                'content_has_text_false': 0,
                'substantial_content': 0,
                'insubstantial_content': 0,
                'missing_text': 0
            }
            
            for match in results.matches:
                metadata = match.metadata
                
                # Check if content_has_text field exists
                if 'content_has_text' in metadata:
                    stats['has_content_has_text'] += 1
                    if metadata['content_has_text']:
                        stats['content_has_text_true'] += 1
                    else:
                        stats['content_has_text_false'] += 1
                
                # Check content quality
                text = metadata.get('text', '') or metadata.get('content', '')
                if not text:
                    stats['missing_text'] += 1
                elif len(text.strip()) >= self.min_content_length:
                    stats['substantial_content'] += 1
                else:
                    stats['insubstantial_content'] += 1
            
            logger.info(f"📊 Content quality stats for {namespace}:")
            logger.info(f"   Total documents: {stats['total_documents']}")
            logger.info(f"   Has content_has_text field: {stats['has_content_has_text']}")
            logger.info(f"   content_has_text=True: {stats['content_has_text_true']}")
            logger.info(f"   content_has_text=False: {stats['content_has_text_false']}")
            logger.info(f"   Substantial content (≥{self.min_content_length} chars): {stats['substantial_content']}")
            logger.info(f"   Insubstantial content: {stats['insubstantial_content']}")
            logger.info(f"   Missing text: {stats['missing_text']}")
            
            return {
                'success': True,
                'namespace': namespace,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error validating content quality: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Main function for running backfill"""
    try:
        backfill = KnowledgeContentBackfill()
        
        # Example usage - backfill specific namespace
        # Replace with your actual namespace
        namespace = "acme-corp-demo-123"  # Example namespace
        
        print("🔧 Knowledge Content Backfill Tool")
        print("=" * 50)
        
        # Validate content quality first
        print(f"📊 Validating content quality in {namespace}...")
        validation_result = backfill.validate_content_quality(namespace)
        
        if validation_result['success']:
            print("✅ Validation complete")
            
            # Ask user if they want to proceed with backfill
            response = input(f"\nProceed with backfill for {namespace}? (y/n): ")
            if response.lower() == 'y':
                print(f"🔧 Starting backfill for {namespace}...")
                result = backfill.backfill_specific_namespace(namespace)
                
                if result['success']:
                    print(f"✅ Backfill complete: {result['updated']}/{result['processed']} documents updated")
                else:
                    print(f"❌ Backfill failed: {result['error']}")
            else:
                print("❌ Backfill cancelled")
        else:
            print(f"❌ Validation failed: {validation_result['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
