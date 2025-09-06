#!/usr/bin/env python3
"""
Backfill Task: Clean up empty knowledge documents
One-time cleanup to fix existing data with empty content
"""

import os
import time
from typing import Dict, List
from pinecone import Pinecone

class KnowledgeBackfillTask:
    def __init__(self):
        """Initialize backfill task"""
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.indexes = {
            'knowledge': 'qudemo-knowledge-index',
            'web': 'qudemo-web-index',
            'legacy': 'qudemo-index'
        }
    
    def scan_and_fix_empty_knowledge(self, company_name: str = None, qudemo_id: str = None) -> Dict:
        """
        Scan for empty knowledge documents and fix them
        
        Args:
            company_name: Specific company to scan (optional)
            qudemo_id: Specific qudemo to scan (optional)
            
        Returns:
            Dict with scan results and fixes applied
        """
        try:
            print(f"🔍 Starting backfill scan for empty knowledge documents...")
            
            total_scanned = 0
            total_empty = 0
            total_fixed = 0
            total_deleted = 0
            
            # Scan all relevant indexes
            for index_name in [self.indexes['knowledge'], self.indexes['web'], self.indexes['legacy']]:
                try:
                    index = self.pc.Index(index_name)
                    
                    # Get all namespaces
                    stats = index.describe_index_stats()
                    namespaces = stats.get('namespaces', {})
                    
                    for namespace_name in namespaces.keys():
                        # Skip if filtering by company/qudemo
                        if company_name and company_name.lower().replace(' ', '-') not in namespace_name:
                            continue
                        if qudemo_id and qudemo_id not in namespace_name:
                            continue
                        
                        print(f"📊 Scanning namespace: {namespace_name} in index: {index_name}")
                        
                        # Query all vectors in this namespace
                        # Note: This is a simplified approach - in production you might want to use list operations
                        query_result = index.query(
                            vector=[0.0] * 3072,  # Dummy vector
                            top_k=10000,  # Large number to get all
                            include_metadata=True,
                            namespace=namespace_name
                        )
                        
                        namespace_scanned = 0
                        namespace_empty = 0
                        namespace_fixed = 0
                        namespace_deleted = 0
                        
                        for match in query_result.matches:
                            namespace_scanned += 1
                            total_scanned += 1
                            
                            metadata = match.metadata
                            
                            # Check if this is a knowledge document
                            source_type = metadata.get('source_type', '').lower()
                            source = metadata.get('source', '').lower()
                            
                            if source_type not in ['web_scraping', 'knowledge'] and source not in ['web_scraping', 'knowledge']:
                                continue
                            
                            # Check for empty content
                            text = metadata.get('text', '') or metadata.get('content', '')
                            content_length = len(text.strip())
                            
                            if content_length == 0:
                                namespace_empty += 1
                                total_empty += 1
                                
                                print(f"⚠️ Found empty knowledge doc: {match.id}")
                                
                                # Option 1: Mark as content_has_text=False
                                try:
                                    # Update metadata to mark as empty
                                    index.update(
                                        id=match.id,
                                        set_metadata={
                                            'content_has_text': False,
                                            'backfill_fixed': True,
                                            'backfill_timestamp': time.time()
                                        },
                                        namespace=namespace_name
                                    )
                                    namespace_fixed += 1
                                    total_fixed += 1
                                    print(f"✅ Marked as empty: {match.id}")
                                    
                                except Exception as update_error:
                                    print(f"❌ Failed to update {match.id}: {update_error}")
                                    
                                    # Option 2: Delete if update fails
                                    try:
                                        index.delete(ids=[match.id], namespace=namespace_name)
                                        namespace_deleted += 1
                                        total_deleted += 1
                                        print(f"🗑️ Deleted empty doc: {match.id}")
                                    except Exception as delete_error:
                                        print(f"❌ Failed to delete {match.id}: {delete_error}")
                            
                            elif content_length < 300:  # Below minimum threshold
                                print(f"⚠️ Found short content ({content_length} chars): {match.id}")
                                # Mark as potentially low quality
                                try:
                                    index.update(
                                        id=match.id,
                                        set_metadata={
                                            'content_has_text': True,
                                            'quality_warning': 'short_content',
                                            'backfill_checked': True,
                                            'backfill_timestamp': time.time()
                                        },
                                        namespace=namespace_name
                                    )
                                    print(f"✅ Marked as short content: {match.id}")
                                except Exception as e:
                                    print(f"❌ Failed to update short content {match.id}: {e}")
                        
                        print(f"📊 Namespace {namespace_name} results:")
                        print(f"   Scanned: {namespace_scanned}")
                        print(f"   Empty: {namespace_empty}")
                        print(f"   Fixed: {namespace_fixed}")
                        print(f"   Deleted: {namespace_deleted}")
                
                except Exception as index_error:
                    print(f"❌ Error scanning index {index_name}: {index_error}")
                    continue
            
            # Final summary
            result = {
                'success': True,
                'total_scanned': total_scanned,
                'total_empty': total_empty,
                'total_fixed': total_fixed,
                'total_deleted': total_deleted,
                'timestamp': time.time()
            }
            
            print(f"🎯 Backfill Summary:")
            print(f"   Total scanned: {total_scanned}")
            print(f"   Empty found: {total_empty}")
            print(f"   Fixed (marked): {total_fixed}")
            print(f"   Deleted: {total_deleted}")
            
            return result
            
        except Exception as e:
            print(f"❌ Backfill task failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_scanned': 0,
                'total_empty': 0,
                'total_fixed': 0,
                'total_deleted': 0
            }
    
    def re_scrape_failed_urls(self, company_name: str, qudemo_id: str) -> Dict:
        """
        Re-scrape URLs that failed content validation
        
        Args:
            company_name: Company name
            qudemo_id: QuDemo ID
            
        Returns:
            Dict with re-scraping results
        """
        try:
            print(f"🔄 Starting re-scraping for {company_name} qudemo {qudemo_id}...")
            
            # This would integrate with your existing scraping pipeline
            # For now, return a placeholder
            return {
                'success': True,
                'message': 'Re-scraping integration needed',
                'urls_processed': 0,
                'urls_successful': 0
            }
            
        except Exception as e:
            print(f"❌ Re-scraping failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Run backfill task"""
    backfill = KnowledgeBackfillTask()
    
    # Run backfill for all data
    result = backfill.scan_and_fix_empty_knowledge()
    
    if result['success']:
        print(f"✅ Backfill completed successfully!")
    else:
        print(f"❌ Backfill failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
