#!/usr/bin/env python3
"""
Company Bucket Service
Handles immediate GCS bucket creation when companies are created
Ensures complete data isolation between companies
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from google_cloud_storage_service import GoogleCloudStorageService

# Configure logging
logger = logging.getLogger(__name__)

class CompanyBucketService:
    """Service for managing company-specific GCS buckets"""
    
    def __init__(self):
        """Initialize company bucket service"""
        self.gcs_service = GoogleCloudStorageService(
            service_account_path=os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service-account-key.json')
        )
        logger.info("✅ Company Bucket Service initialized")
    
    def create_company_bucket(self, company_name: str) -> Dict[str, Any]:
        """
        Create a GCS bucket for a company immediately
        
        Args:
            company_name: Name of the company
            
        Returns:
            Result dictionary with success status and bucket info
        """
        try:
            logger.info(f"🏢 Creating GCS bucket for company: {company_name}")
            
            # Get or create the company bucket
            bucket = self.gcs_service._get_company_bucket(company_name)
            
            # Create initial folder structure
            self._create_initial_folder_structure(bucket, company_name)
            
            logger.info(f"✅ Company bucket created successfully: qudemo-{company_name.lower().replace(' ', '-').replace('_', '-')}")
            
            return {
                'success': True,
                'company_name': company_name,
                'bucket_name': f"qudemo-{company_name.lower().replace(' ', '-').replace('_', '-')}",
                'message': f'Company bucket created successfully for {company_name}'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create company bucket for {company_name}: {e}")
            return {
                'success': False,
                'company_name': company_name,
                'error': str(e),
                'message': f'Failed to create company bucket for {company_name}'
            }
    
    def _create_initial_folder_structure(self, bucket, company_name: str):
        """
        Create initial folder structure in company bucket
        
        Args:
            bucket: GCS bucket object
            company_name: Company name
        """
        try:
            # Create a welcome file to establish the bucket structure
            welcome_content = {
                "company_name": company_name,
                "created_at": "2024-01-15T10:00:00.000Z",
                "bucket_type": "company_qudemos",
                "description": f"Storage bucket for {company_name} QuDemos and video transcripts",
                "structure": {
                    "qudemos": "Each QuDemo gets its own folder",
                    "transcripts": "Video transcripts stored as transcript.json",
                    "qa_answers": "Q&A answers stored as qa_answers.json"
                }
            }
            
            # Upload welcome file
            blob = bucket.blob("README.json")
            blob.upload_from_string(
                json.dumps(welcome_content, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"📁 Created initial folder structure for {company_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create initial folder structure: {e}")
    
    def get_company_bucket_info(self, company_name: str) -> Dict[str, Any]:
        """
        Get information about a company's bucket
        
        Args:
            company_name: Name of the company
            
        Returns:
            Bucket information dictionary
        """
        try:
            bucket = self.gcs_service._get_company_bucket(company_name)
            
            # List all QuDemos in the bucket
            blobs = self.gcs_service.client.list_blobs(bucket.name)
            
            qudemos = {}
            for blob in blobs:
                path_parts = blob.name.split('/')
                if len(path_parts) >= 1 and path_parts[0] != 'README.json':
                    qudemo_id = path_parts[0]
                    filename = path_parts[1] if len(path_parts) > 1 else 'unknown'
                    
                    if qudemo_id not in qudemos:
                        qudemos[qudemo_id] = {
                            'qudemo_id': qudemo_id,
                            'files': [],
                            'created_at': blob.time_created.isoformat() if blob.time_created else 'unknown'
                        }
                    
                    qudemos[qudemo_id]['files'].append({
                        'filename': filename,
                        'size': blob.size,
                        'updated_at': blob.updated.isoformat() if blob.updated else 'unknown'
                    })
            
            return {
                'success': True,
                'company_name': company_name,
                'bucket_name': bucket.name,
                'qudemos': list(qudemos.values()),
                'total_qudemos': len(qudemos),
                'total_files': sum(len(q['files']) for q in qudemos.values())
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get bucket info for {company_name}: {e}")
            return {
                'success': False,
                'company_name': company_name,
                'error': str(e)
            }
    
    def create_qudemo_folder(self, company_name: str, qudemo_id: str) -> Dict[str, Any]:
        """
        Create a QuDemo folder inside the company bucket
        
        Args:
            company_name: Name of the company
            qudemo_id: QuDemo ID
            
        Returns:
            Result dictionary
        """
        try:
            logger.info(f"📁 Creating QuDemo folder for {company_name}/{qudemo_id}")
            
            bucket = self.gcs_service._get_company_bucket(company_name)
            
            # Create QuDemo info file
            qudemo_info = {
                "qudemo_id": qudemo_id,
                "company_name": company_name,
                "created_at": "2024-01-15T10:00:00.000Z",
                "status": "active",
                "description": f"QuDemo folder for {company_name}",
                "files": {
                    "transcript.json": "Video transcript data",
                    "qa_answers.json": "Q&A answers (created when questions asked)"
                }
            }
            
            # Upload QuDemo info file
            blob = bucket.blob(f"{qudemo_id}/qudemo_info.json")
            blob.upload_from_string(
                json.dumps(qudemo_info, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"✅ QuDemo folder created: {company_name}/{qudemo_id}")
            
            return {
                'success': True,
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'message': f'QuDemo folder created successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create QuDemo folder for {company_name}/{qudemo_id}: {e}")
            return {
                'success': False,
                'company_name': company_name,
                'qudemo_id': qudemo_id,
                'error': str(e)
            }
    
    def list_all_companies(self) -> Dict[str, Any]:
        """
        List all companies with their bucket information
        
        Returns:
            Dictionary with all companies and their info
        """
        try:
            companies = self.gcs_service.list_companies()
            
            company_info = {}
            for company in companies:
                info = self.get_company_bucket_info(company)
                if info.get('success'):
                    company_info[company] = info
            
            return {
                'success': True,
                'companies': company_info,
                'total_companies': len(company_info)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to list companies: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Global instance
company_bucket_service = None

def initialize_company_bucket_service():
    """Initialize the company bucket service"""
    global company_bucket_service
    try:
        company_bucket_service = CompanyBucketService()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Company Bucket Service: {e}")
        return False

def get_company_bucket_service():
    """Get the company bucket service instance"""
    return company_bucket_service
