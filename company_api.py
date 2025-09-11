#!/usr/bin/env python3
"""
Company API Endpoints
Handles company creation with immediate GCS bucket creation
"""

import os
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from company_bucket_service import initialize_company_bucket_service, get_company_bucket_service

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

class CompanyCreateRequest(BaseModel):
    name: str
    description: str = ""
    website: str = ""
    logo: str = ""

class CompanyCreateResponse(BaseModel):
    success: bool
    company_name: str
    bucket_name: str
    message: str
    error: str = None

@router.post("/create-company", response_model=CompanyCreateResponse)
async def create_company_with_bucket(request: CompanyCreateRequest):
    """
    Create a new company and immediately create its GCS bucket
    
    This ensures complete data isolation - each company gets its own bucket
    """
    try:
        logger.info(f"🏢 Creating company with GCS bucket: {request.name}")
        
        # Initialize company bucket service if not already done
        if not get_company_bucket_service():
            if not initialize_company_bucket_service():
                raise HTTPException(status_code=500, detail="Failed to initialize company bucket service")
        
        company_bucket_service = get_company_bucket_service()
        
        # Create company bucket immediately
        bucket_result = company_bucket_service.create_company_bucket(request.name)
        
        if not bucket_result.get('success'):
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create company bucket: {bucket_result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"✅ Company and bucket created successfully: {request.name}")
        
        return CompanyCreateResponse(
            success=True,
            company_name=request.name,
            bucket_name=bucket_result['bucket_name'],
            message=f"Company '{request.name}' created with isolated GCS bucket"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Company creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/company/{company_name}/bucket-info")
async def get_company_bucket_info(company_name: str):
    """
    Get information about a company's GCS bucket and QuDemos
    """
    try:
        if not get_company_bucket_service():
            if not initialize_company_bucket_service():
                raise HTTPException(status_code=500, detail="Company bucket service not initialized")
        
        company_bucket_service = get_company_bucket_service()
        bucket_info = company_bucket_service.get_company_bucket_info(company_name)
        
        if not bucket_info.get('success'):
            raise HTTPException(
                status_code=404, 
                detail=f"Company bucket not found: {bucket_info.get('error', 'Unknown error')}"
            )
        
        return bucket_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get bucket info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/company/{company_name}/qudemo/{qudemo_id}/create-folder")
async def create_qudemo_folder(company_name: str, qudemo_id: str):
    """
    Create a QuDemo folder inside the company's bucket
    """
    try:
        if not get_company_bucket_service():
            if not initialize_company_bucket_service():
                raise HTTPException(status_code=500, detail="Company bucket service not initialized")
        
        company_bucket_service = get_company_bucket_service()
        folder_result = company_bucket_service.create_qudemo_folder(company_name, qudemo_id)
        
        if not folder_result.get('success'):
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create QuDemo folder: {folder_result.get('error', 'Unknown error')}"
            )
        
        return folder_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create QuDemo folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/companies")
async def list_all_companies():
    """
    List all companies with their bucket information
    """
    try:
        if not get_company_bucket_service():
            if not initialize_company_bucket_service():
                raise HTTPException(status_code=500, detail="Company bucket service not initialized")
        
        company_bucket_service = get_company_bucket_service()
        companies_result = company_bucket_service.list_all_companies()
        
        if not companies_result.get('success'):
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to list companies: {companies_result.get('error', 'Unknown error')}"
            )
        
        return companies_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to list companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage-structure")
async def get_storage_structure():
    """
    Get the complete GCS storage structure for debugging
    """
    try:
        if not get_company_bucket_service():
            if not initialize_company_bucket_service():
                raise HTTPException(status_code=500, detail="Company bucket service not initialized")
        
        company_bucket_service = get_company_bucket_service()
        structure = company_bucket_service.gcs_service.get_storage_structure()
        
        return {
            "success": True,
            "storage_structure": structure,
            "description": "Complete GCS storage structure showing company buckets and QuDemo folders"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get storage structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))
