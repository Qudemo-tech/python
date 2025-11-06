"""
Fix CORS configuration for GCS buckets to allow video playback from frontend
Run this script once to configure CORS for all company buckets
"""

import os
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def configure_bucket_cors(bucket_name):
    """Configure CORS for a GCS bucket to allow video playback"""
    
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Define CORS configuration
        cors_configuration = [
            {
                "origin": [
                    "http://localhost:3000",
                    "http://localhost:3001",
                    "https://qu-demo.vercel.app",
                    "https://qudemo.com",
                    "https://www.qudemo.com",
                    "https://qudemo-frontend.vercel.app",
                    "https://qudemo.vercel.app",
                    "https://testqudemo.netlify.app",
                    "*"  # Allow all origins (can restrict later)
                ],
                "method": ["GET", "HEAD", "OPTIONS"],
                "responseHeader": [
                    "Content-Type",
                    "Access-Control-Allow-Origin",
                    "Content-Length",
                    "Content-Range",
                    "Accept-Ranges"
                ],
                "maxAgeSeconds": 3600
            }
        ]
        
        # Apply CORS configuration
        bucket.cors = cors_configuration
        bucket.patch()
        
        print(f"✅ CORS configured for bucket: {bucket_name}")
        print(f"   Allowed origins: localhost:3000, localhost:3001, production domains, and all (*)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configuring CORS for {bucket_name}: {e}")
        return False

def make_bucket_public(bucket_name):
    """Make all files in bucket publicly readable"""
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Make bucket public
        policy = bucket.get_iam_policy(requested_policy_version=3)
        policy.bindings.append({
            "role": "roles/storage.objectViewer",
            "members": {"allUsers"}
        })
        bucket.set_iam_policy(policy)
        
        print(f"✅ Bucket {bucket_name} is now publicly readable")
        return True
        
    except Exception as e:
        print(f"❌ Error making bucket public: {e}")
        return False

def fix_all_buckets():
    """Fix CORS for all qudemo buckets"""
    
    try:
        storage_client = storage.Client()
        
        # List all buckets
        buckets = storage_client.list_buckets()
        
        qudemo_buckets = [bucket.name for bucket in buckets if bucket.name.startswith('qudemo-')]
        
        print(f"\n🔧 Found {len(qudemo_buckets)} QuDemo buckets")
        print("=" * 60)
        
        for bucket_name in qudemo_buckets:
            print(f"\n📦 Processing: {bucket_name}")
            
            # Configure CORS
            cors_success = configure_bucket_cors(bucket_name)
            
            # Make bucket public (optional - needed for video playback)
            public_success = make_bucket_public(bucket_name)
            
            if cors_success and public_success:
                print(f"✅ {bucket_name} - READY for video playback!")
            else:
                print(f"⚠️ {bucket_name} - Partial configuration")
        
        print("\n" + "=" * 60)
        print("✅ CORS configuration complete!")
        print("🎥 Videos should now play without CORS errors")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def fix_single_bucket(bucket_name):
    """Fix CORS for a specific bucket"""
    
    print(f"\n🔧 Fixing CORS for: {bucket_name}")
    print("=" * 60)
    
    cors_success = configure_bucket_cors(bucket_name)
    public_success = make_bucket_public(bucket_name)
    
    if cors_success and public_success:
        print(f"\n✅ SUCCESS! {bucket_name} is ready")
        print("🎥 Videos will now play without CORS errors")
    else:
        print(f"\n⚠️ Partial success - check errors above")

if __name__ == "__main__":
    import sys
    
    print("🚀 QuDemo CORS Fix Script")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Fix specific bucket
        bucket_name = sys.argv[1]
        fix_single_bucket(bucket_name)
    else:
        # Fix all buckets
        print("\n⚠️  This will configure CORS for ALL qudemo-* buckets")
        print("   Press Enter to continue, or Ctrl+C to cancel...")
        input()
        fix_all_buckets()
    
    print("\n✨ Done!")

