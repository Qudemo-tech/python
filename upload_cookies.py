#!/usr/bin/env python3
"""
Upload fresh YouTube cookies to Supabase Storage
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def upload_cookies():
    """Upload cookies.txt to Supabase Storage"""
    try:
        # Initialize Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        
        # Check if cookies.txt exists
        cookies_file = "cookies.txt"
        if not os.path.exists(cookies_file):
            print("❌ cookies.txt not found!")
            print("📋 Please export fresh YouTube cookies and save as 'cookies.txt'")
            return False
        
        # Upload to Supabase Storage
        bucket_name = "cookies"
        file_name = "www.youtube.com_cookies.txt"
        
        with open(cookies_file, 'rb') as f:
            file_data = f.read()
        
        # First try to remove existing file
        try:
            supabase.storage.from_(bucket_name).remove([file_name])
            print(f"🗑️ Removed existing file: {file_name}")
        except Exception as e:
            print(f"ℹ️ No existing file to remove: {e}")
        
        # Upload the file
        result = supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=file_data,
            file_options={"content-type": "text/plain"}
        )
        
        print(f"✅ Successfully uploaded {cookies_file} to Supabase Storage")
        print(f"📁 Bucket: {bucket_name}")
        print(f"📄 File: {file_name}")
        print(f"📊 File size: {len(file_data)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading cookies: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Uploading fresh YouTube cookies to Supabase...")
    success = upload_cookies()
    
    if success:
        print("\n🎉 Cookies uploaded successfully!")
        print("🔄 Your video processing should now work with fresh cookies.")
    else:
        print("\n❌ Failed to upload cookies. Please check the error above.") 