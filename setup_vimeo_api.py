#!/usr/bin/env python3
"""
Vimeo API Setup Helper Script
This script helps you configure the Vimeo API key for video processing.
"""

import os
import sys
from dotenv import load_dotenv

def check_vimeo_api_key():
    """Check if Vimeo API key is configured"""
    load_dotenv()
    
    api_key = os.getenv('VIMEO_API_KEY')
    
    if api_key and api_key != 'your_vimeo_api_key_here':
        print("✅ Vimeo API key is configured!")
        print(f"   Key: {api_key[:10]}...{api_key[-4:]}")
        return True
    else:
        print("❌ Vimeo API key is not configured")
        print("   Please follow the setup guide in VIMEO_API_SETUP.md")
        return False

def test_vimeo_api_connection():
    """Test the Vimeo API connection"""
    import requests
    
    api_key = os.getenv('VIMEO_API_KEY')
    if not api_key or api_key == 'your_vimeo_api_key_here':
        print("❌ Cannot test API connection - no API key configured")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Test with a public Vimeo video (Vimeo's own intro video)
        test_video_id = "76979871"  # Vimeo's "What is Vimeo?" video
        url = f"https://api.vimeo.com/videos/{test_video_id}"
        
        print("🔍 Testing Vimeo API connection...")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            video_name = data.get('name', 'Unknown')
            print(f"✅ API connection successful!")
            print(f"   Test video: {video_name}")
            print(f"   Video ID: {test_video_id}")
            return True
        else:
            print(f"❌ API connection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False

def show_setup_instructions():
    """Show setup instructions"""
    print("\n" + "="*60)
    print("🔑 VIMEO API SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Go to: https://developer.vimeo.com/")
    print("2. Sign in with your Vimeo account")
    print("3. Click 'Create App' or 'New App'")
    print("4. Fill in app details:")
    print("   - App Name: QuDemo Video Processor")
    print("   - App Description: Video processing for QuDemo platform")
    print("   - App Category: Other or Business")
    print("5. Go to 'Authentication' tab")
    print("6. Click 'Generate Access Token'")
    print("7. Select scopes: public, private, video_files, video_metadata")
    print("8. Copy the generated token")
    print("9. Add to your .env file: VIMEO_API_KEY=your_token_here")
    print("="*60)

def main():
    """Main function"""
    print("🎬 Vimeo API Setup Helper")
    print("-" * 30)
    
    # Check if API key is configured
    has_key = check_vimeo_api_key()
    
    if has_key:
        # Test the connection
        test_vimeo_api_connection()
    else:
        # Show setup instructions
        show_setup_instructions()
    
    print("\n📚 For detailed instructions, see: VIMEO_API_SETUP.md")

if __name__ == "__main__":
    main() 