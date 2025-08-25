import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def add_video_to_database():
    """Add video data to the qudemo_videos table"""
    
    # Supabase configuration
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase configuration")
        return False
    
    # Video data to add
    video_data = {
        "qudemo_id": "0293bfc6-645f-4da7-a1cc-3b30cf6f1691",
        "video_url": "https://youtu.be/hwko23YbAHs?si=LKWZbN1v4RNK__BS",
        "video_type": "youtube",
        "title": "Recurring Payments Setup",
        "description": "How to set up recurring payments in the system",
        "order_index": 1,
        "metadata": {
            "source": "manual_fix",
            "processing_method": "gemini",
            "chunks_created": 7
        }
    }
    
    try:
        # Insert into qudemo_videos table
        response = requests.post(
            f"{supabase_url}/rest/v1/qudemo_videos",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json=video_data
        )
        
        if response.status_code == 201:
            print("✅ Video data added to database successfully!")
            return True
        else:
            print(f"❌ Failed to add video data: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error adding video data: {e}")
        return False

def check_video_data():
    """Check if video data exists in the database"""
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase configuration")
        return False
    
    try:
        # Query qudemo_videos table
        response = requests.get(
            f"{supabase_url}/rest/v1/qudemo_videos?qudemo_id=eq.0293bfc6-645f-4da7-a1cc-3b30cf6f1691",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}"
            }
        )
        
        if response.status_code == 200:
            videos = response.json()
            print(f"📊 Found {len(videos)} videos for qudemo")
            for video in videos:
                print(f"  - {video.get('title', 'Untitled')}: {video.get('video_url', 'No URL')}")
            return len(videos) > 0
        else:
            print(f"❌ Failed to check video data: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking video data: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking current video data...")
    has_videos = check_video_data()
    
    if not has_videos:
        print("\n📝 Adding video data to database...")
        success = add_video_to_database()
        
        if success:
            print("\n🔍 Verifying video data was added...")
            check_video_data()
    else:
        print("✅ Video data already exists in database")
