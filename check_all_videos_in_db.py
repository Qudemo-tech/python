"""Check all videos in Supabase for the demo QuDemo"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)

DEMO_QUDEMO_ID = "48b29bfb-b290-4669-9f25-ee411cdb1d9d"

print("=" * 80)
print("🔍 ALL VIDEOS IN DATABASE FOR DEMO QUDEMO")
print("=" * 80)
print(f"QuDemo ID: {DEMO_QUDEMO_ID}")
print("=" * 80)
print()

# Query all videos for this QuDemo
result = supabase.table('avatar_videos').select('*').eq(
    'qudemo_id', DEMO_QUDEMO_ID
).order('created_at').execute()

if result.data:
    print(f"✅ Found {len(result.data)} total videos\n")
    
    loom_videos = []
    other_videos = []
    
    for video in result.data:
        if 'kudomo' in video['faq_id'].lower():
            loom_videos.append(video)
        else:
            other_videos.append(video)
    
    if loom_videos:
        print(f"📹 LOOM VIDEOS ({len(loom_videos)}):")
        print("=" * 80)
        for v in loom_videos:
            print(f"FAQ ID: {v['faq_id']}")
            print(f"Status: {v['status']}")
            print(f"Video URL: {v['video_url'][:70]}...")
            print()
    
    if other_videos:
        print(f"🎬 OTHER AI AVATAR VIDEOS ({len(other_videos)}):")
        print("=" * 80)
        for v in other_videos:
            print(f"FAQ ID: {v['faq_id']}")
            print(f"Status: {v['status']}")
            print(f"Video URL: {v.get('video_url', 'NO URL')[:70]}...")
            print()
else:
    print("❌ No videos found in database")

print("=" * 80)

