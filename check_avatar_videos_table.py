"""Check if videos are in Supabase avatar_videos table"""
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
print("🔍 CHECKING SUPABASE avatar_videos TABLE")
print("=" * 80)
print(f"QuDemo ID: {DEMO_QUDEMO_ID}")
print()

# Query avatar_videos table
result = supabase.table('avatar_videos').select('*').eq(
    'qudemo_id', DEMO_QUDEMO_ID
).execute()

if result.data:
    print(f"✅ Found {len(result.data)} videos in Supabase")
    print()
    
    loom_videos = [v for v in result.data if 'kudomo' in v.get('faq_id', '').lower()]
    
    if loom_videos:
        print(f"📹 Loom videos (faq_kudomo_*):")
        print("=" * 80)
        for video in loom_videos:
            print(f"\nFAQ ID: {video['faq_id']}")
            print(f"Status: {video['status']}")
            print(f"Video URL: {video['video_url'][:80]}...")
            print(f"HeyGen ID: {video.get('heygen_video_id', 'N/A')}")
    else:
        print("❌ NO Loom videos found (faq_kudomo_*)")
        print("\n📋 All videos in table:")
        for video in result.data[:5]:
            print(f"   - {video['faq_id']}")
else:
    print("❌ NO videos found in avatar_videos table for this QuDemo!")

print("\n" + "=" * 80)

