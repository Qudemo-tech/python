"""
Replace old Loom videos with 2 new ones
Keep: HeyGen AI avatar videos (if any exist)
Delete: 7 old Loom videos (faq_kudomo_001 to 007)
Add: 2 new Loom videos (create_qudemo, view_interactions)
"""

import os
import json
from dotenv import load_dotenv
from google.cloud import storage
from supabase import create_client

load_dotenv()

# Configuration
DEMO_QUDEMO_ID = "48b29bfb-b290-4669-9f25-ee411cdb1d9d"
COMPANY_NAME = "Sample Qudemo"
GCS_BUCKET = "qudemo-sample-qudemo"

# Initialize services
storage_client = storage.Client.from_service_account_json('service-account-key.json')
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)

# New Loom videos
NEW_VIDEOS = [
    {
        "local_file": "loom_qudemo_segments/create_qudemo.mp4",
        "faq_id": "faq_loom_create_qudemo",
        "question": "How do I create a Qudemo?",
        "answer": "Creating a Qudemo is simple and quick! First, log into your account and navigate to the main dashboard. Click on the 'Create New Qudemo' button. You'll then be guided through a step-by-step process where you can upload your demo video, add relevant documents or website links for context, and configure your settings. The system will process your content, generate FAQs, and create an interactive demo experience. The entire setup takes just a few minutes, and you'll have a fully functional interactive demo ready to share with your customers.",
        "category": "tutorial"
    },
    {
        "local_file": "loom_qudemo_segments/view_interactions.mp4",
        "faq_id": "faq_loom_view_interactions",
        "question": "How can I view customer interactions?",
        "answer": "Viewing customer interactions is easy through the Interactions dashboard. From your QuDemo list, click on the 'View Interactions' button for any QuDemo. You'll see a comprehensive overview showing all visitors who have engaged with your demo. The dashboard displays visitor details (name, email, company), questions they asked, time spent, and their full interaction history. You can click on individual users to see their complete conversation timeline, helping you understand their interests and follow up effectively. This feature gives you valuable insights into customer engagement and sales opportunities.",
        "category": "feature"
    }
]

# Old Loom video IDs to delete
OLD_LOOM_FAQ_IDS = [
    "faq_kudomo_001",
    "faq_kudomo_002",
    "faq_kudomo_003",
    "faq_kudomo_004",
    "faq_kudomo_005",
    "faq_kudomo_006",
    "faq_kudomo_007"
]

print("=" * 80)
print("🔄 REPLACING LOOM VIDEOS")
print("=" * 80)
print(f"Demo QuDemo ID: {DEMO_QUDEMO_ID}")
print(f"Company: {COMPANY_NAME}")
print("=" * 80)
print()

def delete_old_loom_videos_from_gcs():
    """Delete old Loom videos from GCS"""
    print("🗑️  STEP 1: Deleting old Loom videos from GCS...")
    
    bucket = storage_client.bucket(GCS_BUCKET)
    base_path = f"{COMPANY_NAME}/{DEMO_QUDEMO_ID}/avatar_videos"
    
    deleted_count = 0
    for faq_id in OLD_LOOM_FAQ_IDS:
        blob_path = f"{base_path}/{faq_id}.mp4"
        blob = bucket.blob(blob_path)
        
        if blob.exists():
            blob.delete()
            print(f"   ✅ Deleted: {faq_id}.mp4")
            deleted_count += 1
        else:
            print(f"   ⚠️  Not found: {faq_id}.mp4")
    
    print(f"\n   Total deleted: {deleted_count}/{len(OLD_LOOM_FAQ_IDS)}")
    return deleted_count


def upload_new_loom_videos_to_gcs():
    """Upload new Loom videos to GCS"""
    print("\n📤 STEP 2: Uploading new Loom videos to GCS...")
    
    bucket = storage_client.bucket(GCS_BUCKET)
    base_path = f"{COMPANY_NAME}/{DEMO_QUDEMO_ID}/avatar_videos"
    
    uploaded_urls = {}
    
    for video in NEW_VIDEOS:
        local_path = video['local_file']
        faq_id = video['faq_id']
        gcs_path = f"{base_path}/{faq_id}.mp4"
        
        if not os.path.exists(local_path):
            print(f"   ❌ Local file not found: {local_path}")
            continue
        
        # Upload to GCS
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        blob.make_public()
        
        video_url = blob.public_url
        uploaded_urls[faq_id] = video_url
        
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"   ✅ Uploaded: {faq_id}.mp4 ({size_mb:.2f} MB)")
        print(f"      URL: {video_url[:70]}...")
    
    print(f"\n   Total uploaded: {len(uploaded_urls)}/{len(NEW_VIDEOS)}")
    return uploaded_urls


def update_faq_file(uploaded_urls):
    """Update FAQ file in GCS"""
    print("\n📝 STEP 3: Updating FAQ file...")
    
    bucket = storage_client.bucket(GCS_BUCKET)
    faq_blob_path = f"{COMPANY_NAME}/{DEMO_QUDEMO_ID}/faqs_{COMPANY_NAME.replace(' ', '_')}.json"
    blob = bucket.blob(faq_blob_path)
    
    # Download existing FAQs
    blob.reload()
    faq_data = json.loads(blob.download_as_text())
    
    print(f"   📥 Current FAQs: {len(faq_data.get('faqs', []))}")
    
    # Remove old Loom FAQs
    original_count = len(faq_data.get('faqs', []))
    faq_data['faqs'] = [
        faq for faq in faq_data.get('faqs', [])
        if faq.get('id') not in OLD_LOOM_FAQ_IDS
    ]
    removed_count = original_count - len(faq_data['faqs'])
    print(f"   🗑️  Removed {removed_count} old Loom FAQs")
    
    # Add new Loom FAQs
    for video in NEW_VIDEOS:
        faq_id = video['faq_id']
        
        if faq_id in uploaded_urls:
            new_faq = {
                "id": faq_id,
                "question": video['question'],
                "answer": video['answer'],
                "category": video['category'],
                "source": "loom_video",
                "video_url": uploaded_urls[faq_id],
                "has_avatar_video": True,
                "video_status": "completed"
            }
            
            faq_data['faqs'].append(new_faq)
            print(f"   ✅ Added new FAQ: {faq_id}")
    
    # Update metadata
    from datetime import datetime
    faq_data['updated_at'] = datetime.now().isoformat()
    
    # Upload updated FAQs
    fixed_json = json.dumps(faq_data, indent=2, ensure_ascii=False)
    blob.upload_from_string(fixed_json, content_type='application/json')
    
    print(f"   📊 Final FAQ count: {len(faq_data['faqs'])}")
    
    # Show breakdown
    loom_faqs = [f for f in faq_data['faqs'] if 'loom' in f.get('id', '').lower()]
    ai_faqs = [f for f in faq_data['faqs'] if 'intro' in f.get('id', '').lower() or 'fallback' in f.get('id', '').lower() or 'collection' in f.get('id', '').lower()]
    
    print(f"      - Loom videos: {len(loom_faqs)}")
    print(f"      - AI avatar videos: {len(ai_faqs)}")
    print(f"      - Other: {len(faq_data['faqs']) - len(loom_faqs) - len(ai_faqs)}")
    
    return faq_data


def update_supabase_avatar_videos(uploaded_urls):
    """Update Supabase avatar_videos table"""
    print("\n💾 STEP 4: Updating Supabase database...")
    
    # Delete old Loom video records
    for faq_id in OLD_LOOM_FAQ_IDS:
        try:
            supabase.table('avatar_videos').delete().eq(
                'qudemo_id', DEMO_QUDEMO_ID
            ).eq(
                'faq_id', faq_id
            ).execute()
            print(f"   🗑️  Deleted DB record: {faq_id}")
        except Exception as e:
            print(f"   ⚠️  Error deleting {faq_id}: {e}")
    
    # Add new Loom video records
    for video in NEW_VIDEOS:
        faq_id = video['faq_id']
        
        if faq_id in uploaded_urls:
            try:
                supabase.table('avatar_videos').insert({
                    'qudemo_id': DEMO_QUDEMO_ID,
                    'faq_id': faq_id,
                    'status': 'completed',
                    'video_url': uploaded_urls[faq_id],
                    'heygen_video_id': f"loom_{faq_id}",
                    'created_at': 'now()',
                    'updated_at': 'now()'
                }).execute()
                
                print(f"   ✅ Added DB record: {faq_id}")
            except Exception as e:
                print(f"   ❌ Error adding {faq_id}: {e}")
    
    print(f"\n   Total DB records updated")


def main():
    """Main execution"""
    
    # Step 1: Delete old Loom videos from GCS
    delete_old_loom_videos_from_gcs()
    
    # Step 2: Upload new Loom videos to GCS
    uploaded_urls = upload_new_loom_videos_to_gcs()
    
    if not uploaded_urls:
        print("\n❌ No videos uploaded! Exiting.")
        return
    
    # Step 3: Update FAQ file
    updated_faq_data = update_faq_file(uploaded_urls)
    
    # Step 4: Update Supabase
    update_supabase_avatar_videos(uploaded_urls)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ REPLACEMENT COMPLETE!")
    print("=" * 80)
    print("\n📊 SUMMARY:")
    print(f"   - Old Loom videos deleted: {len(OLD_LOOM_FAQ_IDS)}")
    print(f"   - New Loom videos added: {len(uploaded_urls)}")
    print(f"   - Total FAQs in demo: {len(updated_faq_data['faqs'])}")
    print()
    print("📋 NEW LOOM VIDEO QUESTIONS:")
    for i, video in enumerate(NEW_VIDEOS, 1):
        print(f"   {i}. {video['question']}")
    print("=" * 80)
    print("\n⚠️  IMPORTANT: RESTART THE PYTHON BACKEND to load the updated FAQ file!")
    print("=" * 80)


if __name__ == "__main__":
    main()

