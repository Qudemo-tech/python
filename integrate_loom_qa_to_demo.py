"""
Integrate Loom Q&A into Static Demo QuDemo
Uploads video snippets and adds Q&A to the demo QuDemo
"""

import os
import json
from dotenv import load_dotenv
from google.cloud import storage
from supabase import create_client

load_dotenv()

# Configuration
DEMO_QUDEMO_ID = "48b29bfb-b290-4669-9f25-ee411cdb1d9d"  # Static demo QuDemo ID
COMPANY_NAME = "Sample Qudemo"
QA_JSON_FILE = "loom_test_output/questions_answers.json"
SNIPPETS_DIR = "loom_test_output/video_snippets"
GCS_BUCKET = "qudemo-sample-qudemo"

# Initialize services
storage_client = storage.Client.from_service_account_json('service-account-key.json')
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)


def upload_snippets_to_gcs():
    """Upload video snippets to GCS"""
    print("\n📤 Uploading video snippets to GCS...")
    
    bucket = storage_client.bucket(GCS_BUCKET)
    base_path = f"{COMPANY_NAME}/{DEMO_QUDEMO_ID}/avatar_videos"
    
    uploaded_urls = {}
    
    # Get all snippet files
    snippet_files = [f for f in os.listdir(SNIPPETS_DIR) if f.endswith('.mp4')]
    
    for snippet_file in snippet_files:
        local_path = os.path.join(SNIPPETS_DIR, snippet_file)
        
        # Create unique filename (e.g., faq_kudomo_001.mp4)
        faq_id = f"faq_kudomo_{snippet_file.replace('snippet_', '').replace('.mp4', '').zfill(3)}"
        gcs_path = f"{base_path}/{faq_id}.mp4"
        
        # Upload to GCS
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        blob.make_public()
        
        video_url = blob.public_url
        uploaded_urls[snippet_file] = {
            'faq_id': faq_id,
            'video_url': video_url
        }
        
        print(f"  ✅ Uploaded: {snippet_file} → {faq_id}")
    
    return uploaded_urls


def create_faq_list(qa_data, uploaded_urls):
    """Create FAQ list from Q&A data"""
    print("\n📝 Creating FAQ list...")
    
    faqs = []
    
    for topic in qa_data['topics']:
        snippet_file = topic['snippet_filename']
        
        if snippet_file in uploaded_urls:
            faq_id = uploaded_urls[snippet_file]['faq_id']
            video_url = uploaded_urls[snippet_file]['video_url']
            
            faq = {
                "id": faq_id,
                "question": topic['question'],
                "answer": topic['answer'],
                "category": topic['category'].lower(),
                "source": "loom_video",
                "estimated_duration": float(topic['end_time'].split(':')[0]) * 60 + float(topic['end_time'].split(':')[1]) - (float(topic['start_time'].split(':')[0]) * 60 + float(topic['start_time'].split(':')[1])),
                "video_url": video_url,
                "has_avatar_video": True
            }
            
            faqs.append(faq)
            print(f"  ✅ FAQ created: {faq_id} - {topic['question'][:50]}...")
    
    return faqs


def update_demo_qudemo_faqs(faqs):
    """Update demo QuDemo's FAQ file in GCS"""
    print("\n🔄 Updating demo QuDemo FAQ file...")
    
    bucket = storage_client.bucket(GCS_BUCKET)
    faq_blob_path = f"{COMPANY_NAME}/{DEMO_QUDEMO_ID}/faqs_{COMPANY_NAME.replace(' ', '_')}.json"
    blob = bucket.blob(faq_blob_path)
    
    # Download existing FAQs
    if blob.exists():
        existing_data = json.loads(blob.download_as_text())
        print(f"  📥 Downloaded existing FAQ file: {len(existing_data.get('faqs', []))} FAQs")
    else:
        existing_data = {
            "version": "1.0",
            "qudemo_id": DEMO_QUDEMO_ID,
            "company_name": COMPANY_NAME,
            "generated_at": "",
            "faqs": []
        }
    
    # Add new FAQs
    existing_faqs = existing_data.get('faqs', [])
    
    # Remove duplicates by ID
    existing_faq_ids = {faq['id'] for faq in existing_faqs}
    new_faqs = [faq for faq in faqs if faq['id'] not in existing_faq_ids]
    
    existing_data['faqs'].extend(new_faqs)
    
    # Upload updated FAQs
    blob.upload_from_string(
        json.dumps(existing_data, indent=2, ensure_ascii=False),
        content_type='application/json'
    )
    
    print(f"  ✅ Updated FAQ file: Added {len(new_faqs)} new FAQs")
    print(f"  📊 Total FAQs now: {len(existing_data['faqs'])}")
    
    return existing_data


def add_to_avatar_videos_table(faqs):
    """Add video records to avatar_videos table"""
    print("\n💾 Adding videos to Supabase...")
    
    for faq in faqs:
        try:
            # Check if already exists
            result = supabase.table('avatar_videos').select('*').eq(
                'qudemo_id', DEMO_QUDEMO_ID
            ).eq(
                'faq_id', faq['id']
            ).execute()
            
            if result.data:
                print(f"  ⚠️ Already exists: {faq['id']}")
                continue
            
            # Insert new record
            supabase.table('avatar_videos').insert({
                'qudemo_id': DEMO_QUDEMO_ID,
                'faq_id': faq['id'],
                'status': 'completed',
                'video_url': faq['video_url'],
                'heygen_video_id': f"loom_{faq['id']}",
                'created_at': 'now()',
                'updated_at': 'now()'
            }).execute()
            
            print(f"  ✅ Added to DB: {faq['id']}")
            
        except Exception as e:
            print(f"  ❌ Error adding {faq['id']}: {e}")


def create_suggested_questions(qa_data):
    """Create suggested questions list"""
    print("\n💡 Creating suggested questions...")
    
    suggested_questions = []
    
    for topic in qa_data['topics']:
        suggested_questions.append(topic['question'])
        print(f"  ✅ {topic['question']}")
    
    return suggested_questions


def main():
    """Main execution"""
    print("=" * 70)
    print("🎬 INTEGRATING LOOM Q&A INTO DEMO QUDEMO")
    print("=" * 70)
    print(f"Demo QuDemo ID: {DEMO_QUDEMO_ID}")
    print(f"Company: {COMPANY_NAME}")
    print(f"GCS Bucket: {GCS_BUCKET}")
    print("=" * 70)
    
    # Step 1: Load Q&A data
    print("\n📥 Loading Q&A data...")
    with open(QA_JSON_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"✅ Loaded {len(qa_data['topics'])} Q&A pairs")
    
    # Step 2: Upload video snippets to GCS
    uploaded_urls = upload_snippets_to_gcs()
    print(f"\n✅ Uploaded {len(uploaded_urls)} video snippets")
    
    # Step 3: Create FAQ list
    faqs = create_faq_list(qa_data, uploaded_urls)
    print(f"\n✅ Created {len(faqs)} FAQs")
    
    # Step 4: Update GCS FAQ file
    updated_faq_data = update_demo_qudemo_faqs(faqs)
    
    # Step 5: Add to avatar_videos table
    add_to_avatar_videos_table(faqs)
    
    # Step 6: Create suggested questions
    suggested_questions = create_suggested_questions(qa_data)
    
    # Save suggested questions to file
    with open('loom_test_output/suggested_questions.json', 'w', encoding='utf-8') as f:
        json.dump({
            'suggested_questions': suggested_questions,
            'count': len(suggested_questions)
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("✅ INTEGRATION COMPLETE!")
    print("=" * 70)
    print(f"📊 Summary:")
    print(f"   - Video snippets uploaded: {len(uploaded_urls)}")
    print(f"   - FAQs added: {len(faqs)}")
    print(f"   - Suggested questions: {len(suggested_questions)}")
    print(f"   - Total FAQs in demo: {len(updated_faq_data['faqs'])}")
    print("=" * 70)
    print("\n📋 Suggested Questions Added:")
    for i, q in enumerate(suggested_questions, 1):
        print(f"   {i}. {q}")
    print("=" * 70)
    print("\n🎯 Next Steps:")
    print("1. Suggested questions will appear in the widget")
    print("2. When clicked, the corresponding video snippet will play")
    print("3. Test the demo QuDemo to verify")
    print("=" * 70)


if __name__ == "__main__":
    main()

