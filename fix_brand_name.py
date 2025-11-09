"""
Fix brand name from Kudomo to Qudemo in all Q&A
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
QA_JSON_FILE = "loom_test_output/questions_answers.json"
GCS_BUCKET = "qudemo-sample-qudemo"

# Initialize services
storage_client = storage.Client.from_service_account_json('service-account-key.json')
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)


def fix_brand_name_in_text(text):
    """Replace Kudomo with Qudemo"""
    return text.replace('Kudomo', 'Qudemo').replace('kudomo', 'qudemo')


def fix_local_qa_file():
    """Fix the local Q&A JSON file"""
    print("\n📝 Fixing local Q&A file...")
    
    with open(QA_JSON_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # Fix all questions and answers
    for topic in qa_data['topics']:
        topic['question'] = fix_brand_name_in_text(topic['question'])
        topic['answer'] = fix_brand_name_in_text(topic['answer'])
    
    # Save fixed file
    with open(QA_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Fixed {len(qa_data['topics'])} Q&A pairs in local file")
    
    return qa_data


def update_gcs_faqs():
    """Update FAQs in GCS with corrected brand name"""
    print("\n🔄 Updating GCS FAQ file...")
    
    bucket = storage_client.bucket(GCS_BUCKET)
    faq_blob_path = f"{COMPANY_NAME}/{DEMO_QUDEMO_ID}/faqs_{COMPANY_NAME.replace(' ', '_')}.json"
    blob = bucket.blob(faq_blob_path)
    
    # Download existing FAQs
    faq_data = json.loads(blob.download_as_text())
    
    # Fix all FAQ questions and answers
    fixed_count = 0
    for faq in faq_data['faqs']:
        if 'Kudomo' in faq['question'] or 'Kudomo' in faq['answer']:
            faq['question'] = fix_brand_name_in_text(faq['question'])
            faq['answer'] = fix_brand_name_in_text(faq['answer'])
            fixed_count += 1
    
    # Upload updated FAQs
    blob.upload_from_string(
        json.dumps(faq_data, indent=2, ensure_ascii=False),
        content_type='application/json'
    )
    
    print(f"✅ Fixed {fixed_count} FAQs in GCS")
    
    return faq_data


def main():
    """Main execution"""
    print("=" * 70)
    print("🔧 FIXING BRAND NAME: Kudomo → Qudemo")
    print("=" * 70)
    print(f"Demo QuDemo ID: {DEMO_QUDEMO_ID}")
    print(f"Company: {COMPANY_NAME}")
    print("=" * 70)
    
    # Step 1: Fix local Q&A file
    qa_data = fix_local_qa_file()
    
    # Step 2: Update GCS FAQs
    faq_data = update_gcs_faqs()
    
    print("\n" + "=" * 70)
    print("✅ BRAND NAME FIX COMPLETE!")
    print("=" * 70)
    print("\n📋 Updated Questions:")
    for i, topic in enumerate(qa_data['topics'], 1):
        print(f"{i}. {topic['question']}")
    print("=" * 70)
    print("\n✅ All references to 'Kudomo' have been changed to 'Qudemo'!")
    print("✅ The widget will now show the correct brand name!")
    print("=" * 70)


if __name__ == "__main__":
    main()

