"""Add video_status field to all Loom FAQs"""
import json
from google.cloud import storage

# Initialize GCS
storage_client = storage.Client.from_service_account_json('service-account-key.json')
bucket = storage_client.bucket('qudemo-sample-qudemo')
blob_path = 'Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json'
blob = bucket.blob(blob_path)

print("=" * 80)
print("🔧 ADDING video_status TO ALL FAQs")
print("=" * 80)

# Download and parse
faq_data = json.loads(blob.download_as_text())
print(f"Total FAQs: {len(faq_data['faqs'])}")

# Add video_status to all FAQs that have video_url
updated_count = 0
for faq in faq_data['faqs']:
    if faq.get('video_url') and faq.get('has_avatar_video'):
        # Add video_status if missing
        if 'video_status' not in faq:
            faq['video_status'] = 'completed'
            updated_count += 1
            print(f"✅ Added video_status to: {faq['id']}")

# Upload fixed file
fixed_json = json.dumps(faq_data, indent=2, ensure_ascii=False)
blob.upload_from_string(fixed_json, content_type='application/json')

print(f"\n📤 Updated {updated_count} FAQs with video_status")
print(f"✅ File uploaded successfully!")
print("=" * 80)

# Verify
verify_text = blob.download_as_text()
verify_data = json.loads(verify_text)

print("\n🔍 VERIFICATION:")
for faq in verify_data['faqs'][:3]:
    print(f"\nFAQ ID: {faq['id']}")
    print(f"  has_avatar_video: {faq.get('has_avatar_video')}")
    print(f"  video_url: {'YES' if faq.get('video_url') else 'NO'}")
    print(f"  video_status: {faq.get('video_status', 'MISSING')}")

print("\n=" * 80)
print("✅ ALL DONE! Videos should now play in the widget!")
print("=" * 80)

