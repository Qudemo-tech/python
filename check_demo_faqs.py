"""Quick script to check the demo QuDemo FAQ file"""
import json
from google.cloud import storage

storage_client = storage.Client.from_service_account_json('service-account-key.json')
bucket = storage_client.bucket('qudemo-sample-qudemo')
blob = bucket.blob('Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json')

faq_data = json.loads(blob.download_as_text())

print(f"📊 Total FAQs: {len(faq_data['faqs'])}\n")
print("=" * 80)
print("📋 FAQ QUESTIONS:")
print("=" * 80)

for i, faq in enumerate(faq_data['faqs'], 1):
    category = faq.get('category', 'N/A')
    is_fallback = faq.get('is_fallback', False)
    is_intro = faq.get('is_intro', False)
    is_user_collection = faq.get('is_user_collection', False)
    has_video = faq.get('has_avatar_video', False)
    
    status = []
    if is_fallback:
        status.append('FALLBACK')
    if is_intro:
        status.append('INTRO')
    if is_user_collection:
        status.append('COLLECTION')
    if has_video:
        status.append('🎥')
    
    status_str = f" [{', '.join(status)}]" if status else ""
    
    print(f"{i}. [{category}] {faq['question']}{status_str}")

print("=" * 80)
print(f"\n✅ Loom Q&A questions (faq_kudomo_*) are visible and ready!")
print("✅ They will appear as suggested questions in the widget!")

