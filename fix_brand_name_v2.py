"""
Fix brand name from Kudomo to Qudemo - ROBUST VERSION
"""

import json
from google.cloud import storage

# Initialize GCS
storage_client = storage.Client.from_service_account_json('service-account-key.json')
bucket = storage_client.bucket('qudemo-sample-qudemo')
blob_path = 'Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json'
blob = bucket.blob(blob_path)

print("=" * 80)
print("🔧 FIXING BRAND NAME: Kudomo → Qudemo (ROBUST VERSION)")
print("=" * 80)

# Step 1: Download current file
print("\n📥 Step 1: Downloading current FAQ file from GCS...")
current_text = blob.download_as_text()
print(f"   File size: {len(current_text)} bytes")
print(f"   'Kudomo' count BEFORE: {current_text.count('Kudomo')}")

# Step 2: Parse JSON
print("\n📝 Step 2: Parsing JSON...")
faq_data = json.loads(current_text)
print(f"   Total FAQs: {len(faq_data['faqs'])}")

# Step 3: Fix all questions and answers
print("\n🔧 Step 3: Fixing brand names...")
fixed_count = 0

for i, faq in enumerate(faq_data['faqs']):
    before_q = faq['question']
    before_a = faq['answer']
    
    # Replace Kudomo with Qudemo
    faq['question'] = faq['question'].replace('Kudomo', 'Qudemo').replace('kudomo', 'qudemo')
    faq['answer'] = faq['answer'].replace('Kudomo', 'Qudemo').replace('kudomo', 'qudemo')
    
    if faq['question'] != before_q or faq['answer'] != before_a:
        fixed_count += 1
        print(f"   ✅ Fixed FAQ {i+1}: {faq['question'][:60]}...")

print(f"\n   Total fixed: {fixed_count} FAQs")

# Step 4: Convert back to JSON
print("\n📦 Step 4: Converting to JSON...")
fixed_json = json.dumps(faq_data, indent=2, ensure_ascii=False)
print(f"   New file size: {len(fixed_json)} bytes")
print(f"   'Qudemo' count AFTER: {fixed_json.count('Qudemo')}")
print(f"   'Kudomo' count AFTER: {fixed_json.count('Kudomo')}")

# Step 5: Upload to GCS
print("\n📤 Step 5: Uploading fixed file to GCS...")
blob.upload_from_string(fixed_json, content_type='application/json')
print("   ✅ Upload complete!")

# Step 6: Verify upload
print("\n✅ Step 6: Verifying upload...")
verification_text = blob.download_as_text()
verification_data = json.loads(verification_text)

print(f"   'Kudomo' count in uploaded file: {verification_text.count('Kudomo')}")
print(f"   'Qudemo' count in uploaded file: {verification_text.count('Qudemo')}")

if verification_text.count('Kudomo') == 0:
    print("\n" + "=" * 80)
    print("✅✅✅ SUCCESS! Brand name fix complete!")
    print("=" * 80)
    print("\n📋 Sample questions:")
    for i, faq in enumerate(verification_data['faqs'][:3], 1):
        print(f"   {i}. {faq['question']}")
    print("=" * 80)
else:
    print("\n❌ ERROR: File still contains 'Kudomo'!")

