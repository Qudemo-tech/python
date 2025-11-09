"""List ALL questions - Static, Loom videos, and AI avatar videos"""
import json
from google.cloud import storage
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Initialize
client = storage.Client.from_service_account_json('service-account-key.json')
bucket = client.bucket('qudemo-sample-qudemo')
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)

DEMO_QUDEMO_ID = "48b29bfb-b290-4669-9f25-ee411cdb1d9d"

print("=" * 80)
print("📋 ALL QUESTIONS FOR DEMO QUDEMO")
print("=" * 80)
print()

# 1. STATIC QUESTIONS (hardcoded in widget)
print("1️⃣ STATIC QUESTIONS (Hardcoded in Widget):")
print("=" * 80)
static_questions = [
    "What is Qudemo?",
    "How does Qudemo work?",
    "Who is Qudemo for?",
    "How is it different from normal demo videos?",
    "Does it help sales teams too?",
    "What kind of videos can I use with Qudemo?"
]

for i, q in enumerate(static_questions, 1):
    print(f"{i}. {q}")
    print(f"   Type: ❌ NO VIDEO (text-only answer)")
    print()

print(f"Total static questions: {len(static_questions)}")
print()

# 2. LOOM VIDEO QUESTIONS
print("=" * 80)
print("2️⃣ LOOM VIDEO QUESTIONS (Real screen recordings):")
print("=" * 80)

blob = bucket.blob('Sample Qudemo/48b29bfb-b290-4669-9f25-ee411cdb1d9d/faqs_Sample_Qudemo.json')
blob.reload()
faq_data = json.loads(blob.download_as_text())

loom_questions = []
ai_questions = []
other_questions = []

for faq in faq_data['faqs']:
    faq_id = faq.get('id', '')
    question = faq.get('question', '')
    has_video = faq.get('has_avatar_video', False)
    video_url = faq.get('video_url', '')
    
    if 'kudomo' in faq_id.lower():
        loom_questions.append({
            'id': faq_id,
            'question': question,
            'has_video': has_video,
            'video_url': video_url
        })
    elif 'intro' in faq_id.lower() or 'fallback' in faq_id.lower() or 'collection' in faq_id.lower():
        ai_questions.append({
            'id': faq_id,
            'question': question,
            'has_video': has_video,
            'video_url': video_url
        })
    else:
        other_questions.append({
            'id': faq_id,
            'question': question,
            'has_video': has_video,
            'video_url': video_url
        })

for i, loom in enumerate(loom_questions, 1):
    print(f"{i}. {loom['question']}")
    print(f"   FAQ ID: {loom['id']}")
    print(f"   Type: 🎬 LOOM VIDEO (Real screen recording)")
    print(f"   Has Video: {'✅ YES' if loom['has_video'] else '❌ NO'}")
    if loom['video_url']:
        print(f"   Video URL: {loom['video_url'][:70]}...")
    print()

print(f"Total Loom video questions: {len(loom_questions)}")
print()

# 3. AI AVATAR VIDEOS (HeyGen)
if ai_questions:
    print("=" * 80)
    print("3️⃣ AI AVATAR VIDEO QUESTIONS (HeyGen AI-generated):")
    print("=" * 80)
    
    for i, ai in enumerate(ai_questions, 1):
        print(f"{i}. {ai['question']}")
        print(f"   FAQ ID: {ai['id']}")
        print(f"   Type: 🤖 AI AVATAR (HeyGen generated)")
        print(f"   Has Video: {'✅ YES' if ai['has_video'] else '❌ NO'}")
        if ai['video_url']:
            print(f"   Video URL: {ai['video_url'][:70]}...")
        print()
    
    print(f"Total AI avatar questions: {len(ai_questions)}")
    print()

# 4. OTHER QUESTIONS
if other_questions:
    print("=" * 80)
    print("4️⃣ OTHER QUESTIONS:")
    print("=" * 80)
    
    for i, other in enumerate(other_questions, 1):
        print(f"{i}. {other['question']}")
        print(f"   FAQ ID: {other['id']}")
        print(f"   Has Video: {'✅ YES' if other['has_video'] else '❌ NO'}")
        print()
    
    print(f"Total other questions: {len(other_questions)}")
    print()

# SUMMARY
print("=" * 80)
print("📊 SUMMARY:")
print("=" * 80)
print(f"Static questions (no video):     {len(static_questions)}")
print(f"Loom video questions:             {len(loom_questions)}")
print(f"AI avatar video questions:        {len(ai_questions)}")
print(f"Other questions:                  {len(other_questions)}")
print(f"─" * 80)
print(f"TOTAL QUESTIONS:                  {len(static_questions) + len(loom_questions) + len(ai_questions) + len(other_questions)}")
print("=" * 80)

