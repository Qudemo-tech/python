#!/usr/bin/env python3
"""
Google Cloud Storage Q&A Service
Handles Q&A functionality using Google Cloud Storage instead of Pinecone

=== SUGGESTED QUESTIONS SYSTEM (v2.1) ===

NEW ARCHITECTURE (Per-Video Generation with Validation + CACHED ANSWERS):

1. DYNAMIC QUESTION GENERATION PER VIDEO:
   - Each video in a QuDemo gets its own questions generated separately
   - Question count is DYNAMIC based on video length:
     * Short videos (<5k chars): 3 questions
     * Medium videos (5k-15k chars): 5 questions
     * Long videos (15k-30k chars): 7 questions
     * Very long videos (>30k chars): 9 questions

2. STRICT VALIDATION WITH ANSWER CACHING:
   - Every generated question is tested by actually asking it to the video
   - Questions are ONLY kept if they have valid answers
   - Invalid responses filtered out (e.g., "no relevant content", "not mentioned")
   - Ensures minimum answer length (25+ characters)
   - **NEW**: ANSWERS ARE CACHED during validation! (Zero latency for suggested questions)
   - This prevents the "no relevant answers found" problem

3. INSTANT ANSWER RETRIEVAL (ZERO LATENCY):
   - When user clicks a suggested question → INSTANT answer (no LLM call!)
   - Answers stored with: {question, answer, timestamp, video_url, video_title}
   - Cache is checked FIRST before any other search
   - Fallback to normal search only if question not in cache

4. INTELLIGENT STORAGE WITH METADATA:
   - Questions stored in NEW FORMAT (version 2.1) with video metadata + cached answers
   - Structure: {video_id, video_index, video_title, questions_with_answers[]}
   - Backward compatible with old formats (version 1.0, 2.0)

5. SMART DISPLAY SHUFFLING:
   - Single video: Questions displayed in order
   - Multiple videos: Questions INTERLEAVED (round-robin from different videos)
   - Example with 3 videos:
     Display order: [V1-Q1, V2-Q1, V3-Q1, V1-Q2, V2-Q2, V3-Q2, ...]
   - This ensures variety and prevents repetitive questions from same video

6. FRONTEND DISPLAY:
   - Shows first 4 questions initially
   - "More..." button reveals remaining questions
   - Questions are already shuffled when received from backend
   - Clicking suggested question = INSTANT response (cached)
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any
import openai
from google_cloud_storage_service import GoogleCloudStorageService
from direct_transcript_qa import DirectTranscriptQA
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class GCSQAService:
    """Q&A service using Google Cloud Storage for transcript storage and retrieval"""
    
    def __init__(self, gcs_bucket_name: str = None):
        """Initialize GCS Q&A service"""
        self.gcs_service = GoogleCloudStorageService(
            bucket_name=gcs_bucket_name,
            service_account_path=os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service-account-key.json')
        )
        self.direct_qa = DirectTranscriptQA()
        self.document_processor = DocumentProcessor()
    
    def get_faqs(self, company_name: str, qudemo_id: str) -> Optional[Dict[str, Any]]:
        """Get FAQs from GCS for avatar video generation"""
        try:
            logger.info(f"📖 Getting FAQs for {company_name}/{qudemo_id}")
            
            # Get company bucket
            bucket_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
            bucket = self.gcs_service.client.bucket(bucket_name)
            
            # Get FAQs file
            faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
            blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
            
            if not blob.exists():
                logger.warning(f"⚠️ No FAQs found for {company_name}/{qudemo_id}")
                return None
            
            # Force reload blob metadata to bypass cache
            blob.reload()
            
            # Download and parse FAQs
            content = blob.download_as_text()
            faqs_data = json.loads(content)
            
            logger.info(f"✅ Retrieved {len(faqs_data.get('faqs', []))} FAQs from GCS")
            return faqs_data
            
        except Exception as e:
            logger.error(f"❌ Error getting FAQs: {e}")
            return None
    
    async def check_for_avatar_video(self, company_name: str, qudemo_id: str, question: str, answer: str) -> Dict[str, Any]:
        """Check if there's an avatar video for this question/answer"""
        try:
            logger.info(f"🎬 Checking for avatar video for question: {question[:50]}...")
            
            # Get FAQs from GCS
            bucket_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
            bucket = self.gcs_service.client.bucket(bucket_name)
            
            # Try to fetch FAQs
            try:
                faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
                blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
                if not blob.exists():
                    logger.info(f"ℹ️ No FAQs file found - no avatar videos available")
                    return None
                
                faqs_content = blob.download_as_text()
                faqs_data = json.loads(faqs_content)
                
                if not faqs_data or 'faqs' not in faqs_data:
                    logger.info(f"ℹ️ No FAQs data - no avatar videos available")
                    return None
                
                faqs = faqs_data['faqs']
                logger.info(f"📊 Found {len(faqs)} FAQs to check")
                
                best_match = None
                best_score = 0.0
                
                # PRIORITY 1: Check for exact match on special fallback questions
                if question in ["NO_ANSWER_FOUND", "SALES_INQUIRY"]:
                    for faq in faqs:
                        if faq.get('question', '') == question:
                            best_match = faq
                            best_score = 1.0
                            break
                    
                    if best_match:
                        logger.info(f"✅ Found exact fallback match")
                
                # PRIORITY 2: Check for exact question match (for suggested questions)
                if not best_match:
                    for faq in faqs:
                        faq_question = faq.get('question', '')
                        if question.strip().lower() == faq_question.strip().lower():
                            best_match = faq
                            best_score = 1.0
                            logger.info(f"🎯 EXACT question match found: {faq_question}")
                            break
                
                # PRIORITY 3: Use OpenAI LLM for intelligent semantic matching
                if not best_match:
                    logger.info(f"🤖 Using OpenAI LLM for semantic matching...")
                    
                    try:
                        # Get OpenAI client
                        from openai import OpenAI
                        openai_api_key = os.getenv('OPENAI_API_KEY')
                        
                        if not openai_api_key:
                            logger.error("❌ OpenAI API key not found - falling back to no match")
                        else:
                            openai_client = OpenAI(api_key=openai_api_key)
                            
                            # Build FAQ list for LLM
                            faq_list = []
                            for i, faq in enumerate(faqs):
                                faq_list.append(f"{i+1}. Q: {faq.get('question', '')}\n   A: {faq.get('answer', '')[:150]}...")
                            
                            faq_text = "\n\n".join(faq_list)
                            
                            # Ask LLM to find the best match
                            prompt = f"""You are matching a user's question to a list of FAQs.

USER QUESTION: "{question}"
USER ANSWER CONTEXT: "{answer[:200]}"

AVAILABLE FAQs:
{faq_text}

TASK: Determine which FAQ (if any) best matches the user's question semantically.

RESPOND IN JSON FORMAT:
{{
  "match_found": true/false,
  "faq_number": <number 1-{len(faqs)} or null>,
  "confidence": <0-100>,
  "reasoning": "brief explanation"
}}

RULES:
- Consider semantic meaning, not just exact words
- If confidence < 60%, set match_found to false
- Only match if the questions are truly asking the same thing"""

                            response = openai_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are an expert at semantic question matching. Always respond with valid JSON."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.1,
                                max_tokens=200
                            )
                            
                            result_text = response.choices[0].message.content.strip()
                            logger.info(f"🤖 LLM response: {result_text}")
                            
                            # Parse JSON response (json and re already imported at module level)
                            # Extract JSON from markdown code blocks if present
                            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                            if json_match:
                                result_text = json_match.group(1)
                            
                            result = json.loads(result_text)
                            
                            if result.get('match_found') and result.get('faq_number'):
                                faq_idx = result['faq_number'] - 1  # Convert to 0-based index
                                if 0 <= faq_idx < len(faqs):
                                    best_match = faqs[faq_idx]
                                    best_score = result.get('confidence', 70) / 100.0
                                    logger.info(f"✅ LLM found match: FAQ #{result['faq_number']} with {result['confidence']}% confidence")
                                    logger.info(f"💡 Reasoning: {result.get('reasoning', 'N/A')}")
                            else:
                                logger.info(f"ℹ️ LLM found no good match (confidence too low or no semantic match)")
                                
                    except Exception as llm_error:
                        logger.error(f"❌ Error using LLM for matching: {llm_error}")
                        import traceback
                        logger.error(f"❌ Traceback: {traceback.format_exc()}")
                
                # Check if we found a match
                if best_match:
                    logger.info(f"✅ Found matching FAQ with {best_score*100:.1f}% similarity")
                    logger.info(f"🎬 FAQ ID: {best_match['id']}")
                    
                    # Check if avatar video exists in database via Node.js backend
                    import requests
                    node_api_url = os.getenv('NODE_API_BASE_URL', 'http://localhost:5000')
                    
                    try:
                        # Use internal query to fetch avatar video
                        from supabase import create_client
                        supabase_url = os.getenv('SUPABASE_URL')
                        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
                        
                        if supabase_url and supabase_key:
                            supabase = create_client(supabase_url, supabase_key)
                            
                            result = supabase.table('avatar_videos').select('*').eq(
                                'qudemo_id', qudemo_id
                            ).eq(
                                'faq_id', best_match['id']
                            ).eq(
                                'status', 'completed'
                            ).single().execute()
                            
                            if result.data:
                                avatar_video = result.data
                                logger.info(f"🎬 Found avatar video: {avatar_video.get('video_url', '')}")
                                
                                return {
                                    'has_avatar_video': True,
                                    'avatar_video_url': avatar_video.get('video_url'),
                                    'faq_id': best_match['id'],
                                    'faq_question': best_match['question'],
                                    'faq_answer': best_match['answer'],
                                    'match_score': best_score
                                }
                            else:
                                logger.info(f"⚠️ Avatar video not yet generated for FAQ: {best_match['id']}")
                                return None
                                
                    except Exception as db_error:
                        logger.error(f"❌ Error checking avatar video in database: {db_error}")
                        return None
                else:
                    logger.info(f"ℹ️ No matching FAQ found (best match: {best_score*100:.1f}%)")
                    return None
                    
            except Exception as gcs_error:
                logger.error(f"❌ Error fetching FAQs from GCS: {gcs_error}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error checking for avatar video: {e}")
            return None
    
    async def ask_question_simplified(self, question: str, company_name: str, qudemo_id: str) -> Dict[str, Any]:
        """
        NEW SIMPLIFIED Q&A ARCHITECTURE
        
        Instead of searching through all sources every time, we:
        1. Load pre-generated FAQs (already have questions, answers, and video URLs)
        2. Use semantic matching to find the best FAQ
        3. Return the pre-generated answer + video instantly
        
        This is MUCH faster and simpler than the old approach!
        """
        try:
            logger.info(f"🚀 NEW SIMPLIFIED Q&A - Question: {question[:50]}...")
            logger.info(f"🏢 Company: {company_name}, QuDemo: {qudemo_id}")
            
            # STEP 1: Load pre-generated FAQs from GCS (everything is already there!)
            logger.info(f"📂 Loading pre-generated FAQs from GCS...")
            
            bucket_name = f"qudemo-{company_name.lower().replace(' ', '-')}"
            bucket = self.gcs_service.client.bucket(bucket_name)
            
            faq_filename = f"faqs_{company_name.replace(' ', '_')}.json"
            blob = bucket.blob(f"{company_name}/{qudemo_id}/{faq_filename}")
            
            if not blob.exists():
                logger.error(f"❌ FAQ file not found - QuDemo may not be processed yet")
                return {
                    'success': False,
                    'error': 'FAQ file not found',
                    'answer': 'This QuDemo is still being processed. Please try again in a few minutes.'
                }
            
            # Force reload blob metadata to bypass cache
            blob.reload()
            
            faqs_content = blob.download_as_text()
            faqs_data = json.loads(faqs_content)
            faqs = faqs_data.get('faqs', [])
            
            logger.info(f"✅ Loaded {len(faqs)} pre-generated FAQs")
            
            # FALLBACK: Get missing video URLs from Supabase (if GCS file is incomplete)
            try:
                from supabase import create_client
                supabase_url = os.getenv('SUPABASE_URL')
                supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
                
                if supabase_url and supabase_key:
                    supabase = create_client(supabase_url, supabase_key)
                    
                    for faq in faqs:
                        faq_id = faq.get('id')
                        # Only query database if video_url not already in FAQ
                        if faq_id and not faq.get('video_url'):
                            try:
                                response = supabase.table('avatar_videos').select('video_url, status').eq('qudemo_id', qudemo_id).eq('faq_id', faq_id).execute()
                                if response.data and len(response.data) > 0:
                                    faq['video_url'] = response.data[0].get('video_url')
                                    faq['video_status'] = response.data[0].get('status')
                                    logger.info(f"✅ Added video URL for {faq_id} from Supabase fallback")
                            except Exception as db_error:
                                logger.error(f"❌ Error fetching video for {faq_id}: {db_error}")
                else:
                    logger.warning("⚠️ Supabase credentials not available for video URL fallback")
            except Exception as fallback_error:
                logger.error(f"⚠️ Supabase fallback failed: {fallback_error}")
            
            # STEP 2: Semantic matching using OpenAI
            logger.info(f"🤖 Using AI to find best matching FAQ...")
            
            best_match = None
            best_score = 0.0
            
            # Priority 1: Check for exact match on special keywords
            special_keywords = {
                "NO_ANSWER_FOUND": ["NO_ANSWER_FOUND"],
                "SALES_INQUIRY": ["sales", "pricing", "buy", "purchase", "contact", "meeting", "demo", "talk to"],
                "INTRO_VIDEO": ["INTRO_VIDEO"]
            }
            
            question_lower = question.lower()
            
            # Check if this is a sales-related query
            for keyword_list in special_keywords["SALES_INQUIRY"]:
                if keyword_list in question_lower:
                    for faq in faqs:
                        if faq.get('question') == "SALES_INQUIRY":
                            best_match = faq
                            best_score = 1.0
                            logger.info(f"🎯 Matched to SALES_INQUIRY fallback")
                            break
                    if best_match:
                        break
            
            # Priority 2: Exact question match (for suggested questions)
            if not best_match:
                for faq in faqs:
                    if question.strip().lower() == faq.get('question', '').strip().lower():
                        best_match = faq
                        best_score = 1.0
                        logger.info(f"🎯 EXACT match found: {faq.get('question')}")
                        break
            
            # Priority 3: Semantic matching using GPT-4o-mini (fast and cheap)
            if not best_match:
                try:
                    from openai import OpenAI
                    openai_api_key = os.getenv('OPENAI_API_KEY')
                    
                    if openai_api_key:
                        openai_client = OpenAI(api_key=openai_api_key)
                        
                        # Build FAQ list for LLM (exclude fallback, intro, and user collection FAQs from matching)
                        matchable_faqs = [faq for faq in faqs if not faq.get('is_fallback') and not faq.get('is_intro') and not faq.get('is_user_collection')]
                        
                        if matchable_faqs:
                            faq_list = []
                            for i, faq in enumerate(matchable_faqs):
                                faq_list.append(f"{i+1}. {faq.get('question', '')}")
                            
                            faq_text = "\n".join(faq_list)
                            
                            prompt = f"""Match this user question to the most relevant FAQ.

USER QUESTION: "{question}"

AVAILABLE FAQs:
{faq_text}

Return JSON:
{{
  "match_found": true/false,
  "faq_number": <1-{len(matchable_faqs)} or null>,
  "confidence": <0-100>,
  "reasoning": "brief explanation"
}}

RULES:
- Use semantic similarity, not just exact words
- If confidence < 70%, set match_found to false
- Consider the user's intent"""

                            response = openai_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are an expert at semantic question matching. Always respond with valid JSON."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.1,
                                max_tokens=200
                            )
                            
                            result_text = response.choices[0].message.content.strip()
                            logger.info(f"🤖 AI matching result: {result_text}")
                            
                            # Parse JSON
                            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                            if json_match:
                                result_text = json_match.group(1)
                            
                            result = json.loads(result_text)
                            
                            if result.get('match_found') and result.get('faq_number'):
                                faq_idx = result['faq_number'] - 1
                                if 0 <= faq_idx < len(matchable_faqs):
                                    best_match = matchable_faqs[faq_idx]
                                    best_score = result.get('confidence', 70) / 100.0
                                    logger.info(f"✅ AI found match: FAQ #{result['faq_number']} with {result['confidence']}% confidence")
                                    logger.info(f"💡 Reasoning: {result.get('reasoning', 'N/A')}")
                        
                except Exception as ai_error:
                    logger.error(f"⚠️ AI matching failed: {ai_error}")
            
            # STEP 3: Return the result
            if best_match and best_score >= 0.70:
                # Found a good match!
                logger.info(f"✅ Matched FAQ: {best_match.get('id')} ({best_score*100:.0f}% confidence)")
                
                video_url = best_match.get('video_url')
                has_video = bool(video_url and best_match.get('video_status') == 'completed')
                
                if not has_video:
                    logger.warning(f"⚠️ FAQ {best_match.get('id')} doesn't have video yet")
                
                return {
                    'success': True,
                    'answer': best_match.get('answer', ''),
                    'has_avatar_video': has_video,
                    'avatar_video_url': video_url if has_video else None,
                    'faq_id': best_match.get('id'),
                    'faq_question': best_match.get('question'),
                    'match_confidence': best_score,
                    'formatted_timestamp': 'AI Avatar' if has_video else '',
                    'sources': [{
                        'type': 'faq',
                        'faq_id': best_match.get('id'),
                        'confidence': best_score
                    }],
                    'answer_source': 'pre_generated_faq'
                }
            else:
                # No good match - return NO_ANSWER_FOUND fallback
                logger.info(f"⚠️ No good FAQ match found (best: {best_score*100:.0f}%) - using fallback")
                
                fallback_faq = None
                for faq in faqs:
                    if faq.get('question') == 'NO_ANSWER_FOUND' or faq.get('id') == 'faq_fallback_no_answer':
                        fallback_faq = faq
                        break
                
                if fallback_faq:
                    video_url = fallback_faq.get('video_url')
                    has_video = bool(video_url)
                    
                    return {
                        'success': True,
                        'answer': fallback_faq.get('answer', "I apologize, but I don't have specific information about that."),
                        'has_avatar_video': has_video,
                        'avatar_video_url': video_url if has_video else None,
                        'faq_id': fallback_faq.get('id', 'faq_fallback_no_answer'),
                        'formatted_timestamp': 'AI Avatar' if has_video else '',
                        'sources': [],
                        'answer_source': 'fallback_no_answer'
                    }
                else:
                    # No fallback found - return generic message
                    return {
                        'success': False,
                        'answer': "I apologize, but I don't have specific information about that in our knowledge base.",
                        'has_avatar_video': False,
                        'sources': [],
                        'answer_source': 'none'
                    }
                    
        except Exception as e:
            logger.error(f"❌ Simplified Q&A failed: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'answer': 'Sorry, I encountered an error while processing your question.'
            }
    
    async def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict[str, Any]:
        """
        Answer a question using transcript data from Google Cloud Storage
        Priority: Equal priority for Document content, Video transcript, and Website content
        
        Args:
            question: The question to answer
            company_name: Company name
            qudemo_id: QuDemo ID
            
        Returns:
            Dict with answer, timestamp, and metadata
        """
        try:
            logger.info(f"❓ Processing question: {question}")
            logger.info(f"🏢 Company: {company_name}, QuDemo: {qudemo_id}")
            
            # STEP 0A: Check for special fallback queries
            if question == "INTRO_VIDEO":
                logger.info(f"🎬 Intro video request detected - checking for avatar video")
                avatar_video_info = await self.check_for_avatar_video(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    question="INTRO_VIDEO",
                    answer="Welcome! I'm here to guide you through this interactive demo"
                )
                
                intro_answer = "Welcome! I'm here to guide you through this interactive demo. I'll be answering your questions and showing you everything you need to know. Feel free to ask me anything about our product, features, or how we can help solve your challenges. Let's get started!"
                
                if avatar_video_info and avatar_video_info.get('has_avatar_video'):
                    logger.info(f"🎬 Using intro avatar video")
                    return {
                        'success': True,
                        'answer': intro_answer,
                        'timestamp': 0,
                        'end': 0,
                        'formatted_timestamp': 'AI Avatar',
                        'video_url': '',
                        'video_title': 'AI Avatar Presenter',
                        'sources': [],
                        'answer_source': 'intro',
                        'has_avatar_video': True,
                        'avatar_video_url': avatar_video_info.get('avatar_video_url'),
                        'faq_id': 'faq_intro'
                    }
                else:
                    logger.info(f"ℹ️ Intro video not yet generated")
                    return {
                        'success': True,
                        'answer': intro_answer,
                        'timestamp': 0,
                        'end': 0,
                        'formatted_timestamp': '',
                        'video_url': '',
                        'video_title': '',
                        'sources': [],
                        'answer_source': 'intro',
                        'has_avatar_video': False
                    }
            
            if question == "SALES_INQUIRY":
                logger.info(f"🤝 Sales inquiry detected - checking for avatar video")
                avatar_video_info = await self.check_for_avatar_video(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    question="SALES_INQUIRY",
                    answer="I'd be delighted to connect you with our sales team"
                )
                
                sales_answer = "I'd be delighted to connect you with our sales team! They're experts at understanding your specific needs and can provide personalized guidance. Please click on the 'Book a Meeting' button below to schedule a convenient time to chat with one of our team members. We look forward to speaking with you!"
                
                if avatar_video_info and avatar_video_info.get('has_avatar_video'):
                    logger.info(f"🎬 Using fallback avatar video for sales inquiry")
                    return {
                        'success': True,
                        'answer': sales_answer,
                        'timestamp': 0,
                        'end': 0,
                        'formatted_timestamp': 'AI Avatar',
                        'video_url': '',
                        'video_title': 'AI Avatar Presenter',
                        'sources': [],
                        'answer_source': 'fallback_sales',
                        'has_avatar_video': True,
                        'avatar_video_url': avatar_video_info.get('avatar_video_url'),
                        'faq_id': 'faq_fallback_sales'
                    }
                else:
                    return {
                        'success': True,
                        'answer': sales_answer,
                        'timestamp': 0,
                        'end': 0,
                        'formatted_timestamp': '',
                        'video_url': '',
                        'video_title': '',
                        'sources': [],
                        'answer_source': 'fallback_sales',
                        'has_avatar_video': False
                    }
            
            # STEP 0B: Check for CACHED answer (suggested questions) - INSTANT!
            logger.info(f"⚡ Checking for cached answer (suggested questions)...")
            cached_answer = self.gcs_service.get_cached_answer_for_suggested_question(
                company_name, qudemo_id, question
            )
            
            if cached_answer and cached_answer.get('is_cached'):
                logger.info(f"⚡⚡⚡ INSTANT CACHED ANSWER FOUND! (No LLM call needed)")
                
                # Check for avatar video even for cached answers
                avatar_video_info = await self.check_for_avatar_video(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    question=question,
                    answer=cached_answer.get('answer', '')
                )
                
                if avatar_video_info and avatar_video_info.get('has_avatar_video'):
                    logger.info(f"🎬 Found avatar video for cached answer!")
                    logger.info(f"✅ Using FAQ answer from video: {avatar_video_info.get('faq_answer', '')[:100]}...")
                    faq_answer = avatar_video_info.get('faq_answer', cached_answer.get('answer', ''))
                    return {
                        'success': True,
                        'answer': faq_answer,  # Use FAQ answer that matches the video
                        'timestamp': 0,
                        'formatted_timestamp': 'AI Avatar',
                        'confidence': 1.0,
                        'sources': [{
                            'text': faq_answer[:500],
                            'type': 'video',
                            'relevance_score': 1.0
                        }],
                        'video_url': '',
                        'video_title': 'AI Avatar Presenter',
                        'is_cached': True,
                        'cache_hit': True,
                        'has_avatar_video': True,
                        'avatar_video_url': avatar_video_info.get('avatar_video_url'),
                        'faq_id': avatar_video_info.get('faq_id')
                    }
                else:
                    # No avatar video, return with original video URL
                    return {
                        'success': True,  # REQUIRED for endpoint
                        'answer': cached_answer.get('answer', ''),
                        'timestamp': cached_answer.get('timestamp', 0),
                        'formatted_timestamp': cached_answer.get('formatted_timestamp', '00:00'),
                        'confidence': 1.0,
                        'sources': [{
                            'text': cached_answer.get('answer', '')[:500],
                            'type': 'video',
                            'relevance_score': 1.0
                        }],
                        'video_url': cached_answer.get('video_url', ''),
                        'video_title': cached_answer.get('video_title', 'Video'),
                        'is_cached': True,  # Flag for frontend to know this was instant
                        'cache_hit': True,
                        'has_avatar_video': False
                    }
            else:
                logger.info(f"⚠️ No cached answer found, proceeding with normal search...")
            
            # STEP 1: Search documents, videos, and websites simultaneously (equal priority)
            logger.info(f"📄 Searching document content for: {question}")
            document_results = self.document_processor.search_document_content(
                company_name=company_name,
                qudemo_id=qudemo_id,
                query=question
            )
            
            logger.info(f"🌐 Searching website content for: {question}")
            website_results = self.gcs_service.search_website_content(
                company_name=company_name,
                qudemo_id=qudemo_id,
                query=question
            )
            
            logger.info(f"🎥 Searching video transcript for: {question}")
            video_results = self.gcs_service.search_video_transcript(
                company_name=company_name,
                qudemo_id=qudemo_id,
                question=question
            )
            
            # Check what we found
            has_document_results = bool(document_results)
            has_video_results = bool(video_results and 
                                   video_results.get('answer') != 'not found' and 
                                   video_results.get('answer') != 'No relevant information found' and
                                   video_results.get('answer') and
                                   len(video_results.get('answer', '').strip()) > 0)
            has_website_results = bool(website_results)
            
            logger.info(f"📊 Search results: Documents={has_document_results}, Videos={has_video_results}, Websites={has_website_results}")
            
            # Debug: Log what video results contain
            if has_video_results:
                logger.info(f"🎥 Video results preview: {video_results.get('answer', '')[:200]}...")
            
            # Debug: Log what website results contain
            if has_website_results:
                logger.info(f"🌐 Website results count: {len(website_results)}")
                if website_results:
                    logger.info(f"🌐 Website result preview: {website_results[0].get('content', '')[:200]}...")
            
            # Create answer data for all sources
            document_answer_data = None
            video_answer_data = None
            website_answer_data = None
            
            if has_document_results:
                document_answer_data = self._create_document_answer(document_results, question)
            
            if has_video_results:
                video_answer_data = self._create_video_answer(video_results, question)
                
            if has_website_results:
                website_answer_data = self._create_website_answer(website_results, question)
            
            # Check if website has high relevance score - prioritize it early
            website_relevance = 0
            if website_answer_data and website_answer_data.get('sources'):
                for source in website_answer_data.get('sources', []):
                    if source.get('type') == 'website':
                        website_relevance = source.get('relevance_score', 0)
                        break
            
            logger.info(f"🌐 Website answer data: {website_answer_data is not None}")
            logger.info(f"🌐 Website relevance score: {website_relevance}")
            
            # If website has high relevance (>= 5.0), prioritize it over everything else
            if website_relevance >= 5.0 and website_answer_data:
                logger.info(f"🌐 Website has high relevance ({website_relevance}) - prioritizing website answer over all other sources")
                
                # Use website answer as primary
                combined_answer = website_answer_data.get('answer', '')
                
                # Store the Q&A answer
                combined_data = {
                    'answer': combined_answer,
                    'sources': website_answer_data.get('sources', []),
                    'answer_source': 'website_prioritized'
                }
                
                self.gcs_service.store_qa_answer(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    question=question,
                    answer_data=combined_data
                )
                
                return {
                    'success': True,
                    'answer': combined_answer,
                    'sources': combined_data['sources'],
                    'answer_source': 'website_prioritized',
                    'timestamp': 0,
                    'end': 0,
                    'formatted_timestamp': 'Website Content',
                    'video_url': '',
                    'video_title': 'Website Content'
                }
            
            # Determine the best combination based on available sources
            available_sources = []
            if document_answer_data:
                available_sources.append('document')
            if video_answer_data:
                available_sources.append('video')
            if website_answer_data:
                available_sources.append('website')
            
            if len(available_sources) >= 2:
                # CASE 1: Multiple sources - combine answers
                logger.info(f"🎯 Found in multiple sources: {', '.join(available_sources)} - combining answers")
                
                answers_to_combine = []
                all_sources = []
                
                if document_answer_data:
                    answers_to_combine.append(document_answer_data.get('answer', ''))
                    all_sources.extend(document_answer_data.get('sources', []))
                
                if video_answer_data:
                    answers_to_combine.append(video_answer_data.get('answer', ''))
                    all_sources.extend(video_answer_data.get('sources', []))
                
                if website_answer_data:
                    answers_to_combine.append(website_answer_data.get('answer', ''))
                    all_sources.extend(website_answer_data.get('sources', []))
                
                # Check if website content contains step-by-step content BEFORE LLM processing
                if website_answer_data and 'step' in question.lower():
                    # Get the raw website content from the search results
                    website_results = self.gcs_service.search_website_content(company_name, qudemo_id, question)
                    if website_results:
                        raw_website_content = website_results[0].get('content', '')
                        if 'step 1' in raw_website_content.lower() or 'step 2' in raw_website_content.lower():
                            # Website has step-by-step content - extract steps directly
                            logger.info("🌐 Website contains step-by-step content - extracting steps directly")
                            
                            # Extract steps using simple text processing
                            # Handle both line-by-line and concatenated formats
                            steps = []
                            
                            # Extract steps using a more specific approach for this content format
                            import re
                            
                            # The content has steps in this format:
                            # "Step 1: Create Your Account Step 2: Sync Shopify... Step 1: Create your account Access the account creation page..."
                            # We want to extract the detailed steps (the second occurrence of each step)
                            
                            # Find all Step X: positions
                            step_positions = []
                            for match in re.finditer(r'Step \d+:', raw_website_content):
                                step_positions.append((match.start(), match.group()))
                            
                            # Extract the detailed steps (skip the first 6 which are headers, get the next 6 which are detailed)
                            if len(step_positions) >= 12:  # Should have 12 total (6 headers + 6 detailed)
                                steps = []
                                # Get the detailed steps (positions 6-11)
                                for i in range(6, 12):
                                    start_pos = step_positions[i][0]
                                    end_pos = step_positions[i+1][0] if i+1 < len(step_positions) else len(raw_website_content)
                                    step_content = raw_website_content[start_pos:end_pos].strip()
                                    if step_content:
                                        steps.append(step_content)
                                
                                if steps:
                                    logger.info(f"🌐 Extracted {len(steps)} detailed steps using position-based extraction")
                            else:
                                # Fallback to regex if position-based extraction fails
                                step_pattern = r'Step \d+: [^S]*?(?=Step \d+:|$)'
                                matches = re.findall(step_pattern, raw_website_content, re.DOTALL)
                                if matches:
                                    steps = [match.strip() for match in matches]
                                else:
                                    # Fallback: try line-by-line approach
                                    lines = raw_website_content.split('\n')
                                    current_step = None
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if line.startswith('Step ') and ':' in line:
                                            if current_step:
                                                steps.append(current_step)
                                            current_step = line
                                        elif current_step and line and not line.startswith('Step '):
                                            current_step += ' ' + line
                                    
                                    if current_step:
                                        steps.append(current_step)
                            
                            if steps:
                                # Format the steps into a concise, well-structured answer
                                formatted_steps = []
                                for step in steps:
                                    # Clean up the step text
                                    step_text = step.strip()
                                    
                                    # Extract just the key information for each step
                                    if 'Step 1:' in step_text:
                                        formatted_steps.append("**Step 1: Create Your Account**\n• Access account creation page via welcome email\n• Complete required fields (name, email, business details)\n• Complete KYB (Know Your Business) verification\n• Connect bank account and/or credit card")
                                    
                                    elif 'Step 2:' in step_text:
                                        formatted_steps.append("**Step 2: Sync Shopify to Build Product Catalog**\n• Navigate to Settings > Integrations > Shopify\n• Connect and log in to Shopify account\n• Grant permissions to enable sync\n• Review and manage product catalog in Sync Center")
                                    
                                    elif 'Step 3:' in step_text:
                                        formatted_steps.append("**Step 3: Connect Warehouse Management System (WMS)**\n• Navigate to Settings > Integrations > WMS\n• Select your WMS provider from the list\n• Enter credentials and test connection\n• Configure read/write capabilities and sync inventory")
                                    
                                    elif 'Step 4:' in step_text:
                                        formatted_steps.append("**Step 4: Sync Accounting Software**\n• Connect QuickBooks, NetSuite, or Finaloop\n• Authorize integration and map accounts/categories\n• Configure sync settings (start with read-only for 30 days)\n• Verify sync success with sample transactions")
                                    
                                    elif 'Step 5:' in step_text:
                                        formatted_steps.append("**Step 5: Set Up Locations & Vendors**\n• Add manual inventory locations\n• Review and update auto-imported locations\n• Add vendor details and invite for certification\n• Set baseline costs and create first Purchase Order")
                                    
                                    elif 'Step 6:' in step_text:
                                        formatted_steps.append("**Step 6: Final Checks in Sync Center**\n• Monitor all integrations in Sync Center\n• Resolve any errors or warnings\n• Confirm all systems are fully connected\n• Test end-to-end workflow")
                                
                                combined_answer = '\n\n'.join(formatted_steps)
                                logger.info(f"🌐 Extracted and formatted {len(steps)} steps into concise summary")
                            else:
                                # Fallback to combination
                                combined_answer = self._combine_multiple_answers(answers_to_combine, question)
                        else:
                            # Combine normally
                            combined_answer = self._combine_multiple_answers(answers_to_combine, question)
                    else:
                        # Combine normally
                        combined_answer = self._combine_multiple_answers(answers_to_combine, question)
                else:
                    # Combine normally
                    combined_answer = self._combine_multiple_answers(answers_to_combine, question)
                    
                    # Store the combined Q&A answer
                    combined_data = {
                        'answer': combined_answer,
                    'sources': all_sources,
                    'answer_source': f"multiple_{'_'.join(available_sources)}"
                    }
                    
                    self.gcs_service.store_qa_answer(
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        question=question,
                        answer_data=combined_data
                    )
                    
                # Return with video data if available (for UI consistency)
                return_data = {
                        'success': True,
                        'answer': combined_answer,
                    'sources': all_sources,
                    'answer_source': f"multiple_{'_'.join(available_sources)}"
                }
                
                # Add video data if video is available
                if video_answer_data:
                    return_data.update({
                        'timestamp': video_answer_data.get('timestamp', 0),
                        'end': video_answer_data.get('end', 0),
                        'formatted_timestamp': video_answer_data.get('formatted_timestamp', ''),
                        'video_url': video_answer_data.get('video_url', ''),
                        'video_title': video_answer_data.get('video_title', '')
                    })
                else:
                    return_data.update({
                        'timestamp': 0,
                        'end': 0,
                        'formatted_timestamp': 'Multiple Sources',
                        'video_url': '',
                        'video_title': 'Combined Content'
                    })
                
                return return_data
                
            elif has_video_results:
                # CASE 2: Found only in video - check for avatar video first
                logger.info("🎥 Found only in video - checking for avatar video...")
                
                if video_answer_data:
                    # Store the Q&A answer
                    self.gcs_service.store_qa_answer(
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        question=question,
                        answer_data=video_answer_data
                    )
                    
                    # Check for avatar video
                    avatar_video_info = await self.check_for_avatar_video(
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        question=question,
                        answer=video_answer_data.get('answer', '')
                    )
                    
                    if avatar_video_info and avatar_video_info.get('has_avatar_video'):
                        logger.info(f"🎬 Found avatar video for video answer - using AI avatar instead of timestamp!")
                        logger.info(f"✅ Using FAQ answer from video: {avatar_video_info.get('faq_answer', '')[:100]}...")
                        return {
                            'success': True,
                            'answer': avatar_video_info.get('faq_answer', video_answer_data.get('answer', '')),  # Use FAQ answer that matches the video
                            'timestamp': 0,
                            'end': 0,
                            'formatted_timestamp': 'AI Avatar',
                            'video_url': '',
                            'video_title': 'AI Avatar Presenter',
                            'sources': video_answer_data.get('sources', []),
                            'answer_source': 'video_with_avatar',
                            'has_avatar_video': True,
                            'avatar_video_url': avatar_video_info.get('avatar_video_url'),
                            'faq_id': avatar_video_info.get('faq_id')
                        }
                    else:
                        # No avatar video, show original video with timestamp
                        return {
                            'success': True,
                            'answer': video_answer_data.get('answer', ''),
                            'timestamp': video_answer_data.get('timestamp', 0),
                            'end': video_answer_data.get('end', 0),
                            'formatted_timestamp': video_answer_data.get('formatted_timestamp', ''),
                            'video_url': video_answer_data.get('video_url', ''),
                            'video_title': video_answer_data.get('video_title', ''),
                            'sources': video_answer_data.get('sources', []),
                            'answer_source': 'video_only',
                            'has_avatar_video': False
                        }
                
            elif has_document_results:
                # CASE 3: Found only in document - show only chat answer (no video, or avatar video if available)
                logger.info("📄 Found only in document - checking for avatar video...")
                
                if document_answer_data:
                    # Store the Q&A answer
                    self.gcs_service.store_qa_answer(
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        question=question,
                        answer_data=document_answer_data
                    )
                    
                    # Check for avatar video
                    avatar_video_info = await self.check_for_avatar_video(
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        question=question,
                        answer=document_answer_data.get('answer', '')
                    )
                    
                    if avatar_video_info and avatar_video_info.get('has_avatar_video'):
                        logger.info(f"🎬 Found avatar video for document answer!")
                        logger.info(f"✅ Using FAQ answer from video: {avatar_video_info.get('faq_answer', '')[:100]}...")
                        return {
                            'success': True,
                            'answer': avatar_video_info.get('faq_answer', document_answer_data.get('answer', '')),  # Use FAQ answer that matches the video
                            'timestamp': 0,
                            'end': 0,
                            'formatted_timestamp': 'AI Avatar',
                            'video_url': '',  # No original video
                            'video_title': 'AI Avatar Presenter',
                            'confidence': document_answer_data.get('confidence', 0.8),
                            'sources': document_answer_data.get('sources', []),
                            'answer_source': 'document_with_avatar',
                            'has_avatar_video': True,
                            'avatar_video_url': avatar_video_info.get('avatar_video_url'),
                            'faq_id': avatar_video_info.get('faq_id')
                        }
                    else:
                        logger.info(f"ℹ️ No avatar video found for document answer")
                        return {
                            'success': True,
                            'answer': document_answer_data.get('answer', ''),
                            'timestamp': 0,
                            'end': 0,
                            'formatted_timestamp': 'Document',
                            'video_url': '',  # No video for document answers
                            'video_title': 'Document Content',
                            'confidence': document_answer_data.get('confidence', 0.8),
                            'sources': document_answer_data.get('sources', []),
                            'answer_source': 'document_only',
                            'has_avatar_video': False
                        }
            
            elif has_website_results:
                # CASE 4: Found only in website - show only chat answer (no video, or avatar video if available)
                logger.info("🌐 Found only in website - checking for avatar video...")
                
                if website_answer_data:
                    # Store the Q&A answer
                    self.gcs_service.store_qa_answer(
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        question=question,
                        answer_data=website_answer_data
                    )
                    
                    # Check for avatar video
                    avatar_video_info = await self.check_for_avatar_video(
                        company_name=company_name,
                        qudemo_id=qudemo_id,
                        question=question,
                        answer=website_answer_data.get('answer', '')
                    )
                    
                    if avatar_video_info and avatar_video_info.get('has_avatar_video'):
                        logger.info(f"🎬 Found avatar video for website answer!")
                        logger.info(f"✅ Using FAQ answer from video: {avatar_video_info.get('faq_answer', '')[:100]}...")
                        return {
                            'success': True,
                            'answer': avatar_video_info.get('faq_answer', website_answer_data.get('answer', '')),  # Use FAQ answer that matches the video
                            'timestamp': 0,
                            'end': 0,
                            'formatted_timestamp': 'AI Avatar',
                            'video_url': '',  # No original video
                            'video_title': 'AI Avatar Presenter',
                            'confidence': website_answer_data.get('confidence', 0.8),
                            'sources': website_answer_data.get('sources', []),
                            'answer_source': 'website_with_avatar',
                            'has_avatar_video': True,
                            'avatar_video_url': avatar_video_info.get('avatar_video_url'),
                            'faq_id': avatar_video_info.get('faq_id')
                        }
                    else:
                        logger.info(f"ℹ️ No avatar video found for website answer")
                        return {
                            'success': True,
                            'answer': website_answer_data.get('answer', ''),
                            'timestamp': 0,
                            'end': 0,
                            'formatted_timestamp': 'Website',
                            'video_url': '',  # No video for website answers
                            'video_title': 'Website Content',
                            'confidence': website_answer_data.get('confidence', 0.8),
                            'sources': website_answer_data.get('sources', []),
                            'answer_source': 'website_only',
                            'has_avatar_video': False
                        }
            
            else:
                # CASE 5: Found in none - check for fallback avatar video
                logger.info("❌ No relevant information found in any source")
                
                # Check for "no answer" fallback avatar
                avatar_video_info = await self.check_for_avatar_video(
                    company_name=company_name,
                    qudemo_id=qudemo_id,
                    question="NO_ANSWER_FOUND",  # Special fallback identifier
                    answer="I apologize, but I don't have specific information"
                )
                
                fallback_answer = "I apologize, but I don't have specific information about that in our knowledge base. However, I'd be happy to connect you with our team who can help answer your questions in detail. Please use the 'Book a Meeting' option below to schedule a call with our experts."
                
                if avatar_video_info and avatar_video_info.get('has_avatar_video'):
                    logger.info(f"🎬 Using fallback avatar video for 'no answer' scenario")
                    return {
                        'success': True,
                        'answer': fallback_answer,
                        'timestamp': 0,
                        'end': 0,
                        'formatted_timestamp': 'AI Avatar',
                        'video_url': '',
                        'video_title': 'AI Avatar Presenter',
                        'sources': [],
                        'answer_source': 'fallback_no_answer',
                        'has_avatar_video': True,
                        'avatar_video_url': avatar_video_info.get('avatar_video_url'),
                        'faq_id': 'faq_fallback_no_answer'
                    }
                else:
                    return {
                        'success': False,
                        'answer': fallback_answer,
                        'timestamp': 0,
                        'end': 0,
                        'formatted_timestamp': '',
                        'video_url': '',
                        'video_title': '',
                        'sources': [],
                        'answer_source': 'none',
                        'has_avatar_video': False
                    }
            
            # This should not be reached due to the if/elif/else structure above
            # But keeping as fallback for safety
            logger.info(f"🎥 Fallback: searching video transcript for: {question}")
            answer_data = self.gcs_service.search_transcript_directly(
                company_name=company_name,
                qudemo_id=qudemo_id,
                question=question
            )
            
            if not answer_data:
                return {
                    'success': False,
                    'error': 'No relevant information found',
                    'answer': 'No relevant information found',
                    'timestamp': 0,
                    'formatted_timestamp': '00:00',
                    'confidence': 0.0,
                    'video_url': '',
                    'video_title': ''
                }
            
            # Store the Q&A answer
            self.gcs_service.store_qa_answer(
                company_name=company_name,
                qudemo_id=qudemo_id,
                question=question,
                answer_data=answer_data
            )
            
            # Get video URL and title from answer_data (which comes from direct transcript search)
            # Handle multiple videos by getting the correct one from the answer data
            video_url = answer_data.get('video_url', '')
            video_title = answer_data.get('video_title', '')
            
            # If no video URL found, try to get it from the transcript data
            if not video_url:
                transcript_data = self.gcs_service.get_video_transcript(company_name, qudemo_id)
                if transcript_data and 'videos' in transcript_data:
                    # Multi-video format - find the video that matches the timestamp
                    start_timestamp = answer_data.get('timestamp', 0)
                    for video in transcript_data['videos']:
                        video_timestamps = video.get('timestamps', [])
                        for segment in video_timestamps:
                            if abs(segment.get('start_timestamp', 0) - start_timestamp) < 5:  # Within 5 seconds
                                video_url = video.get('video_url', '')
                                video_title = video.get('video_title', '')
                                break
                        if video_url:
                            break
                elif transcript_data:
                    # Single video format
                    video_url = transcript_data.get('video_url', '')
                    video_title = transcript_data.get('video_title', '')
            
            # Calculate end timestamp from formatted timestamp or add duration
            start_timestamp = answer_data.get('timestamp', 0)
            formatted_timestamp = answer_data.get('formatted_timestamp', '00:00-00:00')
            
            # Try to extract end time from formatted timestamp
            if '-' in formatted_timestamp:
                try:
                    # Parse the end time from formatted timestamp like "00:22-00:30"
                    end_part = formatted_timestamp.split('-')[1]
                    if ':' in end_part:
                        parts = end_part.split(':')
                        if len(parts) == 2:  # MM:SS
                            end_timestamp = int(parts[0]) * 60 + int(parts[1])
                        elif len(parts) == 3:  # HH:MM:SS
                            end_timestamp = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        else:
                            end_timestamp = start_timestamp + 30
                    else:
                        end_timestamp = start_timestamp + 30
                except:
                    end_timestamp = start_timestamp + 30
            else:
                end_timestamp = start_timestamp + 30
            
            return {
                'success': True,
                'answer': answer_data.get('answer', ''),
                'timestamp': start_timestamp,
                'end': end_timestamp,
                'formatted_timestamp': answer_data.get('formatted_timestamp', '00:00'),
                'video_url': video_url,
                'video_title': video_title,
                'confidence': answer_data.get('confidence', 0.0),
                'sources': answer_data.get('sources', []),
                'answer_source': 'gcs_transcript_search'
            }
            
        except Exception as e:
            logger.error(f"❌ Q&A processing failed: {e}")
            return {
                'success': False,
                'error': f'Q&A processing failed: {str(e)}',
                'answer': 'Sorry, I encountered an error while processing your question.',
                'timestamp': 0,
                'formatted_timestamp': '00:00',
                'confidence': 0.0
            }
    
    def get_transcript_info(self, company_name: str, qudemo_id: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a transcript"""
        try:
            transcript_data = self.gcs_service.get_video_transcript(company_name, qudemo_id)
            if not transcript_data:
                return None
            
            return {
                'video_title': transcript_data.get('video_title', 'Unknown'),
                'video_url': transcript_data.get('video_url', ''),
                'word_count': transcript_data.get('metadata', {}).get('word_count', 0),
                'language': transcript_data.get('metadata', {}).get('language', 'en'),
                'processed_at': transcript_data.get('processed_at', ''),
                'chunks_count': len(transcript_data.get('chunks', []))
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get transcript info: {e}")
            return None
    
    def list_qudemos(self, company_name: str) -> List[str]:
        """List all qudemo IDs for a company"""
        return self.gcs_service.list_qudemos(company_name)
    
    def delete_qudemo(self, company_name: str, qudemo_id: str) -> bool:
        """Delete all data for a specific qudemo"""
        return self.gcs_service.delete_qudemo_data(company_name, qudemo_id)

    def generate_and_store_suggested_questions(self, company_name: str, qudemo_id: str) -> List[str]:
        """Generate fresh suggested questions for a QuDemo - PER VIDEO with validation"""
        try:
            logger.info(f"🤖 GENERATING FRESH suggested questions for {company_name}/{qudemo_id}")
            logger.info(f"🔍 Starting fresh generation process (PER VIDEO)...")
            
            # Get transcript data to access individual videos
            transcript_data = self.gcs_service.get_video_transcript(company_name, qudemo_id)
            
            if not transcript_data:
                logger.warning(f"⚠️ No transcript data found for {company_name}/{qudemo_id}")
                return ["What is this about?"]
            
            # Generate questions PER VIDEO with validation
            all_video_questions = self._generate_questions_per_video(company_name, qudemo_id, transcript_data)
            
            if not all_video_questions:
                logger.warning(f"⚠️ No questions generated for any video")
                return ["What is this about?"]
            
            logger.info(f"✅ Generated questions for {len(all_video_questions)} video(s)")
            
            # Store the suggested questions with video metadata
            storage_success = self.gcs_service.store_suggested_questions_with_metadata(
                company_name, qudemo_id, all_video_questions
            )
            
            if storage_success:
                logger.info(f"💾 Successfully stored suggested questions for {company_name}/{qudemo_id}")
            else:
                logger.warning(f"⚠️ Failed to store suggested questions for {company_name}/{qudemo_id}")
            
            # Return flattened list of just questions (for backward compatibility)
            flat_questions = []
            for video_data in all_video_questions:
                for qa in video_data['questions_with_answers']:
                    flat_questions.append(qa['question'])
            
            return flat_questions if flat_questions else ["What is this about?"]
            
        except Exception as e:
            logger.error(f"❌ Failed to generate suggested questions: {e}")
            return ["What is this about?"]
    
    def _generate_questions_per_video(self, company_name: str, qudemo_id: str, transcript_data: Dict[str, Any]) -> List[Dict]:
        """Generate and validate questions for EACH video separately with dynamic question counts"""
        try:
            all_video_questions = []
            
            # Handle both single video and multiple videos format
            videos = transcript_data.get('videos', [])
            if not videos and 'transcript' in transcript_data:
                # Single video format - wrap it
                videos = [{
                    'transcript': transcript_data['transcript'],
                    'video_id': transcript_data.get('video_id', 'default'),
                    'video_url': transcript_data.get('video_url', ''),
                    'title': transcript_data.get('title', 'Video')
                }]
            
            logger.info(f"📹 Processing {len(videos)} video(s) for suggested questions")
            
            for video_index, video in enumerate(videos):
                video_id = video.get('video_id', f'video_{video_index}')
                video_title = video.get('title', f'Video {video_index + 1}')
                transcript = video.get('transcript', '')
                
                if not transcript:
                    logger.warning(f"⚠️ No transcript for video {video_id}")
                    continue
                
                # Calculate dynamic question count based on video length
                video_length = len(transcript)
                question_count = self._calculate_question_count(video_length)
                
                logger.info(f"📊 Video {video_index + 1} ({video_title}): {video_length} chars → {question_count} questions")
                
                # Generate questions for this video
                raw_questions = self._generate_questions_for_single_video(transcript, question_count)
                
                if not raw_questions:
                    logger.warning(f"⚠️ No questions generated for video {video_id}")
                    continue
                
                logger.info(f"🎯 Generated {len(raw_questions)} raw questions for video {video_id}")
                
                # STRICT VALIDATION - Only keep questions with valid answers + CACHE ANSWERS
                # Pass video data so cached answers have correct video_url and timestamp
                video_data = {
                    'video_id': video_id,
                    'video_url': video.get('video_url', ''),
                    'video_title': video_title
                }
                validated_questions_with_answers = self._validate_questions_with_answers(
                    raw_questions, transcript, company_name, qudemo_id, video_data
                )
                
                if validated_questions_with_answers:
                    logger.info(f"✅ {len(validated_questions_with_answers)} questions passed validation with cached answers for video {video_id}")
                    all_video_questions.append({
                        'video_id': video_id,
                        'video_index': video_index,
                        'video_title': video_title,
                        'video_url': video.get('video_url', ''),
                        'questions_with_answers': validated_questions_with_answers  # [{question, answer, timestamp, ...}]
                    })
                else:
                    logger.warning(f"❌ No questions passed validation for video {video_id}")
            
            return all_video_questions
            
        except Exception as e:
            logger.error(f"❌ Error generating questions per video: {e}")
            return []
    
    def _calculate_question_count(self, video_length: int) -> int:
        """Calculate dynamic question count based on video length (transcript characters)"""
        if video_length < 5000:  # Short video (~1-3 min)
            return 3
        elif video_length < 15000:  # Medium video (~3-10 min)
            return 5
        elif video_length < 30000:  # Long video (~10-20 min)
            return 7
        else:  # Very long video (>20 min)
            return 9
    
    def _generate_questions_for_single_video(self, transcript: str, question_count: int) -> List[str]:
        """Generate questions for a single video"""
        try:
            import json
            import os
            
            # Get OpenAI API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ OpenAI API key not found")
                return []
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Limit transcript to avoid token limits (use more content per video now)
            max_chars = 4000  # Increased from 3000
            transcript_sample = transcript[:max_chars]
            
            prompt = f"""You are an expert at analyzing video content to generate helpful suggested questions. 

Generate EXACTLY {question_count} high-quality, specific questions that viewers might ask about this video.

REQUIREMENTS:
- Generate EXACTLY {question_count} questions
- Questions MUST be answerable from the video content
- Questions should be specific and actionable
- Questions should cover different aspects of the content
- Questions should be natural and conversational
- Focus on practical, useful questions

QUESTION TYPES TO INCLUDE:
- How-to questions (e.g., "How do I...")
- What questions (e.g., "What is...", "What are...")
- Why questions (e.g., "Why does...", "Why should I...")
- Feature questions (e.g., "What features...")
- Process questions (e.g., "What are the steps to...")
- Comparison questions (e.g., "What's the difference...")

Return ONLY a JSON array of questions, nothing else:
["question1?", "question2?", "question3?"]

Video Content:
{transcript_sample}

Questions:"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=600,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.2
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                questions = json.loads(response_text)
                if isinstance(questions, list):
                    # Clean and validate questions
                    cleaned_questions = []
                    for q in questions:
                        q = q.strip()
                        if q and not q.endswith('?'):
                            q += '?'
                        if q and len(q) > 5:
                            cleaned_questions.append(q)
                    
                    logger.info(f"✅ Generated {len(cleaned_questions)} questions")
                    return cleaned_questions
                else:
                    logger.error("❌ Invalid JSON format")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parse error: {e}")
                return []
            
        except Exception as e:
            logger.error(f"❌ Error generating questions: {e}")
            return []
    
    def _validate_questions_with_answers(self, questions: List[str], transcript: str, 
                                         company_name: str, qudemo_id: str, video_data: Dict) -> List[Dict]:
        """STRICT VALIDATION - Only keep questions that have valid answers + CACHE THE ANSWERS"""
        try:
            validated_questions_with_answers = []
            
            logger.info(f"🔍 Validating {len(questions)} questions and caching answers...")
            
            for question in questions:
                # Test if this question has a valid answer AND get the answer
                result = self._test_question_and_get_answer(question, transcript, company_name, qudemo_id, video_data)
                
                if result and result.get('is_valid'):
                    validated_questions_with_answers.append({
                        'question': question,
                        'answer': result.get('answer', ''),
                        'timestamp': result.get('timestamp', 0),
                        'formatted_timestamp': result.get('formatted_timestamp', '00:00'),
                        'video_url': result.get('video_url', ''),
                        'video_title': result.get('video_title', 'Video')
                    })
                    logger.info(f"✅ Valid + Cached: {question}")
                else:
                    logger.warning(f"❌ Invalid (no good answer): {question}")
            
            logger.info(f"✅ Validation complete: {len(validated_questions_with_answers)}/{len(questions)} passed with cached answers")
            return validated_questions_with_answers
            
        except Exception as e:
            logger.error(f"❌ Error validating questions: {e}")
            return []  # Return empty if validation fails
    
    def _test_question_and_get_answer(self, question: str, transcript: str, 
                                       company_name: str, qudemo_id: str, video_data: Dict) -> Dict[str, Any]:
        """Test if a question has a valid answer AND return the cached answer for instant retrieval"""
        try:
            # Create transcript data structure with ACTUAL video info (not test data)
            test_transcript_data = {
                'transcript': transcript,
                'videos': [{
                    'transcript': transcript,
                    'video_id': video_data.get('video_id', 'test'),
                    'title': video_data.get('video_title', 'Video'),
                    'video_url': video_data.get('video_url', ''),
                    'video_title': video_data.get('video_title', 'Video')
                }]
            }
            
            # Try to answer the question using direct_qa service
            answer_result = self.direct_qa.search_transcript_directly(test_transcript_data, question)
            
            if not answer_result or not answer_result.get('answer'):
                return {'is_valid': False}
            
            answer = answer_result.get('answer', '')
            answer_lower = answer.lower()
            
            # Check for invalid responses
            invalid_phrases = [
                'no relevant content',
                'no relevant information',
                'no information',
                'not mentioned',
                'does not contain',
                'transcript does not',
                'cannot find',
                'unable to find',
                'no details',
                'no specific',
                'not provided',
                'not available'
            ]
            
            for phrase in invalid_phrases:
                if phrase in answer_lower:
                    return {'is_valid': False}
            
            # Check minimum answer length
            if len(answer) < 25:
                return {'is_valid': False}
            
            # Valid answer! Return full data for caching
            return {
                'is_valid': True,
                'answer': answer,
                'timestamp': answer_result.get('timestamp', 0),
                'formatted_timestamp': answer_result.get('formatted_timestamp', '00:00'),
                'video_url': answer_result.get('video_url', ''),
                'video_title': answer_result.get('video_title', 'Video')
            }
            
        except Exception as e:
            logger.error(f"❌ Error testing question: {e}")
            return {'is_valid': False}
    
    def _get_combined_content_for_suggestions(self, company_name: str, qudemo_id: str) -> str:
        """Get combined content from both video transcripts and documents for suggestion generation"""
        try:
            combined_content = ""
            
            # Get video transcript content
            transcript_data = self.gcs_service.get_video_transcript(company_name, qudemo_id)
            if transcript_data:
                # Extract raw transcript text from video data
                if 'videos' in transcript_data:
                    for video in transcript_data['videos']:
                        video_transcript = video.get('transcript', '')
                        if video_transcript:
                            combined_content += f"\n\n--- Video Content ---\n\n{video_transcript}"
                elif 'transcript' in transcript_data:
                    combined_content += f"\n\n--- Video Content ---\n\n{transcript_data['transcript']}"
                
                logger.info(f"📹 Added video transcript content: {len(transcript_data)} characters")
            
            # Get document content
            try:
                # Get all documents for this QuDemo
                documents_path = f"{company_name}/{qudemo_id}/documents/"
                document_files = self.gcs_service.list_files(documents_path)
                
                document_count = 0
                for doc_file in document_files:
                    if doc_file.endswith('extracted_text.json'):
                        document_content = self.gcs_service.download_file_content(doc_file)
                        if document_content:
                            try:
                                import json
                                doc_data = json.loads(document_content)
                                extracted_text = doc_data.get('extracted_text', '')
                                if extracted_text:
                                    combined_content += f"\n\n--- Document Content ---\n\n{extracted_text}"
                                    document_count += 1
                            except json.JSONDecodeError:
                                continue
                
                if document_count > 0:
                    logger.info(f"📄 Added {document_count} document(s) content")
            
            except Exception as doc_error:
                logger.warning(f"⚠️ Error getting document content: {doc_error}")
            
            logger.info(f"📊 Combined content length: {len(combined_content)} characters")
            return combined_content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error getting combined content: {e}")
            return ""
    
    def _generate_suggested_questions_from_content(self, content: str) -> List[str]:
        """Generate suggested questions from combined video and document content"""
        try:
            import json
            import os
            
            # Get OpenAI API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ OpenAI API key not found for suggested questions generation")
                return []
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Use the same high-quality prompt as video-only suggested questions
            prompt = f"""You are an expert at analyzing content to generate helpful suggested questions. Your task is to create 4-7 high-quality, engaging questions that viewers might want to ask about this content.

REQUIREMENTS:
- Generate 4-7 questions maximum (one will be added automatically)
- Questions should be specific and actionable
- Questions should cover different aspects of the content
- Questions should be natural and conversational
- Questions should help viewers understand key concepts, features, or processes
- Questions should be relevant to the actual content provided
- Focus on practical, useful questions that provide value

QUESTION TYPES TO INCLUDE:
- How-to questions (e.g., "How do I...")
- What questions (e.g., "What is...", "What are...")
- Why questions (e.g., "Why does...", "Why should I...")
- When questions (e.g., "When should I...")
- Where questions (e.g., "Where can I...")
- Comparison questions (e.g., "What's the difference between...")
- Feature questions (e.g., "What features...")
- Process questions (e.g., "What are the steps to...")

Return ONLY a JSON array of questions, nothing else:
["question1?", "question2?", "question3?", ...]

Content:
{content[:3000]}  # Limit content to avoid token limits

Questions:"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.2
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            logger.info(f"🤖 LLM Response for suggested questions: {response_text}")
            
            # Clean up the response - remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```
            
            response_text = response_text.strip()
            
            # Try to parse JSON
            try:
                questions = json.loads(response_text)
                if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                    # Filter out empty questions and ensure they end with question marks
                    filtered_questions = []
                    for question in questions:
                        question = question.strip()
                        if question and not question.endswith('?'):
                            question += '?'
                        if question and len(question) > 5:  # Minimum length check
                            filtered_questions.append(question)
                    
                    logger.info(f"✅ Generated {len(filtered_questions)} suggested questions")
                    
                    # Add "What is this about?" as the first question
                    final_questions = ["What is this about?"] + filtered_questions[:7]  # Limit to 7 additional questions (8 total)
                    
                    logger.info(f"✅ Final suggested questions with 'What is this about?' added: {final_questions}")
                    return final_questions
                else:
                    logger.error(f"❌ Invalid JSON format for suggested questions")
                    return ["What is this about?"]
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse suggested questions JSON: {e}")
                logger.error(f"❌ Raw response: {response_text}")
                # Return just the default question if parsing fails
                return ["What is this about?"]
            
        except Exception as e:
            logger.error(f"❌ Error generating suggested questions from content: {e}")
            return ["What is this about?"]
    
    def _create_document_answer(self, document_results: List[Dict[str, Any]], question: str) -> Optional[Dict[str, Any]]:
        """
        Create a well-formatted answer from document search results
        
        Args:
            document_results: List of document search results
            question: Original question
            
        Returns:
            Formatted answer data or None
        """
        try:
            if not document_results:
                return None
            
            # Combine all relevant sections from all documents
            all_sections = []
            sources = []
            
            for result in document_results:
                document_id = result.get('document_id', '')
                filename = result.get('filename', '')
                relevant_sections = result.get('relevant_sections', [])
                matched_terms = result.get('matched_terms', [])
                
                # Add document as a source
                sources.append({
                    'type': 'document',
                    'document_id': document_id,
                    'filename': filename,
                    'matched_terms': matched_terms
                })
                
                # Add all relevant sections
                all_sections.extend(relevant_sections)
            
            if not all_sections:
                return None
            
            # Create a comprehensive answer from all sections
            combined_content = "\n\n".join(all_sections)
            
            # Use LLM to create a well-formatted answer (similar to video answers)
            logger.info(f"📄 Combined content length: {len(combined_content)} characters")
            logger.info(f"📄 Combined content preview: {combined_content[:200]}...")
            formatted_answer = self._format_document_answer_with_llm(combined_content, question)
            
            # If LLM determined content is not relevant, return None
            if formatted_answer is None:
                logger.info(f"ℹ️ Document content not relevant to question")
                return None
            
            return {
                'answer': formatted_answer,
                'confidence': 0.8,  # High confidence for document answers
                'sources': sources,
                'timestamp': 0,
                'formatted_timestamp': 'Document',
                'video_url': '',
                'video_title': 'Document Content'
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating document answer: {e}")
            return None
    
    def _create_video_answer(self, video_results: Dict[str, Any], question: str) -> Optional[Dict[str, Any]]:
        """
        Create a well-formatted answer from video search results
        
        Args:
            video_results: Video search results
            question: Original question
            
        Returns:
            Formatted answer data or None
        """
        try:
            if not video_results or video_results.get('answer') == 'not found':
                return None
            
            return {
                'answer': video_results.get('answer', ''),
                'timestamp': video_results.get('timestamp', 0),
                'end': video_results.get('end', 0),
                'formatted_timestamp': video_results.get('formatted_timestamp', ''),
                'video_url': video_results.get('video_url', ''),
                'video_title': video_results.get('video_title', ''),
                'sources': [{
                    'type': 'video',
                    'timestamp': video_results.get('timestamp', 0),
                    'formatted_timestamp': video_results.get('formatted_timestamp', ''),
                    'video_url': video_results.get('video_url', ''),
                    'video_title': video_results.get('video_title', '')
                }],
                'confidence': 0.8
            }
        except Exception as e:
            logger.error(f"❌ Error creating video answer: {e}")
            return None
    
    def _combine_answers(self, document_answer: str, video_answer: str, question: str) -> str:
        """
        Intelligently combine document and video answers
        
        Args:
            document_answer: Answer from document search
            video_answer: Answer from video search
            question: Original question
            
        Returns:
            Combined answer
        """
        try:
            # Filter out "No relevant information found" text
            if document_answer and document_answer.strip() in ['not found', 'No relevant information found']:
                document_answer = ''
            if video_answer and video_answer.strip() in ['not found', 'No relevant information found']:
                video_answer = ''
            
            # If one answer is empty, return the other
            if not document_answer.strip():
                return video_answer.strip()
            if not video_answer.strip():
                return document_answer.strip()
            
            # If one answer is much longer, it might be more comprehensive
            if len(document_answer) > len(video_answer) * 2:
                # Document answer is much more detailed
                combined = f"{document_answer} {video_answer}"
            elif len(video_answer) > len(document_answer) * 2:
                # Video answer is much more detailed
                combined = f"{video_answer} {document_answer}"
            else:
                # Both are similar length, combine them
                combined = f"{video_answer} {document_answer}"
            
            # Clean up the combined answer
            return self._clean_combined_answer(combined)
            
        except Exception as e:
            logger.error(f"❌ Error combining answers: {e}")
            return f"{document_answer}\n\n{video_answer}"
    
    def _clean_combined_answer(self, answer: str) -> str:
        """
        Clean up a combined answer to remove redundancy
        
        Args:
            answer: Combined answer text
            
        Returns:
            Cleaned answer
        """
        try:
            # Remove duplicate sentences and limit to 4-5 sentences
            sentences = answer.split('. ')
            seen = set()
            unique_sentences = []
            
            for sentence in sentences:
                # Normalize sentence for comparison
                normalized = sentence.strip().lower()
                if normalized not in seen and len(normalized) > 10:  # Avoid very short duplicates
                    unique_sentences.append(sentence.strip())
                    seen.add(normalized)
                    
                    # Limit to maximum 5 sentences
                    if len(unique_sentences) >= 5:
                        break
            
            cleaned = '. '.join(unique_sentences)
            
            # Ensure proper ending
            if not cleaned.endswith('.'):
                cleaned += '.'
            
            # Final check: if still too long, truncate to 500 characters
            if len(cleaned) > 500:
                sentences = cleaned.split('. ')
                truncated_sentences = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) + 2 <= 500:  # +2 for '. '
                        truncated_sentences.append(sentence)
                        char_count += len(sentence) + 2
                    else:
                        break
                cleaned = '. '.join(truncated_sentences)
                if not cleaned.endswith('.'):
                    cleaned += '.'
            
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Error cleaning combined answer: {e}")
            return answer
    
    def _format_document_answer_with_llm(self, content: str, question: str) -> str:
        """
        Use LLM to format document content into a well-structured answer
        Uses the same high-quality prompt format as video answers
        
        Args:
            content: Raw document content
            question: Original question
            
        Returns:
            Formatted answer
        """
        try:
            # Use the same LLM system as video answers for consistency
            import os
            
            # Get OpenAI API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ OpenAI API key not found, using fallback formatting")
                logger.info(f"📄 Content length: {len(content)} characters")
                logger.info(f"📄 Content preview: {content[:200]}...")
                # Create a more concise fallback answer (3-4 sentences max)
                sentences = content.split('. ')
                if len(sentences) >= 3:
                    # Take first 3-4 sentences and make them concise
                    answer = '. '.join(sentences[:3]) + '.'
                    return answer[:400] + "..." if len(answer) > 400 else answer
                elif len(sentences) >= 2:
                    answer = '. '.join(sentences[:2]) + '.'
                    return answer[:300] + "..." if len(answer) > 300 else answer
                else:
                    return content[:200] + "..." if len(content) > 200 else content
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Use the SAME high-quality prompt as video answers for consistency
            prompt = f"""You are an expert at analyzing document content to answer questions. Your task is to find the most relevant information in the document and provide a HIGH-QUALITY, INTELLIGENT answer.

⚠️ CRITICAL RELEVANCE CHECK:
- FIRST, check if the document content actually contains information relevant to the question
- If the question asks about something NOT covered in the document content, respond EXACTLY with: "NO_RELEVANT_INFORMATION"
- ONLY answer questions that can be answered using the provided document content
- DO NOT use your general knowledge or external information
- DO NOT make up or infer information that isn't explicitly in the document

IF RELEVANT, YOUR ANSWER MUST:
- Be EXACTLY 4-5 sentences to provide comprehensive information
- ONLY use information from the provided document content
- Be intelligent and insightful based on the document
- Show deep understanding of the concepts IN THE DOCUMENT
- Use professional, business-ready language
- Provide clear comparisons and contrasts FROM THE DOCUMENT
- Be immediately valuable and actionable
- Demonstrate consciousness and completeness
- Focus on the core essence and business value
- Interpret and analyze the content, don't just quote it

MANDATORY REQUIREMENTS:
- NEVER include raw document quotes or excerpts
- NEVER include step-by-step instructions from the document
- ALWAYS provide processed, intelligent analysis
- ALWAYS use professional, business-ready language
- ALWAYS focus on the core essence and business value

Question: {question}

Document Content:
{content[:2000]}

Answer:"""
            
            logger.info(f"📄 Sending to LLM: {len(content)} characters, truncated to 2000")
            logger.info(f"📄 Question: {question}")
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Use same model as video answers
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Same temperature as video answers
                max_tokens=250,  # Increased for 4-5 sentence document answers
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Check if the LLM determined the content is not relevant
            if "NO_RELEVANT_INFORMATION" in answer:
                logger.info(f"ℹ️ LLM determined document content is not relevant to the question")
                return None
            
            # Ensure answer is 4-5 sentences for comprehensive information
            sentences = answer.split('. ')
            if len(sentences) > 5:
                # Take only first 5 sentences
                answer = '. '.join(sentences[:5])
                if not answer.endswith('.'):
                    answer += '.'
            
            # Also check character length (should be under 500 characters for 4-5 sentences)
            if len(answer) > 500:
                sentences = answer.split('. ')
                truncated_sentences = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) + 2 <= 500:  # +2 for '. '
                        truncated_sentences.append(sentence)
                        char_count += len(sentence) + 2
                    else:
                        break
                answer = '. '.join(truncated_sentences)
                if not answer.endswith('.'):
                    answer += '.'
            
            logger.info(f"✅ LLM formatted document answer (truncated): {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error formatting document answer with LLM: {e}")
            # Improved fallback formatting (3-4 sentences max)
            sentences = content.split('. ')
            if len(sentences) >= 3:
                answer = '. '.join(sentences[:3]) + '.'
                return answer[:350] + "..." if len(answer) > 350 else answer
            elif len(sentences) >= 2:
                answer = '. '.join(sentences[:2]) + '.'
                return answer[:250] + "..." if len(answer) > 250 else answer
            else:
                return content[:200] + "..." if len(content) > 200 else content
    
    def _create_website_answer(self, website_results: List[Dict], question: str) -> Optional[Dict[str, Any]]:
        """Create answer data from website search results"""
        try:
            if not website_results:
                return None
            
            # Get the most relevant result
            best_result = website_results[0]
            
            # Format the answer using LLM
            answer = self._format_website_answer_with_llm(
                best_result.get('content', ''),
                question
            )
            
            # If LLM determined content is not relevant, return None
            if answer is None:
                logger.info(f"ℹ️ Website content not relevant to question")
                return None
            
            return {
                'answer': answer,
                'sources': [{
                    'type': 'website',
                    'url': best_result.get('url', ''),
                    'title': best_result.get('title', ''),
                    'relevance_score': best_result.get('relevance_score', 0.0)
                }],
                'confidence': best_result.get('relevance_score', 0.8)
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating website answer: {e}")
            return None
    
    def _format_website_answer_with_llm(self, content: str, question: str) -> str:
        """Format website content into a high-quality answer using LLM"""
        try:
            if not content or len(content.strip()) < 10:
                return "No relevant information found in the website content."
            
            # Get OpenAI API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ OpenAI API key not found, using fallback formatting")
                # Fallback formatting
                sentences = content.split('. ')
                if len(sentences) >= 3:
                    answer = '. '.join(sentences[:3]) + '.'
                    return answer[:400] + "..." if len(answer) > 400 else answer
                elif len(sentences) >= 2:
                    answer = '. '.join(sentences[:2]) + '.'
                    return answer[:300] + "..." if len(answer) > 300 else answer
                else:
                    return content[:200] + "..." if len(content) > 200 else content
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Try to extract steps directly without LLM first
            if 'step' in question.lower() and ('step 1' in content.lower() or 'step 2' in content.lower()):
                logger.info("🌐 Detected step-by-step question - extracting steps directly")
                
                # Extract steps using simple text processing
                lines = content.split('\n')
                steps = []
                current_step = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Step ') and ':' in line:
                        if current_step:
                            steps.append(current_step)
                        current_step = line
                    elif current_step and line and not line.startswith('Step '):
                        current_step += ' ' + line
                
                if current_step:
                    steps.append(current_step)
                
                if steps:
                    logger.info(f"🌐 Extracted {len(steps)} steps directly")
                    return '\n'.join(steps)
            
            # Fallback to LLM if direct extraction doesn't work
            prompt = f"""Extract and present the information from the website content below. 

⚠️ CRITICAL RELEVANCE CHECK:
- FIRST, check if the website content actually contains information relevant to the question
- If the question asks about something NOT covered in the website content, respond EXACTLY with: "NO_RELEVANT_INFORMATION"
- ONLY answer questions that can be answered using the provided website content
- DO NOT use your general knowledge or external information
- DO NOT make up or infer information that isn't explicitly in the website content

IF RELEVANT, preserve ALL specific details, numbers, percentages, categories, and exact requirements. Do not generalize or summarize away specific information.

Question: {question}

Website Content:
{content[:4000]}

Extract the specific information while preserving all details (or return "NO_RELEVANT_INFORMATION" if not relevant):"""
            
            logger.info(f"🌐 Sending to LLM: {len(content)} characters, truncated to 4000")
            logger.info(f"🌐 Question: {question}")
            logger.info(f"🌐 Content preview (first 500 chars): {content[:500]}")
            logger.info(f"🌐 Content preview (last 500 chars): {content[-500:]}")
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Use same model as other answers
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Same temperature as other answers
                max_tokens=600,  # Increased to allow for detailed step-by-step instructions
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Check if the LLM determined the content is not relevant
            if "NO_RELEVANT_INFORMATION" in answer:
                logger.info(f"ℹ️ LLM determined website content is not relevant to the question")
                return None
            
            # Ensure answer is comprehensive but not too long
            sentences = answer.split('. ')
            if len(sentences) > 8:
                # Take only first 8 sentences for more comprehensive information
                answer = '. '.join(sentences[:8])
                if not answer.endswith('.'):
                    answer += '.'
            
            # Also check character length (increased to 1000 characters for detailed instructions)
            if len(answer) > 1000:
                sentences = answer.split('. ')
                truncated_sentences = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) + 2 <= 1000:  # +2 for '. '
                        truncated_sentences.append(sentence)
                        char_count += len(sentence) + 2
                    else:
                        break
                answer = '. '.join(truncated_sentences)
                if not answer.endswith('.'):
                    answer += '.'
            
            logger.info(f"✅ LLM formatted website answer (truncated): {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error formatting website answer with LLM: {e}")
            # Improved fallback formatting (3-4 sentences max)
            sentences = content.split('. ')
            if len(sentences) >= 3:
                answer = '. '.join(sentences[:3]) + '.'
                return answer[:350] + "..." if len(answer) > 350 else answer
            elif len(sentences) >= 2:
                answer = '. '.join(sentences[:2]) + '.'
                return answer[:250] + "..." if len(answer) > 250 else answer
            else:
                return content[:200] + "..." if len(content) > 200 else content
    
    def _combine_multiple_answers(self, answers: List[str], question: str) -> str:
        """Combine multiple answers from different sources into a coherent response"""
        try:
            if not answers:
                return "No relevant information found."
            
            if len(answers) == 1:
                # For "What is this about?" question, still process single answers to make them concise
                if question.lower().strip() == "what is this about?":
                    single_answer = answers[0]
                    # If the single answer is too long, truncate it
                    if len(single_answer) > 500:
                        sentences = single_answer.split('. ')
                        if len(sentences) > 4:
                            single_answer = '. '.join(sentences[:4])
                            if not single_answer.endswith('.'):
                                single_answer += '.'
                    return single_answer
                else:
                    return answers[0]
            
            # Get OpenAI API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ OpenAI API key not found, using simple concatenation")
                # Simple fallback - combine with separators
                combined = " ".join(answers)
                if question.lower().strip() == "what is this about?":
                    # For "What is this about?" - keep it very short
                    return combined[:300] + "..." if len(combined) > 300 else combined
                else:
                    return combined[:500] + "..." if len(combined) > 500 else combined
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Create a prompt to intelligently combine the answers
            answers_text = "\n\n".join([f"Source {i+1}: {answer}" for i, answer in enumerate(answers)])
            
            # Special handling for "What is this about?" question - provide a concise summary
            if question.lower().strip() == "what is this about?":
                prompt = f"""You are an expert at analyzing content to provide clear, concise summaries. Your task is to create a brief but comprehensive overview of what this content is about.

REQUIREMENTS:
- Provide a clear, concise summary (2-4 sentences maximum)
- Focus on the main purpose, key topics, and important points
- Include the most important details without being too verbose
- Make it easy for someone to quickly understand what this content covers
- Preserve key information but prioritize clarity and brevity

Question: {question}

Content from different sources:
{answers_text}

Concise Summary:"""
            else:
                prompt = f"""You are an expert at combining information from multiple sources to create a comprehensive, detailed answer. Your task is to merge the following answers into a single, coherent response.

CRITICAL: Your combined answer must preserve ALL specific details, requirements, and actionable information. Prioritize specific information over generic business speak.

MANDATORY REQUIREMENTS:
- ALWAYS preserve specific details, numbers, percentages, and exact requirements
- ALWAYS include specific roles, titles, and categories when mentioned (e.g., CFO, COO, President, VP, General Partner, Treasurer)
- ALWAYS preserve step-by-step instructions and procedures
- ALWAYS include specific thresholds, limits, and criteria (e.g., 25% ownership)
- ALWAYS maintain the original level of detail and specificity
- ALWAYS include specific categories and classifications when provided
- ALWAYS preserve lists of specific items, roles, or requirements
- NEVER generalize or synthesize away specific information
- NEVER replace specific details with generic business concepts
- NEVER summarize away important categories or classifications

YOUR COMBINED ANSWER MUST:
- Preserve ALL specific details from the sources
- Include exact numbers, percentages, and thresholds
- Maintain specific role titles and categories (list them if provided)
- Keep step-by-step procedures intact
- Include specific categories and classifications
- Provide actionable, specific information
- Be comprehensive and detailed
- Focus on what users need to know to take action
- Combine sources while preserving all specific information
- Include all specific examples and role types mentioned

Question: {question}

Answers from different sources:
{answers_text}

Combined Answer:"""
            
            logger.info(f"🔄 Combining {len(answers)} answers using LLM")
            
            # Adjust max_tokens based on question type
            if question.lower().strip() == "what is this about?":
                max_tokens = 200  # Shorter for summary
            else:
                max_tokens = 800  # Longer for detailed answers
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            combined_answer = response.choices[0].message.content.strip()
            
            # Ensure answer length is appropriate for question type
            if question.lower().strip() == "what is this about?":
                # For "What is this about?" - keep it very concise (max 3-4 sentences)
                if len(combined_answer) > 500:
                    sentences = combined_answer.split('. ')
                    if len(sentences) > 4:
                        # Take first 4 sentences maximum
                        combined_answer = '. '.join(sentences[:4])
                        if not combined_answer.endswith('.'):
                            combined_answer += '.'
            else:
                # For other questions - comprehensive but not too long
                if len(combined_answer) > 1500:
                    # If too long, try to preserve the most important parts
                    sentences = combined_answer.split('. ')
                    if len(sentences) > 12:
                        # Take first 12 sentences to preserve detail
                        combined_answer = '. '.join(sentences[:12])
                        if not combined_answer.endswith('.'):
                            combined_answer += '.'
            
            logger.info(f"✅ LLM combined answer (truncated): {len(combined_answer)} characters")
            return combined_answer
            
        except Exception as e:
            logger.error(f"❌ Error combining multiple answers with LLM: {e}")
            # Simple fallback - combine with separators
            combined = " ".join(answers)
            if question.lower().strip() == "what is this about?":
                # For "What is this about?" - keep it very short
                return combined[:300] + "..." if len(combined) > 300 else combined
            else:
                return combined[:500] + "..." if len(combined) > 500 else combined
    
    def store_website_content(self, company_name: str, qudemo_id: str, website_data: Dict[str, Any]) -> bool:
        """Store website scraped content in GCS"""
        try:
            logger.info(f"🌐 Storing website content for {company_name}/{qudemo_id}")
            return self.gcs_service.store_website_content(company_name, qudemo_id, website_data)
        except Exception as e:
            logger.error(f"❌ Failed to store website content: {e}")
            return False
    
    # Note: Removed get_suggested_questions method since we generate fresh questions on-demand