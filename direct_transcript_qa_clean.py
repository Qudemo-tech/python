"""
Direct Transcript Q&A System
Queries directly through raw transcript data instead of chunks
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple

class DirectTranscriptQA:
    def __init__(self):
        pass
    
    def search_transcript_directly(self, transcript_data: Dict[str, Any], question: str) -> Optional[Dict[str, Any]]:
        """Search transcript using LLM with full raw transcript data"""
        try:
            # Get the raw transcript text
            raw_transcript = self._get_raw_transcript_text(transcript_data)
            if not raw_transcript:
                print(f"❌ No raw transcript found in data: {transcript_data.keys()}")
                return None
            
            print(f"🔍 Using LLM to search full transcript for question: {question}")
            print(f"📄 Raw transcript length: {len(raw_transcript)} characters")
            
            # Use LLM to find the answer with timestamp
            llm_result = self._ask_llm_for_answer(raw_transcript, question)
            
            if not llm_result or llm_result.get('answer') == 'not found':
                return {
                    'answer': 'No relevant information found',
                    'timestamp': 0,
                    'formatted_timestamp': '00:00-00:00',
                    'confidence': 0.0,
                    'sources': [],
                    'video_url': '',
                    'video_title': 'No relevant content'
                }
            
            # Use video info from LLM response if available, otherwise fallback to first video
            if llm_result.get('video_url') and llm_result.get('video_url') != 'not found':
                video_url = llm_result['video_url']
                video_title = llm_result.get('video_title', 'Unknown')
            else:
                # Fallback to first video info
                video_info = self._get_video_info(transcript_data)
                video_url = video_info['video_url']
                video_title = video_info['video_title']
            
            # Use the LLM's answer directly without overwriting it
            # The LLM already provides a complete, accurate answer
            formatted_answer = llm_result['answer']
            
            return {
                'answer': formatted_answer,
                'timestamp': self._parse_timestamp(llm_result.get('closest_timestamp', '00:00')),
                'formatted_timestamp': llm_result.get('closest_timestamp', '00:00'),
                'confidence': 1.0,  # LLM confidence
                'sources': [{'text': llm_result['answer'][:500] + '...' if len(llm_result['answer']) > 500 else llm_result['answer']}],
                'video_url': video_url,  # ✅ Specific video URL from LLM
                'video_title': video_title  # ✅ Specific video title from LLM
            }
            
        except Exception as e:
            print(f"❌ Failed to search transcript with LLM: {e}")
            return None
    
    def _get_raw_transcript_text(self, transcript_data: Dict[str, Any]) -> str:
        """Extract raw transcript text from transcript data with video identifiers"""
        try:
            if 'videos' in transcript_data and transcript_data['videos']:
                # Multi-video format - combine all transcripts with video identifiers
                all_transcripts = []
                for i, video in enumerate(transcript_data['videos']):
                    raw_transcript = video.get('transcript', '')
                    video_url = video.get('video_url', '')
                    video_title = video.get('video_title', f'Video {i+1}')
                    
                    if raw_transcript:
                        # Add video identifier to each transcript
                        video_header = f"=== VIDEO {i+1}: {video_title} ===\n{video_url}\n\n"
                        all_transcripts.append(video_header + raw_transcript)
                
                return '\n\n'.join(all_transcripts)
            else:
                # Single video format (legacy)
                return transcript_data.get('transcript', '')
        except Exception as e:
            print(f"❌ Error extracting raw transcript: {e}")
            return ""
    
    def _get_video_info(self, transcript_data: Dict[str, Any]) -> Dict[str, str]:
        """Get video URL and title from transcript data"""
        try:
            if 'videos' in transcript_data and transcript_data['videos']:
                # Multi-video format - get first video info
                first_video = transcript_data['videos'][0]
                return {
                    'video_url': first_video.get('video_url', ''),
                    'video_title': first_video.get('video_title', 'Unknown')
                }
            else:
                # Single video format (legacy)
                return {
                    'video_url': transcript_data.get('video_url', ''),
                    'video_title': transcript_data.get('video_title', 'Unknown')
                }
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return {'video_url': '', 'video_title': 'Unknown'}
    
    def _ask_llm_for_answer(self, transcript: str, question: str) -> Optional[Dict[str, str]]:
        """Ask LLM to find answer with timestamp and video info from full transcript"""
        try:
            import openai
            import os
            import json
            
            # Get OpenAI API key
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                print("❌ OPENAI_API_KEY not found")
                return None
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Create the prompt for multi-video support
            prompt = f"""You are an expert at analyzing video transcripts to answer questions. Your task is to find the most relevant information in the transcript and provide a HIGH-QUALITY, COMPREHENSIVE answer that matches ChatGPT's business-focused style.

CRITICAL: Your answer must be COMPREHENSIVE but CONCISE - 2-3 sentences maximum. Think like ChatGPT - business-focused, persuasive, and comprehensive.

IMPORTANT: Copy the exact structure and phrasing from the example above. Use the same words, phrases, and sentence structure as ChatGPT.

EXAMPLE OF EXCELLENT ANSWER FORMAT:
"The key difference is that 'Text-to-SQL' simply translates a natural language query into a SQL statement, returning raw data without interpretation. In contrast, our AI delivers 'Text-to-Insight,' which means it not only processes the query but also interprets the results, highlights patterns, and surfaces actionable insights in plain language. This ensures your users don't just get rows of data—they get meaningful, context-aware answers that help them make better decisions instantly."

CRITICAL STYLE REQUIREMENTS:
- Start with "The key difference is that" (not "While")
- Use "simply" (not "merely") 
- Include "not only processes the query but also"
- Use "your users" (not just "users")
- Include "don't just get rows of data—they get"
- End with "help them make better decisions instantly"

YOUR ANSWER MUST:
- Be 2-3 sentences maximum
- Start with "The key difference is that" (exactly)
- Use "simply" (not "merely" or "just")
- Include "not only processes the query but also" (exactly)
- Use "your users" (not just "users")
- Include "don't just get rows of data—they get" (exactly)
- End with "help them make better decisions instantly" (exactly)
- Match ChatGPT's exact phrasing and structure

Return ONLY valid JSON in the format:
{{
  "closest_timestamp": "<timestamp or 'not found'>",
  "answer": "<EXTREMELY SHORT high-quality answer or 'not found'>",
  "video_url": "<video URL or 'not found'>",
  "video_title": "<video title or 'not found'>"
}}

Rules:
- MAXIMUM 1-2 sentences for the answer
- Be direct and concise like ChatGPT
- Focus on the key difference only
- If no relevant information exists, return "not found" for all fields
- Do not include explanations or commentary outside JSON
- Ensure JSON is syntactically valid
- Include the video URL and title from the video that contains the answer
- Use the exact video URL and title from the transcript headers



transcript:
{transcript}

question: {question}"""

            # Call OpenAI API with optimized parameters for ChatGPT-quality responses
            response = client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4o for highest quality
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower temperature for more focused, consistent responses
                max_tokens=300,   # Increased for comprehensive business-focused answers
                top_p=0.95,      # High precision for quality
                frequency_penalty=0.2,  # Penalty to reduce repetition
                presence_penalty=0.1    # Encourage new concepts
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM Response: {response_text}")
            
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
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse LLM JSON response: {e}")
                print(f"❌ Raw response: {response_text}")
                return None
            
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            return None
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string to seconds"""
        try:
            if timestamp_str == 'not found' or not timestamp_str:
                return 0.0
            
            # Handle range format like "00:01-00:07" - use the start time
            if '-' in timestamp_str and ':' in timestamp_str:
                start_time = timestamp_str.split('-')[0].strip()
                timestamp_str = start_time
            
            # Handle MM:SS format
            if ':' in timestamp_str:
                parts = timestamp_str.split(':')
                if len(parts) == 2:  # MM:SS
                    minutes, seconds = map(float, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(float, parts)
                    return hours * 3600 + minutes * 60 + seconds
            
            # Handle plain number (seconds)
            return float(timestamp_str)
            
        except Exception as e:
            print(f"❌ Error parsing timestamp '{timestamp_str}': {e}")
            return 0.0
