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
            # Check if this is a casual greeting or small talk
            if self._is_casual_greeting(question):
                print(f"👋 Detected casual greeting: {question}")
                return self._handle_casual_greeting(transcript_data)
            
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
            
            # Post-process to ensure no raw transcript content
            if self._contains_raw_transcript_content(formatted_answer):
                print("⚠️ Detected raw transcript content, requesting reprocessing...")
                # Try to get a better answer by being more specific
                reprocessed_result = self._ask_llm_for_answer_reprocessed(raw_transcript, question)
                if reprocessed_result and reprocessed_result.get('answer') != 'not found':
                    formatted_answer = reprocessed_result['answer']
            
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
            prompt = f"""You are an expert at analyzing video transcripts to answer questions. Your task is to find the most relevant information in the transcript and provide a HIGH-QUALITY, INTELLIGENT answer.

CRITICAL: Your answer must be EXACTLY 4-5 sentences to provide comprehensive information. Each sentence should be detailed and informative.

MANDATORY REQUIREMENTS:
- NEVER include raw transcript quotes like "Hey there" or "Great question"
- NEVER include step-by-step instructions from the transcript
- ALWAYS provide processed, intelligent analysis
- ALWAYS use professional, business-ready language
- ALWAYS focus on the core essence and business value

YOUR ANSWER MUST:
- Be EXACTLY 4-5 sentences to provide comprehensive information - Each sentence should be detailed and informative
- Be intelligent and insightful (like ChatGPT)
- Show deep understanding of the concepts
- Use professional, business-ready language
- Provide clear comparisons and contrasts
- Be immediately valuable and actionable
- Demonstrate consciousness and completeness
- Focus on the core essence and business value
- Interpret and analyze the content, don't just quote it
- Provide processed insights, not raw transcript text

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
                max_tokens=500,   # Increased to allow longer 4-5 sentence responses
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
                
                # Validate and truncate the answer if needed
                if result.get('answer') and result.get('answer') != 'not found':
                    answer = result['answer']
                    
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
                    
                    result['answer'] = answer
                    print(f"✅ Answer truncated to {len(answer)} characters: {answer[:100]}...")
                
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse LLM JSON response: {e}")
                print(f"❌ Raw response: {response_text}")
                
                # Try to fix common JSON issues
                try:
                    # Try to fix unterminated strings by adding missing quotes
                    if '"video_url":' in response_text and not response_text.strip().endswith('"'):
                        # Find the last incomplete video_url and fix it
                        last_video_url = response_text.rfind('"video_url":')
                        if last_video_url != -1:
                            # Extract the video URL part and fix it
                            url_part = response_text[last_video_url + len('"video_url":'):].strip()
                            if url_part.startswith('"') and not url_part.endswith('"'):
                                # Remove the opening quote and add closing quote and brace
                                url_value = url_part[1:].strip()
                                fixed_response = response_text[:last_video_url + len('"video_url":') + 1] + url_value + '"}'
                                print(f"🔧 Attempting to fix JSON: {fixed_response}")
                                result = json.loads(fixed_response)
                                return result
                except:
                    pass
                
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
    
    def _is_casual_greeting(self, question: str) -> bool:
        """Check if the question is a casual greeting or small talk"""
        question_lower = question.lower().strip()
        
        # Common casual greetings and small talk
        casual_greetings = [
            "hi", "hello", "hey", "hiya", "howdy",
            "good morning", "good afternoon", "good evening",
            "what's up", "whats up", "sup", "wassup",
            "how are you", "how are you doing", "how's it going",
            "nice to meet you", "pleasure to meet you",
            "thanks", "thank you", "thx", "ty",
            "bye", "goodbye", "see you later", "talk to you later",
            "ok", "okay", "alright", "sure", "yes", "no",
            "cool", "awesome", "great", "nice", "good",
            "lol", "haha", "hehe", "😊", "😄", "👍",
            "how do you do", "howdy", "greetings",
            "what's happening", "whats happening",
            "how's everything", "hows everything",
            "what's new", "whats new", "what's going on", "whats going on"
        ]
        
        # Check for exact matches
        if question_lower in casual_greetings:
            return True
        
        # Check for partial matches (greeting + additional text)
        for greeting in casual_greetings:
            if question_lower.startswith(greeting + " ") or question_lower.startswith(greeting + ","):
                return True
        
        # Check for very short questions (likely casual)
        if len(question_lower.split()) <= 2 and any(word in casual_greetings for word in question_lower.split()):
            return True
        
        return False
    
    def _handle_casual_greeting(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle casual greetings with a friendly response"""
        try:
            # Get video info for the response
            video_info = self._get_video_info(transcript_data)
            
            return {
                'answer': 'Hi! I am an AI assistant for this demo. You can ask me anything related to this video demo and I can help you with answers.',
                'timestamp': 0,
                'formatted_timestamp': '00:00',
                'confidence': 1.0,
                'sources': [],
                'video_url': video_info['video_url'],
                'video_title': video_info['video_title']
            }
        except Exception as e:
            print(f"❌ Error handling casual greeting: {e}")
            return {
                'answer': 'Hi! I am an AI assistant for this demo. You can ask me anything related to this video demo and I can help you with answers.',
                'timestamp': 0,
                'formatted_timestamp': '00:00',
                'confidence': 1.0,
                'sources': [],
                'video_url': '',
                'video_title': 'Demo Video'
            }
    
    def generate_suggested_questions(self, transcript_data: Dict[str, Any]) -> List[str]:
        """Generate suggested questions from the transcript content"""
        try:
            # Get the raw transcript text
            raw_transcript = self._get_raw_transcript_text(transcript_data)
            if not raw_transcript:
                print(f"❌ No raw transcript found for suggested questions generation")
                return []
            
            print(f"🤖 Generating suggested questions from transcript...")
            print(f"📄 Raw transcript length: {len(raw_transcript)} characters")
            
            # Use LLM to generate suggested questions
            suggested_questions = self._ask_llm_for_suggested_questions(raw_transcript)
            
            if suggested_questions:
                print(f"✅ Generated {len(suggested_questions)} suggested questions")
                
                # Validate each question to ensure it has a good answer
                validated_questions = self._validate_suggested_questions(suggested_questions, transcript_data)
                
                if validated_questions:
                    print(f"✅ Validated {len(validated_questions)} suggested questions")
                    return validated_questions
                else:
                    print(f"⚠️ No valid suggested questions after validation")
                    return []
            else:
                print(f"⚠️ No suggested questions generated")
                return []
            
        except Exception as e:
            print(f"❌ Failed to generate suggested questions: {e}")
            return []
    
    def _validate_suggested_questions(self, questions: List[str], transcript_data: Dict[str, Any]) -> List[str]:
        """Validate suggested questions to ensure they have good answers"""
        try:
            validated_questions = []
            
            for question in questions:
                print(f"🔍 Validating question: {question}")
                
                # Test if the question has a good answer
                answer_result = self.answer_question(question, transcript_data)
                
                # Check if the answer is valid (not "no relevant content")
                if (answer_result and 
                    answer_result.get('answer') and 
                    'no relevant content' not in answer_result.get('answer', '').lower() and
                    'no relevant information' not in answer_result.get('answer', '').lower() and
                    len(answer_result.get('answer', '')) > 20):  # Minimum answer length
                    
                    validated_questions.append(question)
                    print(f"✅ Question validated: {question}")
                else:
                    print(f"❌ Question rejected: {question} - Answer: {answer_result.get('answer', 'No answer')}")
            
            return validated_questions
            
        except Exception as e:
            print(f"❌ Error validating suggested questions: {e}")
            return questions  # Return original questions if validation fails
    
    def _ask_llm_for_suggested_questions(self, transcript: str) -> List[str]:
        """Ask LLM to generate suggested questions from the transcript"""
        try:
            import openai
            import os
            import json
            
            # Get OpenAI API key
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                print("❌ OPENAI_API_KEY not found")
                return []
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Create the prompt for generating suggested questions
            prompt = f"""You are an expert at analyzing video transcripts to generate helpful suggested questions. Your task is to create 5-8 high-quality, engaging questions that viewers might want to ask about this video content.

REQUIREMENTS:
- Generate 5-8 questions maximum
- Questions should be specific and actionable
- Questions should cover different aspects of the content
- Questions should be natural and conversational
- Questions should help viewers understand key concepts, features, or processes
- Questions should be relevant to the actual content in the transcript
- Avoid generic questions like "What is this about?"
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

Return ONLY a JSON array of question strings:
[
  "Question 1?",
  "Question 2?",
  "Question 3?",
  "Question 4?",
  "Question 5?"
]

Rules:
- Return ONLY the JSON array, no other text
- Each question should end with a question mark
- Questions should be 10-20 words long
- Make questions specific to the content
- Ensure JSON is syntactically valid
- Do not include explanations or commentary

transcript:
{transcript}"""

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Higher temperature for more creative questions
                max_tokens=500,   # Enough for 5-8 questions
                top_p=0.9,
                frequency_penalty=0.3,  # Encourage variety
                presence_penalty=0.2
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM Response for suggested questions: {response_text}")
            
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
                    
                    return filtered_questions[:8]  # Limit to 8 questions max
                else:
                    print(f"❌ Invalid JSON format for suggested questions")
                    return []
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse suggested questions JSON: {e}")
                print(f"❌ Raw response: {response_text}")
                return []
            
        except Exception as e:
            print(f"❌ Error calling LLM for suggested questions: {e}")
            return []
    
    def _contains_raw_transcript_content(self, answer: str) -> bool:
        """Check if answer contains raw transcript content that should be processed"""
        raw_indicators = [
            "Hey there",
            "Great question",
            "Let me walk you through",
            "Step 1:",
            "Step 2:",
            "As we discussed",
            "I wanted to show you",
            "you would go to",
            "and then you'd come up here",
            "click",
            "in here you can"
        ]
        
        answer_lower = answer.lower()
        for indicator in raw_indicators:
            if indicator.lower() in answer_lower:
                return True
        return False
    
    def _ask_llm_for_answer_reprocessed(self, transcript: str, question: str) -> Optional[Dict[str, str]]:
        """Reprocess with stricter prompt to avoid raw transcript content"""
        try:
            import openai
            import os
            import json
            
            # Get OpenAI API key
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                return None
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Stricter prompt for reprocessing
            prompt = f"""You are an expert at analyzing video transcripts to answer questions. Your task is to find the most relevant information in the transcript and provide a HIGH-QUALITY, INTELLIGENT answer.

CRITICAL: Your answer must be EXACTLY 4-5 sentences to provide comprehensive information. Each sentence should be detailed and informative.

ABSOLUTELY FORBIDDEN:
- NEVER include raw transcript quotes like "Hey there" or "Great question"
- NEVER include step-by-step instructions from the transcript
- NEVER include casual conversation starters
- NEVER include "click here" or "go to" instructions

MANDATORY REQUIREMENTS:
- ALWAYS provide processed, intelligent analysis
- ALWAYS use professional, business-ready language
- ALWAYS focus on the core essence and business value
- ALWAYS interpret and analyze, never quote directly

YOUR ANSWER MUST:
- Be EXACTLY 4-5 sentences to provide comprehensive information - Each sentence should be detailed and informative
- Be intelligent and insightful (like ChatGPT)
- Show deep understanding of the concepts
- Use professional, business-ready language
- Provide clear comparisons and contrasts
- Be immediately valuable and actionable
- Demonstrate consciousness and completeness
- Focus on the core essence and business value

Return ONLY valid JSON in the format:
{{
  "closest_timestamp": "<timestamp or 'not found'>",
  "answer": "<PROCESSED high-quality answer or 'not found'>",
  "video_url": "<video URL or 'not found'>",
  "video_title": "<video title or 'not found'>"
}}

transcript:
{transcript}

question: {question}"""

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,   # Increased to allow longer 4-5 sentence responses
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.1
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Clean up the response - remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse reprocessed LLM JSON response: {e}")
                return None
            
        except Exception as e:
            print(f"❌ Error in reprocessing LLM call: {e}")
            return None
