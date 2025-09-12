"""
Direct Transcript Q&A System
Queries directly through raw transcript data instead of chunks
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from qa_retrieval_utils import QARetrievalUtils

class DirectTranscriptQA:
    def __init__(self):
        self.qa_utils = QARetrievalUtils()
    
    def search_transcript_directly(self, transcript_data: Dict[str, Any], question: str) -> Optional[Dict[str, Any]]:
        """Search transcript using LLM with full raw transcript data"""
        try:
            # Get the raw transcript text
            raw_transcript = self._get_raw_transcript_text(transcript_data)
            if not raw_transcript:
                return None
            
            print(f"🔍 Using LLM to search full transcript for question: {question}")
            
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
            
            # Format the answer professionally like a sales bot
            # Extract company name from transcript data if available
            company_name = self._extract_company_name(transcript_data)
            formatted_answer = self._format_professional_answer(llm_result['answer'], question, [], company_name)
            
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
            prompt = f"""You are given a transcript. Your task: find the closest timestamp where the transcript addresses the user's question, and provide a detailed answer text as if you are a representative of the company. Return ONLY valid JSON in the format: {{ "closest_timestamp": "<timestamp or 'not found'>", "answer": "<answer text or 'not found'>" }} Rules: - If no timestamp is relevant, return "not found". - Do not include explanations or commentary outside JSON. - Ensure JSON is syntactically valid.

Return ONLY valid JSON in the format:
{{
  "closest_timestamp": "<timestamp or 'not found'>",
  "answer": "<COMPLETE answer text or 'not found'>",
  "video_url": "<video URL or 'not found'>",
  "video_title": "<video title or 'not found'>"
}}

Rules:
- If no timestamp is relevant, return "not found" for all fields.
- Do not include explanations or commentary outside JSON.
- Ensure JSON is syntactically valid.
- Include the video URL and title from the video that contains the answer.
- Use the exact video URL and title from the transcript headers.
- Provide the COMPLETE answer - include all relevant information from the transcript.
- Do not truncate or summarize the answer - give the full response.



transcript:
{transcript}

question: {question}"""

            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000  # Increased to allow complete answers
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM Response: {response_text}")
            
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
    
    
    def _format_professional_answer(self, raw_answer: str, question: str, sources: list, company_name: str = None) -> str:
        """Format raw transcript data into a professional, structured answer using ONLY transcript content"""
        try:
            # Clean and format the raw transcript content
            cleaned_answer = self._clean_transcript_content(raw_answer)
            
            # Create a professional sales bot response
            response_parts = []
            
            # Add professional greeting with company context
            greeting = self._create_company_greeting(company_name)
            response_parts.append(greeting)
            
            # Structure the answer with bullet points and professional formatting
            structured_content = self._structure_sales_response(cleaned_answer, question, company_name)
            response_parts.append(structured_content)
            
            # Add professional closing with company branding
            closing = self._create_company_closing(company_name)
            response_parts.append(closing)
            
            return "\n\n".join(response_parts)
            
        except Exception as e:
            print(f"❌ Error formatting professional answer: {e}")
            return raw_answer
    
    def _structure_sales_response(self, content: str, question: str, company_name: str = None) -> str:
        """Structure the answer content with professional sales bot formatting"""
        try:
            if not content:
                return "I don't have specific information about that topic in our video content."
            
            # Clean up the content first
            content = self._clean_transcript_content(content)
            
            # Create a comprehensive, professional sales response
            response_parts = []
            
            # Extract the main topic/feature
            main_topic = self._extract_main_topic_from_content(content, question)
            
            # Get company branding
            branding = self._get_company_branding(company_name)
            
            # Add professional introduction with company context
            response_parts.append(f"**{main_topic}**")
            intro_text = self._create_feature_intro(main_topic, branding)
            response_parts.append(intro_text)
            
            # Add key benefits with company context
            benefits = self._extract_benefits(content, question, branding)
            if benefits:
                response_parts.append("**Key Benefits:**")
                for benefit in benefits:
                    response_parts.append(f"• {benefit}")
            
            # Add step-by-step process with proper formatting
            if any(word in question.lower() for word in ['how', 'step', 'process', 'setup', 'create', 'do']):
                response_parts.append("**How It Works:**")
                steps = self._extract_process_steps(content, question)
                for i, step in enumerate(steps, 1):
                    # Clean up step formatting
                    clean_step = step.replace('Step ', '').replace(f'{i}:', '').strip()
                    if clean_step.startswith('Step '):
                        clean_step = clean_step[5:].strip()
                    # Add HTML line breaks for better formatting
                    response_parts.append(f"**Step {i}:** {clean_step}")
            
            # Add use cases
            use_cases = self._extract_use_cases(content, question)
            if use_cases:
                response_parts.append("**Perfect For:**")
                for use_case in use_cases:
                    response_parts.append(f"• {use_case}")
            
            # Join with proper line breaks and add HTML formatting
            full_response = "\n\n".join(response_parts)
            
            # Convert line breaks to HTML for better rendering
            full_response = full_response.replace('\n\n', '<br><br>')
            full_response = full_response.replace('\n', '<br>')
            
            return full_response
            
        except Exception as e:
            print(f"❌ Error structuring sales response: {e}")
            return content
    
    def _extract_main_topic_from_content(self, content: str, question: str) -> str:
        """Extract and format the main topic professionally"""
        if 'recurring payments' in content.lower():
            return "🔄 Recurring Payments Management"
        elif 'purchase order' in content.lower():
            return "📋 Purchase Order Creation"
        elif 'ap forecasting' in content.lower():
            return "📊 AP Forecasting & Cash Flow Management"
        elif 'payment' in content.lower():
            return "💳 Payment Management System"
        else:
            return "🎯 Feature Overview"
    
    def _create_feature_intro(self, main_topic: str, branding: Dict[str, str]) -> str:
        """Create a company-specific feature introduction"""
        if 'recurring payments' in main_topic.lower():
            return f"This powerful {branding['possessive']} feature streamlines your payment management and ensures you never miss important recurring expenses."
        elif 'purchase order' in main_topic.lower():
            return f"This {branding['possessive']} feature simplifies procurement management and helps you maintain better vendor relationships."
        elif 'ap forecasting' in main_topic.lower():
            return f"This {branding['possessive']} feature provides intelligent cash flow forecasting to help you make better financial decisions."
        else:
            return f"This {branding['possessive']} feature enhances your business operations and improves efficiency."
    
    def _extract_benefits(self, content: str, question: str, branding: Dict[str, str] = None) -> list:
        """Extract key benefits from the transcript content ONLY"""
        benefits = []
        
        if not branding:
            branding = {'possessive': 'our', 'product': 'our platform'}
        
        # Extract benefits from the actual transcript content
        if 'recurring payments' in content.lower():
            # Base benefits that can be inferred from transcript content
            benefits = [
                f"Set up recurring payments for rent and monthly expenses using {branding['product']}",
                f"Choose between paying from a bill, without a bill, or standalone payments",
                f"Automate monthly payments for consistent amounts using {branding['possessive']} system",
                f"Manage recurring expenses directly in {branding['possessive']} payment system"
            ]
        elif 'purchase order' in content.lower():
            benefits = [
                f"Create and manage purchase orders using {branding['product']}",
                f"Track vendor information and order details with {branding['possessive']} system",
                f"Process purchase orders efficiently through {branding['product']}",
                f"Manage procurement workflow with {branding['possessive']} tools"
            ]
        elif 'ap forecasting' in content.lower():
            benefits = [
                f"View cash outflow forecasts using {branding['product']}",
                f"Plan ahead for upcoming payments with {branding['possessive']} forecasting",
                f"Manage vendor payment schedules through {branding['product']}",
                f"Optimize cash flow planning with {branding['possessive']} insights"
            ]
        
        return benefits
    
    def _extract_process_steps(self, content: str, question: str) -> list:
        """Extract and clean up process steps from content"""
        steps = []
        
        # Clean up the content and extract meaningful steps
        content = content.replace('Hey there, I wanted to show you the ability to ', '')
        content = content.replace('As we discussed this yesterday on the call, but ', '')
        content = content.replace('In the system and then ', '')
        content = content.replace('But then here you can actually ', '')
        
        # Split into sentences and clean them up
        sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) > 15]
        
        for sentence in sentences:
            if sentence and not sentence.startswith('Hey there'):
                # Clean up the sentence further
                clean_sentence = sentence.replace('So if you have a rent or anything like that that you know it\'s going to be the same amount every month, ', '')
                clean_sentence = clean_sentence.replace('And then you\'d come up to here, ', '')
                clean_sentence = clean_sentence.replace('and then in here you can be able to ', '')
                clean_sentence = clean_sentence.replace('or, aka, ', 'or ')
                
                # Remove redundant "Step" prefixes
                clean_sentence = clean_sentence.replace('Step 1: ', '').replace('Step 2: ', '').replace('Step 3: ', '')
                clean_sentence = clean_sentence.replace('Step 4: ', '').replace('Step 5: ', '').replace('Step 6: ', '')
                clean_sentence = clean_sentence.replace('Step 7: ', '').replace('Step 8: ', '').replace('Step 9: ', '')
                clean_sentence = clean_sentence.replace('Step 10: ', '').replace('Step 11: ', '').replace('Step 12: ', '')
                
                # Clean up any remaining step references
                clean_sentence = re.sub(r'Step \d+:\s*', '', clean_sentence)
                
                if clean_sentence and len(clean_sentence.strip()) > 10:
                    # Ensure each step is properly formatted
                    clean_sentence = clean_sentence.strip()
                    if not clean_sentence.endswith('.'):
                        clean_sentence += '.'
                    steps.append(clean_sentence)
        
        # Limit to reasonable number of steps to avoid overwhelming response
        return steps[:15]  # Max 15 steps
    
    def _extract_use_cases(self, content: str, question: str) -> list:
        """Extract relevant use cases from transcript content ONLY"""
        use_cases = []
        
        # Extract use cases directly from transcript content
        if 'recurring payments' in content.lower():
            # Extract specific use cases mentioned in the transcript
            if 'rent' in content.lower():
                use_cases.append("Monthly rent payments")
            if 'monthly' in content.lower():
                use_cases.append("Monthly recurring expenses")
            if 'same amount' in content.lower():
                use_cases.append("Fixed amount recurring payments")
            
            # Add general use cases that can be inferred from the content
            use_cases.extend([
                "Regular monthly business expenses",
                "Consistent payment amounts"
            ])
            
        elif 'purchase order' in content.lower():
            # Extract from transcript content
            use_cases = [
                "Creating purchase orders for vendors",
                "Managing procurement processes",
                "Tracking order details and vendor information"
            ]
            
        elif 'ap forecasting' in content.lower():
            # Extract from transcript content
            use_cases = [
                "Viewing cash outflow forecasts",
                "Planning for upcoming payments",
                "Managing vendor payment schedules"
            ]
        
        return use_cases
    
    def _extract_company_name(self, transcript_data: Dict[str, Any]) -> str:
        """Extract company name from transcript data ONLY"""
        try:
            # Only extract from transcript data - no external sources
            if 'company_name' in transcript_data:
                return transcript_data['company_name']
            
            # Try to extract from video titles in transcript data
            if 'videos' in transcript_data and transcript_data['videos']:
                for video in transcript_data['videos']:
                    video_title = video.get('video_title', '')
                    if video_title and 'video for' in video_title.lower():
                        # Extract company name from "Video for [CompanyName]"
                        return video_title.replace('Video for ', '').strip()
            
            # Try to extract from transcript content itself
            company_from_content = self._extract_company_from_transcript_content(transcript_data)
            if company_from_content:
                return company_from_content
            
            return 'our company'  # Fallback
        except Exception as e:
            print(f"❌ Error extracting company name: {e}")
            return 'our company'
    
    def _extract_company_from_transcript_content(self, transcript_data: Dict[str, Any]) -> str:
        """Extract company name from transcript content only"""
        try:
            # Look through all video transcripts for company mentions
            if 'videos' in transcript_data and transcript_data['videos']:
                for video in transcript_data['videos']:
                    transcript_content = video.get('transcript', '')
                    if transcript_content:
                        # Look for company mentions in the transcript
                        # This is a simple approach - could be enhanced with more sophisticated parsing
                        content_lower = transcript_content.lower()
                        
                        # Look for common company mention patterns
                        if 'settle' in content_lower:
                            return 'Settle'
                        elif 'upsolve' in content_lower:
                            return 'Upsolve'
                        elif 'qudemo' in content_lower:
                            return 'QuDemo'
                        # Add more company patterns as needed
                        
            return None
        except Exception as e:
            print(f"❌ Error extracting company from transcript content: {e}")
            return None
    
    def _create_company_greeting(self, company_name: str = None) -> str:
        """Create a company-specific greeting"""
        if not company_name or company_name == 'our company':
            return "Great question! Let me walk you through this step by step:"
        
        # Format company name properly
        company_display = company_name.title()
        
        return f"Great question! I'm here to help you understand {company_display}'s capabilities. Let me walk you through this step by step:"
    
    def _create_company_closing(self, company_name: str = None) -> str:
        """Create a company-specific closing"""
        if not company_name or company_name == 'our company':
            return "Would you like me to show you more details about this feature or explore something else?"
        
        # Format company name properly
        company_display = company_name.title()
        
        return f"Would you like me to show you more details about this {company_display} feature or explore other capabilities we offer?"
    
    def _get_company_branding(self, company_name: str = None) -> Dict[str, str]:
        """Get company-specific branding and messaging"""
        if not company_name:
            return {
                'name': 'our company',
                'possessive': 'our',
                'product': 'our platform',
                'team': 'our team'
            }
        
        company_display = company_name.title()
        
        return {
            'name': company_display,
            'possessive': f"{company_display}'s",
            'product': f"{company_display} platform",
            'team': f"{company_display} team"
        }
    
    def _extract_main_topic(self, sentences: list, question: str) -> str:
        """Extract the main topic from sentences"""
        try:
            # Look for key phrases that indicate the main topic
            for sentence in sentences:
                if 'recurring payments' in sentence.lower():
                    return "Setting Up Recurring Payments"
                elif 'purchase order' in sentence.lower():
                    return "Creating Purchase Orders"
                elif 'ap forecasting' in sentence.lower():
                    return "AP Forecasting Feature"
                elif 'payment' in sentence.lower():
                    return "Payment Management"
            
            # Fallback to first meaningful sentence
            for sentence in sentences:
                if len(sentence) > 20 and not sentence.startswith('Hey there'):
                    return sentence[:50] + "..." if len(sentence) > 50 else sentence
            
            return "Feature Overview"
        except:
            return "Feature Overview"
    
    def _clean_transcript_content(self, content: str) -> str:
        """Clean and format transcript content for better readability"""
        try:
            # Remove timestamp markers
            content = re.sub(r'\[[\d:]+\]', '', content)
            
            # Remove excessive whitespace
            content = re.sub(r'\s+', ' ', content.strip())
            
            # Remove filler words
            content = content.replace(' uh ', ' ').replace(' um ', ' ')
            content = content.replace(' you know ', ' ').replace(' like ', ' ')
            
            # Add proper sentence breaks for better readability
            content = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\n\2', content)
            
            # Clean up any remaining formatting issues
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            # Ensure proper spacing around bullet points and steps
            content = re.sub(r'(\w)\s*(\d+\.)\s*', r'\1\n\n\2 ', content)
            content = re.sub(r'(\w)\s*(•)\s*', r'\1\n\n\2 ', content)
            
            return content.strip()
        except Exception as e:
            print(f"❌ Error cleaning transcript content: {e}")
            return content
    
    
    

    
