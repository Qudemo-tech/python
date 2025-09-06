#!/usr/bin/env python3
"""
Simple Enhanced QA System - WORKING VERSION
Based on the user's previous working code, updated for Pinecone Standard Plan
"""

import os
import re
from typing import List, Dict
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleEnhancedQA:
    def __init__(self):
        """Initialize simple enhanced QA system"""
        # Pinecone Standard Plan - Multiple indexes
        self.indexes = {
            'video': 'qudemo-video-index',
            'knowledge': 'qudemo-knowledge-index',
            'web': 'qudemo-web-index',
            'legacy': 'qudemo-index'  # Fallback for existing content
        }
        
    def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Ask a question and get an intelligent answer from video and knowledge sources"""
        try:
            print(f"❓ Question for {company_name} qudemo {qudemo_id}: {question}")
            
            # Get question embedding
            question_embedding = self._get_embedding(question)
            
            # Search video transcripts
            video_result = self._search_video_transcripts(question, company_name, qudemo_id, question_embedding)
            
            # Search knowledge sources
            knowledge_result = self._search_knowledge_sources(question, company_name, qudemo_id, question_embedding)
            
            # Select the best answer intelligently
            final_answer = self._select_best_answer(video_result, knowledge_result, question)
            
            return final_answer
            
        except Exception as e:
            print(f"❌ Error in ask_question: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': "I encountered an error while processing your question. Please try again.",
                'start': 0,
                'end': 0,
                'video_url': None,
                'sources': [],
                'total_sources': 0,
                'search_score': 0,
                'content_types_found': [],
                'difficulty_level': 'beginner',
                'estimated_time': '1 minute'
            }

    def _search_video_transcripts(self, question: str, company_name: str, qudemo_id: str, question_embedding: List[float]) -> Dict:
        """Search video transcripts for relevant content - UPDATED FOR STANDARD PLAN"""
        try:
            print(f"🎬 Searching video transcripts for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            
            # Try video index first, then legacy index as fallback
            video_index_name = self.indexes['video']
            legacy_index_name = self.indexes['legacy']
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            print(f"🔍 Searching in namespace: {namespace}")
            
            # Search in video index first
            try:
                index = pc.Index(video_index_name)
                query_results = index.query(
                    vector=question_embedding,
                    top_k=20,
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=0.1
                )
                
                if query_results.matches:
                    print(f"✅ Found {len(query_results.matches)} matches in video index")
                    if len(query_results.matches) > 0:
                        print(f"🔍 First match score: {query_results.matches[0].score}")
                        print(f"🔍 First match metadata keys: {list(query_results.matches[0].metadata.keys())}")
                    return self._process_video_matches(query_results, question, company_name, qudemo_id)
                
            except Exception as video_error:
                print(f"⚠️ Video index search failed: {video_error}")
            
            # Fallback to legacy index
            try:
                print("🔄 Trying legacy index as fallback...")
                index = pc.Index(legacy_index_name)
                query_results = index.query(
                    vector=question_embedding,
                    top_k=20,
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=0.1
                )
                
                if query_results.matches:
                    print(f"✅ Found {len(query_results.matches)} matches in legacy index")
                    return self._process_video_matches(query_results, question, company_name, qudemo_id)
                
            except Exception as legacy_error:
                print(f"⚠️ Legacy index search failed: {legacy_error}")
            
            print("❌ No video content found in any index")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'video',
                'video_url': None,
                'start_time': 0,
                'end_time': 0
            }
                
        except Exception as e:
            print(f"❌ Error searching video transcripts: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'video',
                'video_url': None,
                'start_time': 0,
                'end_time': 0
            }

    def _process_video_matches(self, query_results, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Process video matches and extract best answer"""
        try:
            # Filter for video content and find best match
            video_matches = []
            for match in query_results.matches:
                metadata = match.metadata
                source_type = metadata.get('source_type', '')
                source = metadata.get('source', '')
                title = metadata.get('title', '')
                
                # Check if this is video content
                is_video = (
                    source_type == 'video_transcript' or 
                    source == 'video' or 
                    'video' in title.lower() or 
                    'transcription' in title.lower()
                )
                
                if is_video:
                    video_matches.append(match)
                    print(f"✅ Accepted video match: {metadata.get('title', 'NO_TITLE')[:50]}...")
            
            if not video_matches:
                print("⚠️ No video content found in this qudemo")
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'video',
                    'video_url': None,
                    'start_time': 0,
                    'end_time': 0
                }
            
            # Get best match
            best_match = video_matches[0]
            metadata = best_match.metadata
            raw_text = metadata.get('text', '')
            video_url = metadata.get('video_url', '') or metadata.get('url', '')
            
            # Debug: Print all metadata fields
            print(f"🔍 DEBUG: All metadata fields: {list(metadata.keys())}")
            print(f"🔍 DEBUG: Video URL: '{video_url}'")
            print(f"🔍 DEBUG: Start timestamp: {metadata.get('start', 'NOT_FOUND')}")
            print(f"🔍 DEBUG: End timestamp: {metadata.get('end', 'NOT_FOUND')}")
            print(f"🔍 DEBUG: Raw text preview: {raw_text[:200]}...")
            
            # Extract timestamp from metadata first, then fallback to text content
            # Note: Pinecone stores timestamps as 'start' and 'end', not 'start_timestamp' and 'end_timestamp'
            start_time = metadata.get('start', 0)
            end_time = metadata.get('end', 0)
            
            # If no precise timestamp in metadata, try to extract from text
            if start_time == 0:
                timestamp_match = re.search(r'\[(\d{1,2}):(\d{2})\]', raw_text)
                if timestamp_match:
                    minutes = int(timestamp_match.group(1))
                    seconds = int(timestamp_match.group(2))
                    start_time = minutes * 60 + seconds
                    end_time = start_time + 30  # 30 second window
                else:
                    start_time = 0
                    end_time = 30
            
            # Log the extracted timestamp
            if start_time > 0:
                print(f"✅ Using metadata timestamp: {start_time}s - {end_time}s")
            else:
                print(f"⚠️ No timestamp found in metadata, using fallback logic")
            
            # Format the timestamp for display
            if start_time > 0 and start_time <= 3600:
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                formatted_timestamp = f"{minutes:02d}:{seconds:02d}"
            else:
                formatted_timestamp = "00:00"
                print(f"⚠️ Invalid timestamp: {start_time}s, using 00:00")
            
            # Clean text by removing timestamps
            clean_text = re.sub(r'\[\d{1,2}:\d{2}\]', '', raw_text).strip()
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(question, clean_text)
            
            print(f"✅ Best video match - Score: {best_match.score:.3f}, Relevance: {relevance_score:.3f}")
            print(f"📹 Video URL: {video_url}")
            print(f"⏰ Timestamp: {formatted_timestamp} ({start_time}s - {end_time}s)")
            
            # Check if video is relevant enough to include
            MIN_VIDEO_RELEVANCE = 0.5
            if relevance_score < MIN_VIDEO_RELEVANCE:
                print(f"❌ Video relevance too low ({relevance_score:.3f} < {MIN_VIDEO_RELEVANCE}) - excluding video")
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'video',
                    'video_url': None,
                    'start_time': 0,
                    'end_time': 0
                }
            
            # Add source indicator for test purposes
            clean_text_with_source = clean_text + "\n\n[SOURCE: Video Data]"
            
            return {
                'success': True,
                'answer': clean_text_with_source,
                'score': best_match.score,
                'relevance_score': relevance_score,
                'source': 'video',
                'video_url': video_url,
                'start': start_time,
                'end': end_time,
                'start_time': start_time,
                'end_time': end_time,
                'formatted_timestamp': formatted_timestamp,
                'raw_text': raw_text
            }
            
        except Exception as e:
            print(f"❌ Error processing video matches: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'video',
                'video_url': None,
                'start_time': 0,
                'end_time': 0
            }

    def _search_knowledge_sources(self, question: str, company_name: str, qudemo_id: str, question_embedding: List[float]) -> Dict:
        """Search knowledge sources for relevant content - UPDATED FOR STANDARD PLAN"""
        try:
            print(f"📚 Searching knowledge sources for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            
            # Try knowledge index first, then legacy index as fallback
            knowledge_index_name = self.indexes['knowledge']
            legacy_index_name = self.indexes['legacy']
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            print(f"🔍 Searching in namespace: {namespace}")
            
            # Search in knowledge index first
            try:
                index = pc.Index(knowledge_index_name)
                query_results = index.query(
                    vector=question_embedding,
                    top_k=20,
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=0.1
                )
                
                if query_results.matches:
                    print(f"✅ Found {len(query_results.matches)} matches in knowledge index")
                    return self._process_knowledge_matches(query_results, question)
                
            except Exception as knowledge_error:
                print(f"⚠️ Knowledge index search failed: {knowledge_error}")
            
            # Fallback to legacy index
            try:
                print("🔄 Trying legacy index as fallback...")
                index = pc.Index(legacy_index_name)
                query_results = index.query(
                    vector=question_embedding,
                    top_k=20,
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=0.1
                )
                
                if query_results.matches:
                    print(f"✅ Found {len(query_results.matches)} matches in legacy index")
                    return self._process_knowledge_matches(query_results, question)
                
            except Exception as legacy_error:
                print(f"⚠️ Legacy index search failed: {legacy_error}")
            
            print("❌ No knowledge content found in any index")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'knowledge'
            }
                
        except Exception as e:
            print(f"❌ Error searching knowledge sources: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'knowledge'
            }

    def _process_knowledge_matches(self, query_results, question: str) -> Dict:
        """Process knowledge matches and extract best answer"""
        try:
            # More inclusive filtering - accept any content that's not explicitly video
            knowledge_matches = []
            for match in query_results.matches:
                metadata = match.metadata
                source_type = metadata.get('source_type', '')
                source = metadata.get('source', '')
                
                # Accept web_scraping, knowledge, or any non-video content
                is_knowledge = (
                    source_type == 'web_scraping' or 
                    source_type == 'knowledge' or 
                    source == 'web_scraping' or
                    source == 'knowledge' or
                    (source_type != 'video_transcript' and 'video' not in source_type.lower())
                )
                
                if is_knowledge:
                    knowledge_matches.append(match)
                    print(f"✅ Accepted knowledge match: {metadata.get('title', 'NO_TITLE')[:50]}...")
            
            if not knowledge_matches:
                print("⚠️ No knowledge-specific matches found after filtering")
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'knowledge'
                }
            
            best_match = knowledge_matches[0]
            metadata = best_match.metadata
            content = metadata.get('text', '')
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(question, content)
            
            print(f"✅ Best knowledge match - Score: {best_match.score:.3f}, Relevance: {relevance_score:.3f}")
            
            # Clean the content by removing markdown formatting and unwanted symbols
            clean_content = content
            
            # Remove markdown formatting
            clean_content = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_content)  # Remove bold
            clean_content = re.sub(r'\*(.*?)\*', r'\1', clean_content)      # Remove italic
            clean_content = re.sub(r'`(.*?)`', r'\1', clean_content)        # Remove code
            clean_content = re.sub(r'#{1,6}\s*(.*)', r'\1', clean_content)  # Remove headers
            clean_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_content)  # Remove links
            
            # Remove table formatting
            clean_content = re.sub(r'\|.*?\|', '', clean_content)  # Remove table rows
            clean_content = re.sub(r'\|-+\|', '', clean_content)   # Remove table separators
            
            # Remove bullet points and list markers
            clean_content = re.sub(r'^\s*[\*\-+]\s+', '', clean_content, flags=re.MULTILINE)
            clean_content = re.sub(r'^\s*\d+\.\s+', '', clean_content, flags=re.MULTILINE)
            
            # Remove extra whitespace and clean up
            clean_content = re.sub(r'\n\s*\n', '\n\n', clean_content)  # Remove extra blank lines
            clean_content = re.sub(r'^\s+', '', clean_content, flags=re.MULTILINE)  # Remove leading spaces
            clean_content = clean_content.strip()
            
            # Use GPT to format the raw content into structured, user-friendly answer
            formatted_answer = self._format_knowledge_answer(question, clean_content)
            
            return {
                'success': True,
                'answer': formatted_answer,
                'score': best_match.score,
                'relevance_score': relevance_score,
                'source': 'knowledge',
                'url': metadata.get('url', ''),
                'title': metadata.get('title', '')
            }
            
        except Exception as e:
            print(f"❌ Error processing knowledge matches: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'knowledge'
            }

    def _format_knowledge_answer(self, question: str, raw_content: str) -> str:
        """Format raw scraped content into structured, user-friendly answer using GPT"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            prompt = f"""
You are a helpful product knowledge assistant. The user asked: "{question}"

Here is the raw scraped content from a help center or documentation:
{raw_content}

Please format this content into a clear, structured, step-by-step answer that directly addresses the user's question.
Follow these guidelines:
1. Remove any navigation elements, headers, footers, or irrelevant UI text
2. Organize the information into clear numbered steps or bullet points
3. Focus on actionable instructions and practical guidance
4. Use a friendly, helpful tone
5. Remove any formatting symbols, HTML tags, or technical jargon
6. Make it easy to follow and understand
7. If there are tables or lists, convert them to readable text format
8. Remove any "Did this answer your question?" or similar feedback elements

Format the answer as:
1. [Step/Point 1]
2. [Step/Point 2]
3. [Step/Point 3]
etc.

Answer:
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful product knowledge assistant who formats raw content into clear, structured answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            formatted_answer = response.choices[0].message.content.strip()
            
            # Add source indicator for test purposes
            formatted_answer += "\n\n[SOURCE: Scraped Data]"
            
            # Fallback to basic cleaning if GPT fails
            if not formatted_answer or len(formatted_answer) < 50:
                print("⚠️ GPT formatting failed, using basic cleaning")
                basic_answer = self._basic_content_cleaning(raw_content)
                basic_answer += "\n\n[SOURCE: Scraped Data]"
                return basic_answer
            
            return formatted_answer
            
        except Exception as e:
            print(f"❌ Error formatting knowledge answer with GPT: {e}")
            # Fallback to basic cleaning
            return self._basic_content_cleaning(raw_content)

    def _basic_content_cleaning(self, content: str) -> str:
        """Basic content cleaning as fallback when GPT formatting fails"""
        import re
        
        # Remove common navigation and UI elements
        content = re.sub(r'Skip to main content.*?Search for articles', '', content, flags=re.DOTALL)
        content = re.sub(r'Did this answer your question.*', '', content, flags=re.DOTALL)
        content = re.sub(r'😞.*', '', content, flags=re.DOTALL)
        content = re.sub(r'Help Center.*', '', content, flags=re.DOTALL)
        
        # Remove extra whitespace and clean up
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)
        content = content.strip()
        
        return content

    def _calculate_relevance_score(self, question: str, content: str) -> float:
        """Calculate relevance score combining semantic and keyword matching"""
        try:
            # Semantic score (already provided by Pinecone)
            semantic_score = 0.6  # Base semantic weight
            
            # Keyword relevance
            question_lower = question.lower()
            content_lower = content.lower()
            
            # Key terms from question
            key_terms = question_lower.split()
            keyword_matches = sum(1 for term in key_terms if term in content_lower)
            keyword_score = min(keyword_matches / len(key_terms), 1.0) * 0.2
            
            # Required terms (must be present for high relevance)
            required_terms = []
            if 'workflow' in question_lower:
                required_terms.extend(['workflow', 'process', 'flow'])
            if 'graph' in question_lower:
                required_terms.extend(['graph', 'chart', 'visualization'])
            if 'ap' in question_lower:
                required_terms.extend(['ap', 'accounts payable', 'purchasing'])
            if 'purchasing' in question_lower:
                required_terms.extend(['purchasing', 'order', 'procurement'])
            
            required_score = 0
            if required_terms:
                required_matches = sum(1 for term in required_terms if term in content_lower)
                required_score = (required_matches / len(required_terms)) * 0.2
            
            # Negative terms (penalty for irrelevant content)
            negative_terms = ['forecast', 'forecasting', 'prediction', 'future']
            negative_penalty = 0
            for term in negative_terms:
                if term in content_lower:
                    negative_penalty += 0.15
            
            # Combined score
            combined_score = semantic_score + keyword_score + required_score - negative_penalty
            combined_score = max(0.0, min(1.0, combined_score))
            
            return combined_score
            
        except Exception as e:
            print(f"❌ Error calculating relevance score: {e}")
            return 0.0

    def _select_best_answer(self, video_result: Dict, knowledge_result: Dict, question: str) -> Dict:
        """Intelligently select the best answer from video and knowledge sources"""
        try:
            print(f"🤔 Selecting best answer from video and knowledge sources")
            
            # Define relevance thresholds
            HIGH_RELEVANCE = 0.7
            MEDIUM_RELEVANCE = 0.6
            LOW_RELEVANCE = 0.4
            
            video_score = video_result.get('relevance_score', 0) if video_result.get('success') else 0
            knowledge_score = knowledge_result.get('relevance_score', 0) if knowledge_result.get('success') else 0
            
            print(f"📊 Video score: {video_score:.3f}, Knowledge score: {knowledge_score:.3f}")
            print(f"🔍 DEBUG: video_result keys: {list(video_result.keys()) if video_result else 'None'}")
            print(f"🔍 DEBUG: video_result start: {video_result.get('start', 'NOT_FOUND') if video_result else 'None'}")
            print(f"🔍 DEBUG: video_result video_url: {video_result.get('video_url', 'NOT_FOUND') if video_result else 'None'}")
            
            # Decision matrix - prioritize knowledge when it's highly relevant
            # TEMPORARY FIX: Always use video answer when video content is found and relevant
            if video_result.get('success') and video_score >= 0.5:
                print("🎬 TEMPORARY FIX: Using video answer for timestamp jumping")
                return self._generate_guided_answer(video_result, question)
            elif knowledge_score >= HIGH_RELEVANCE and video_score < MEDIUM_RELEVANCE:
                # Knowledge highly relevant, video not relevant enough - use knowledge only
                print("📚 Knowledge highly relevant - using knowledge answer only")
                return {
                    'success': True,
                    'answer': knowledge_result['answer'],  # Already formatted by _format_knowledge_answer
                    'start': 0,
                    'end': 0,
                    'video_url': None,
                    'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}],
                    'total_sources': 1,
                    'search_score': knowledge_result.get('score', 0),
                    'content_types_found': ['knowledge'],
                    'difficulty_level': 'intermediate',
                    'estimated_time': '2-3 minutes'
                }
                
            elif video_score >= HIGH_RELEVANCE and knowledge_score >= HIGH_RELEVANCE:
                # Both highly relevant - combine them
                print("🔄 Both sources highly relevant - combining answers")
                return self._generate_combined_answer(video_result, knowledge_result, question)
                
            elif video_score >= HIGH_RELEVANCE and knowledge_score < MEDIUM_RELEVANCE:
                # Video highly relevant, knowledge less so - use video only
                print("🎬 Video highly relevant - using video answer")
                return self._generate_guided_answer(video_result, question)
                
            elif video_score >= MEDIUM_RELEVANCE and knowledge_score >= MEDIUM_RELEVANCE:
                # Both moderately relevant - check if knowledge is significantly better
                score_difference = knowledge_score - video_score
                if score_difference >= 0.1:  # Knowledge is significantly better (0.1 difference)
                    print(f"📚 Knowledge significantly better (diff: {score_difference:.3f}) - using knowledge answer only")
                    return {
                        'success': True,
                        'answer': knowledge_result['answer'],  # Already formatted by _format_knowledge_answer
                        'start': 0,
                        'end': 0,
                        'video_url': None,
                        'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}]
                    }
                elif video_score < 0.65:  # Video not relevant enough even if above medium threshold
                    print(f"🎬 Video not relevant enough (score: {video_score:.3f}) - using knowledge answer only")
                    return {
                        'success': True,
                        'answer': knowledge_result['answer'],  # Already formatted by _format_knowledge_answer
                        'start': 0,
                        'end': 0,
                        'video_url': None,
                        'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}],
                        'total_sources': 1,
                        'search_score': knowledge_result.get('score', 0),
                        'content_types_found': ['knowledge'],
                        'difficulty_level': 'intermediate',
                        'estimated_time': '2-3 minutes'
                    }
                else:
                    # Both are similarly relevant - combine them
                    print("🔄 Both sources moderately relevant - combining answers")
                    return self._generate_combined_answer(video_result, knowledge_result, question)
                
            elif video_score >= LOW_RELEVANCE or knowledge_score >= LOW_RELEVANCE:
                # At least one source has some relevance - use the better one
                if video_score > knowledge_score:
                    print("🎬 Video more relevant - using video answer")
                    return self._generate_guided_answer(video_result, question)
                elif video_score >= 0.5:  # If video is reasonably relevant, prefer it for timestamp jumping
                    print("🎬 Video reasonably relevant - using video answer for timestamp jumping")
                    return self._generate_guided_answer(video_result, question)
                else:
                    print("📚 Knowledge more relevant - using knowledge answer")
                    return {
                        'success': True,
                        'answer': knowledge_result['answer'],  # Already formatted by _format_knowledge_answer
                        'start': 0,
                        'end': 0,
                        'video_url': None,
                        'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}],
                        'total_sources': 1,
                        'search_score': knowledge_result.get('score', 0),
                        'content_types_found': ['knowledge'],
                        'difficulty_level': 'intermediate',
                        'estimated_time': '2-3 minutes'
                    }
            else:
                # Neither source is relevant enough
                print("❌ Neither source is relevant enough")
                return {
                    'success': False,
                    'answer': f"I couldn't find specific information about '{question}' in the available content for this qudemo. You might want to try rephrasing your question or ask about a different aspect of the content.",
                    'start': 0,
                    'end': 0,
                    'video_url': None,
                    'sources': [],
                    'total_sources': 0,
                    'search_score': 0,
                    'content_types_found': [],
                    'difficulty_level': 'beginner',
                    'estimated_time': '1 minute'
                }
                
        except Exception as e:
            print(f"❌ Error selecting best answer: {e}")
            return {
                'success': False,
                'answer': "I encountered an error while processing your question. Please try again.",
                'start': 0,
                'end': 0,
                'video_url': None,
                'sources': [],
                'total_sources': 0,
                'search_score': 0,
                'content_types_found': [],
                'difficulty_level': 'beginner',
                'estimated_time': '1 minute'
            }

    def _generate_guided_answer(self, video_result: Dict, question: str) -> Dict:
        """Generate a guided answer from video content using GPT"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            prompt = f"""
You are a helpful sales manager assistant. The user asked: "{question}"

Here is the relevant video transcript content:
{video_result['answer']}

Please provide a clear, step-by-step answer that explains how to accomplish what the user is asking for. 
Write it in a friendly, helpful tone as if you're guiding them through the process.
Focus on practical steps and actionable advice.
Do not include timestamps, technical jargon, or any formatting symbols like *, |, #, -, etc.
Write in plain text format only without any markdown or special formatting.

Answer:
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful sales manager assistant who provides clear, step-by-step guidance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            guided_answer = response.choices[0].message.content.strip()
            
            # Add source indicator for test purposes
            guided_answer += "\n\n[SOURCE: Video Data]"
            
            return {
                'success': True,
                'answer': guided_answer,
                'start': video_result.get('start', 0),
                'end': video_result.get('end', 0),
                'video_url': video_result.get('video_url'),
                'formatted_timestamp': video_result.get('formatted_timestamp', '00:00'),
                'sources': [{'type': 'video', 'url': video_result.get('video_url', ''), 'title': 'Video Transcript'}],
                'total_sources': 1,
                'search_score': video_result.get('score', 0),
                'content_types_found': ['video'],
                'difficulty_level': 'intermediate',
                'estimated_time': '3-5 minutes'
            }
            
        except Exception as e:
            print(f"❌ Error generating guided answer: {e}")
            # Fallback to raw video answer
            fallback_answer = video_result['answer'] + "\n\n[SOURCE: Video Data]"
            return {
                'success': True,
                'answer': fallback_answer,
                'start': video_result.get('start', 0),
                'end': video_result.get('end', 0),
                'video_url': video_result.get('video_url'),
                'formatted_timestamp': video_result.get('formatted_timestamp', '00:00'),
                'sources': [{'type': 'video', 'url': video_result.get('video_url', ''), 'title': 'Video Transcript'}],
                'total_sources': 1,
                'search_score': video_result.get('score', 0),
                'content_types_found': ['video'],
                'difficulty_level': 'intermediate',
                'estimated_time': '3-5 minutes'
            }

    def _generate_combined_answer(self, video_result: Dict, knowledge_result: Dict, question: str) -> Dict:
        """Generate a combined answer from both video and knowledge sources using GPT"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            prompt = f"""
You are a helpful sales manager assistant. The user asked: "{question}"

Here is the relevant video transcript content:
{video_result['answer']}

Here is the relevant knowledge base content:
{knowledge_result['answer']}

Please provide a comprehensive answer that combines the best information from both sources.
Write it in a clear, step-by-step format that helps the user accomplish what they're asking for.
Organize the information logically and avoid repetition.
Do not include timestamps, technical jargon, or any formatting symbols like *, |, #, -, etc.
Write in plain text format only without any markdown or special formatting.

Combined Answer:
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful sales manager assistant who provides comprehensive guidance combining multiple sources."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            
            combined_answer = response.choices[0].message.content.strip()
            
            # Add source indicator for test purposes
            combined_answer += "\n\n[SOURCE: Video + Scraped Data]"
            
            return {
                'success': True,
                'answer': combined_answer,
                'start': video_result.get('start', 0) if video_result else 0,
                'end': video_result.get('end', 0) if video_result else 0,
                'video_url': video_result.get('video_url') if video_result else None,
                'formatted_timestamp': video_result.get('formatted_timestamp', '00:00') if video_result else '00:00',
                'sources': [
                    {'type': 'video', 'url': video_result.get('video_url', '') if video_result else '', 'title': 'Video Transcript'},
                    {'type': 'knowledge', 'url': knowledge_result.get('url', '') if knowledge_result else '', 'title': knowledge_result.get('title', 'Knowledge Base') if knowledge_result else 'Knowledge Base'}
                ],
                'total_sources': 2,
                'search_score': max(video_result.get('score', 0), knowledge_result.get('score', 0)),
                'content_types_found': ['video', 'knowledge'],
                'difficulty_level': 'intermediate',
                'estimated_time': '4-6 minutes'
            }
            
        except Exception as e:
            print(f"❌ Error generating combined answer: {e}")
            # Fallback to video answer only
            return self._generate_guided_answer(video_result, question)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            # Return a dummy embedding if OpenAI fails
            return [0.0] * 3072  # text-embedding-3-large uses 3072 dimensions

# Global instance
simple_enhanced_qa = None

def initialize_simple_enhanced_qa():
    """Initialize the simple enhanced QA system"""
    global simple_enhanced_qa
    
    try:
        simple_enhanced_qa = SimpleEnhancedQA()
        print("✅ Simple Enhanced QA system initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize simple enhanced QA system: {e}")
        return False

def get_simple_enhanced_qa():
    """Get the simple enhanced QA system instance"""
    if simple_enhanced_qa is None:
        raise RuntimeError("Simple Enhanced QA System not initialized. Call initialize_simple_enhanced_qa() first.")
    return simple_enhanced_qa
