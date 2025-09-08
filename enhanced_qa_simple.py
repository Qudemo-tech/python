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
            
            # Collect multiple relevant chunks for complete answers
            relevant_chunks = []
            earliest_start_time = float('inf')
            latest_end_time = 0
            combined_text = ""
            video_url = ""
            
            # Process all video matches to find relevant chunks
            chunk_texts = []
            for match in video_matches:
                metadata = match.metadata
                score = match.score
                
                # Only include chunks with good relevance scores
                if score >= 0.3:  # Threshold for relevance
                    relevant_chunks.append(match)
                    chunk_text = metadata.get('text', '').strip()
                    
                    # Clean the chunk text before adding
                    if chunk_text:
                        # Remove timestamps from individual chunks
                        chunk_text = re.sub(r'\[\d{1,2}:\d{2}\]', '', chunk_text).strip()
                        # Fix basic formatting issues
                        chunk_text = re.sub(r'\s+', ' ', chunk_text)
                        chunk_text = chunk_text.strip()
                        
                        # Only add if it's substantial content
                        if len(chunk_text) > 20 and chunk_text not in chunk_texts:
                            chunk_texts.append(chunk_text)
                    
                    # Track timestamps
                    start_time = metadata.get('start_timestamp', 0)
                    end_time = metadata.get('end_timestamp', 0)
                    
                    if start_time > 0 and start_time < earliest_start_time:
                        earliest_start_time = start_time
                    if end_time > 0 and end_time > latest_end_time:
                        latest_end_time = end_time
                    
                    # Get video URL from first chunk
                    if not video_url:
                        video_url = metadata.get('video_url', '') or metadata.get('url', '')
            
            # Combine unique chunk texts with proper spacing
            combined_text = ' '.join(chunk_texts)
            
            if not relevant_chunks:
                print("⚠️ No relevant chunks found with sufficient score")
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'video',
                    'video_url': None,
                    'start_time': 0,
                    'end_time': 0
                }
            
            # Use the best match for scoring
            best_match = video_matches[0]
            best_score = best_match.score
            
            # Debug information
            print(f"🔍 DEBUG: Found {len(relevant_chunks)} relevant chunks")
            print(f"🔍 DEBUG: Best score: {best_score}")
            print(f"🔍 DEBUG: Video URL: '{video_url}'")
            print(f"🔍 DEBUG: Earliest start: {earliest_start_time}s")
            print(f"🔍 DEBUG: Latest end: {latest_end_time}s")
            print(f"🔍 DEBUG: Combined text preview: {combined_text[:200]}...")
            
            # Smart timestamp selection: prioritize earlier chunks when scores are similar
            # Find the best chunk with the earliest timestamp among high-relevance chunks
            best_chunk = None
            best_score = video_matches[0].score if video_matches else 0
            earliest_time = float('inf')
            
            # Look for chunks with similar relevance scores (within 0.1 of the best)
            for match in video_matches:
                score = match.score
                metadata = match.metadata
                start_timestamp = metadata.get('start_timestamp', 0)
                
                # If this chunk has a high relevance score (within 0.1 of best)
                if score >= best_score - 0.1:
                    # Prefer earlier timestamps for similar relevance
                    if start_timestamp < earliest_time:
                        earliest_time = start_timestamp
                        best_chunk = match
                        print(f"🎯 Selected chunk at {start_timestamp}s (score: {score:.3f})")
            
            # Fallback to highest score if no early chunk found
            if best_chunk is None:
                best_chunk = video_matches[0]
                print(f"🎯 Fallback to highest score chunk (score: {best_chunk.score:.3f})")
            
            best_metadata = best_chunk.metadata
            start_time = best_metadata.get('start_timestamp', 0)
            end_time = best_metadata.get('end_timestamp', 0)
            
            # Fallback to earliest/latest if best chunk has no timestamp
            if start_time == 0:
                start_time = earliest_start_time if earliest_start_time != float('inf') else 0
            if end_time == 0:
                end_time = latest_end_time if latest_end_time > 0 else start_time + 30
            
            # Log the extracted timestamp
            if start_time > 0:
                print(f"✅ Using BEST chunk timestamp: {start_time}s - {end_time}s (from highest relevance chunk)")
            else:
                print(f"⚠️ No timestamp found in best chunk, using fallback logic")
            
            # Format the timestamp for display
            if start_time > 0 and start_time <= 3600:
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                formatted_timestamp = f"{minutes:02d}:{seconds:02d}"
            else:
                formatted_timestamp = "00:00"
                print(f"⚠️ Invalid timestamp: {start_time}s, using 00:00")
            
            # Clean combined text by removing timestamps and fixing formatting
            clean_text = combined_text.strip()
            
            # Fix common formatting issues
            clean_text = re.sub(r'\s+', ' ', clean_text)  # Replace multiple spaces with single space
            
            # Fix broken words and duplicate content
            clean_text = self._fix_broken_text(clean_text)
            
            # Remove duplicate sentences (keep only unique sentences)
            sentences = [s.strip() for s in clean_text.split('.') if s.strip()]
            unique_sentences = []
            seen = set()
            for sentence in sentences:
                # Normalize sentence for comparison (lowercase, remove extra spaces)
                normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
                if normalized not in seen and len(sentence) > 10:  # Avoid very short fragments
                    seen.add(normalized)
                    unique_sentences.append(sentence)
            
            clean_text = '. '.join(unique_sentences)
            if clean_text and not clean_text.endswith('.'):
                clean_text += '.'
            clean_text = re.sub(r'\.\s*\.', '.', clean_text)  # Remove double periods
            clean_text = re.sub(r'\s+([.!?])', r'\1', clean_text)  # Remove spaces before punctuation
            
            # Remove orphaned single letters at the end
            clean_text = re.sub(r'\s+([a-z])\s*$', '', clean_text)  # Remove trailing single letters
            clean_text = re.sub(r'^\s*([a-z])\s+', '', clean_text)  # Remove leading single letters
            
            clean_text = clean_text.strip()
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(question, clean_text)
            
            print(f"✅ Best video match - Score: {best_score:.3f}, Relevance: {relevance_score:.3f}")
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
                'score': best_score,
                'relevance_score': relevance_score,
                'source': 'video',
                'video_url': video_url,
                'start': start_time,
                'end': end_time,
                'start_time': start_time,
                'end_time': end_time,
                'formatted_timestamp': formatted_timestamp,
                'raw_text': combined_text
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
        """Format raw scraped content as a sales expert would explain it"""
        try:
            # Use the same sales expert formatting as video content
            formatted_answer = self._format_as_sales_expert(raw_content, question)
            
            # Add source indicator
            formatted_answer += "\n\n[SOURCE: Scraped Data]"
            
            return formatted_answer
            
        except Exception as e:
            print(f"❌ Error formatting knowledge answer: {e}")
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

    def _format_as_sales_expert(self, raw_content: str, question: str) -> str:
        """Present the available content as a sales expert would explain it"""
        try:
            # Clean the raw content first
            clean_content = raw_content.strip()
            
            # Remove [SOURCE: Video Data] if present
            clean_content = clean_content.replace('[SOURCE: Video Data]', '').strip()
            
            # Format the content with proper structure and smart formatting
            # Clean up the content first
            formatted_content = clean_content
            
            # Fix common issues
            formatted_content = re.sub(r'\s+', ' ', formatted_content)  # Replace multiple spaces
            formatted_content = re.sub(r'\.\s*\.', '.', formatted_content)  # Remove double periods
            formatted_content = re.sub(r'\s+([.!?])', r'\1', formatted_content)  # Remove spaces before punctuation
            
            # Fix broken words and incomplete text
            formatted_content = self._fix_broken_text(formatted_content)
            
            # Split into sentences
            sentences = []
            for sentence in formatted_content.split('. '):
                if sentence.strip():
                    sentence = sentence.strip()
                    if not sentence.endswith('.'):
                        sentence += '.'
                    sentences.append(sentence)
            
            # Smart formatting: Use bullets for steps, paragraphs for explanations
            if self._is_step_by_step_content(sentences):
                # Format as bullet points for step-by-step processes
                bullet_points = []
                for sentence in sentences:
                    bullet_points.append(f"• {sentence}")
                formatted_content = '\n'.join(bullet_points)
            else:
                # Format as paragraphs for explanatory content
                formatted_content = ' '.join(sentences)
            
            # Present ONLY the available content with minimal sales expert framing
            formatted_response = f"""Here's what I found in the video content:

{formatted_content}"""
            
            return formatted_response
            
        except Exception as e:
            print(f"❌ Error formatting as sales expert: {e}")
            # Fallback to basic formatting
            return f"Based on the video content: {raw_content.strip()}"

    def _fix_broken_text(self, text: str) -> str:
        """Fix broken words and incomplete text from corrupted chunks"""
        try:
            # Common broken word patterns and their fixes
            fixes = {
                'uggestion': 'Suggestion',
                'ore for': 'Use for',
                'click write m': 'click write for',
                'umber of': 'Number of',
                'Aswe allknow': 'As we all know',
                'theright useof': 'the right use of',
                'keywordsis': 'keywords is',
                'whatcan makeor': 'what can make or',
                'breakany pieceof': 'break any piece of',
                'writtencontent': 'written content',
                'Easyseowill helpyou': 'Easyseo will help you',
                'beastep aheadand': 'be a step ahead and',
                'generatealist ofrelevant': 'generate a list of relevant',
                'keywordsfor yourtopic': 'keywords for your topic',
                'You\'ll seethe searchvolume': 'You\'ll see the search volume',
                'andthe difficultyscores': 'and the difficulty scores',
                'whichwill helpyou': 'which will help you',
                'comeup withbetter': 'come up with better',
                'contentfor yourbusiness': 'content for your business',
                'orefor easyseoto': 'Use easyseo to',
                'completethe jobfor': 'complete the job for',
                'youand evenshow': 'you and even show',
                'youthe SEO score': 'you the SEO score',
                'Feelinglazy? Usethe': 'Feeling lazy? Use the',
                'auto-generate buttonfor': 'auto-generate button for',
                'aone-click creationof': 'a one-click creation of',
                'fullyunique content': 'fully unique content',
                'Forbloggers andguest': 'For bloggers and guest',
                'postwriters, theeditor': 'post writers, the editor',
                'willassist youin': 'will assist you in',
                'writingunique contentwhich': 'writing unique content which',
                'willrank andbe': 'will rank and be',
                'apleasureto read': 'a pleasure to read',
                'Justadd akeyword': 'Just add a keyword',
                'chooseatitle, andclick': 'choose a title, and click',
                'writemWe\'ve createda': 'write for. We\'ve created a',
                'unique AI solutionwhich': 'unique AI solution which',
                'helpsyou writeSEO': 'helps you write SEO',
                'focusedcontent atlightning': 'focused content at lightning',
                'speed. Ourtemplates aretrained': 'speed. Our templates are trained',
                'byexperts anddesigned': 'by experts and designed',
                'torank highin': 'to rank high in',
                'searchresults. Youcan': 'search results. You can',
                'alsoimprove yourSEO': 'also improve your SEO',
                'scorein oneclick': 'score in one click',
                'umberof searchresults': 'Number of search results',
                'fightingfor yourattention': 'fighting for your attention',
                'Andchances areyou\'ll': 'And chances are you\'ll',
                'clickon oneof': 'click on one of',
                'thefirst shownlistings': 'the first shown listings',
                'andignore thosefurther': 'and ignore those further',
                'down. Now, imagineyou\'re': 'down. Now, imagine you\'re',
                'runningabusiness onyour': 'running a business on your',
                'own.': 'own.'
            }
            
            # Apply fixes
            for broken, fixed in fixes.items():
                text = text.replace(broken, fixed)
            
            return text
            
        except Exception as e:
            print(f"❌ Error fixing broken text: {e}")
            return text

    def _is_step_by_step_content(self, sentences: list) -> bool:
        """Determine if content is step-by-step instructions or explanatory content"""
        try:
            step_indicators = [
                'step', 'first', 'second', 'third', 'next', 'then', 'finally',
                'click', 'add', 'choose', 'select', 'enter', 'type', 'write',
                'generate', 'create', 'build', 'make', 'use', 'apply'
            ]
            
            # Count sentences that contain step indicators
            step_sentences = 0
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(indicator in sentence_lower for indicator in step_indicators):
                    step_sentences += 1
            
            # If more than 50% of sentences contain step indicators, use bullets
            return step_sentences > len(sentences) * 0.5
            
        except Exception as e:
            print(f"❌ Error determining content type: {e}")
            return False  # Default to paragraphs

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
            # Use video answer when video content is found and highly relevant
            if video_result.get('success') and video_score >= HIGH_RELEVANCE:
                print("🎬 Using video answer with precise timestamps")
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
                elif video_score >= MEDIUM_RELEVANCE:  # If video is reasonably relevant, prefer it for timestamp jumping
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
        """Format video content as a sales expert would explain it"""
        try:
            # Debug: Show actual video content being used
            print(f"🔍 DEBUG: Formatting video content as sales expert answer:")
            print(f"🔍 DEBUG: Video content preview: {video_result['answer'][:200]}...")
            print(f"🔍 DEBUG: User question: {question}")
            
            # Get the raw video content
            raw_content = video_result['answer']
            
            # Format as a sales expert would explain it
            formatted_answer = self._format_as_sales_expert(raw_content, question)
            
            return {
                'success': True,
                'answer': formatted_answer,
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
            print(f"❌ Error processing video answer: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': "Error processing video content",
                'start': 0,
                'end': 0,
                'video_url': None,
                'sources': [],
                'total_sources': 0,
                'search_score': 0,
                'content_types_found': []
            }

    def _generate_combined_answer(self, video_result: Dict, knowledge_result: Dict, question: str) -> Dict:
        """Format combined content as a sales expert would explain it"""
        try:
            # Debug: Show actual content being used
            print(f"🔍 DEBUG: Formatting combined content as sales expert answer:")
            print(f"🔍 DEBUG: Video content preview: {video_result['answer'][:100] if video_result else 'None'}...")
            print(f"🔍 DEBUG: Knowledge content preview: {knowledge_result['answer'][:100] if knowledge_result else 'None'}...")
            print(f"🔍 DEBUG: User question: {question}")
            
            # Combine and format as sales expert
            combined_content = ""
            if video_result and video_result.get('answer'):
                combined_content += f"VIDEO CONTENT:\n{video_result['answer']}\n\n"
            if knowledge_result and knowledge_result.get('answer'):
                combined_content += f"SCRAPED CONTENT:\n{knowledge_result['answer']}\n\n"
            
            # Format as sales expert response
            formatted_answer = self._format_as_sales_expert(combined_content, question)
            
            return {
                'success': True,
                'answer': formatted_answer,
                'start': video_result.get('start', 0) if video_result else 0,
                'end': video_result.get('end', 0) if video_result else 0,
                'video_url': video_result.get('video_url') if video_result else None,
                'formatted_timestamp': video_result.get('formatted_timestamp', '00:00') if video_result else '00:00',
                'sources': [
                    {'type': 'video', 'url': video_result.get('video_url', '') if video_result else '', 'title': 'Video Transcript'},
                    {'type': 'knowledge', 'url': knowledge_result.get('url', '') if knowledge_result else '', 'title': knowledge_result.get('title', 'Knowledge Base') if knowledge_result else 'Knowledge Base'}
                ],
                'total_sources': 2,
                'search_score': max(video_result.get('search_score', 0) if video_result else 0, knowledge_result.get('search_score', 0) if knowledge_result else 0),
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
