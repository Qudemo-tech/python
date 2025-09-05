#!/usr/bin/env python3
"""
Simple Enhanced QA System - Fast & Accurate (Parallel Search + Strict Gating)

Key changes:
- Parallel search across video + knowledge (ThreadPoolExecutor)
- Real relevance scoring (uses Pinecone score + dynamic keyword/context)
- Strict "no-video" guarantee when knowledge is more relevant or video < threshold
- Robust video/knowledge detection + timestamp validation/clamping
- Lower latency defaults: smaller top_k, score_threshold, optional GPT formatting
- Content relevance prioritization over raw scores
- Title-first matching with domain guard to prevent irrelevant videos
"""

print("🚀 LOADING ENHANCED QA SIMPLE - UPDATED VERSION WITH CONTENT RELEVANCE PRIORITIZATION - VERSION 3.0")

import os
import re
import math
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from pinecone import Pinecone


class SimpleEnhancedQA:
    # ------- Tunables (strict gating to prevent irrelevant videos) -------
    VIDEO_MIN_REL = 0.58          # minimum relevance needed to show any video
    KNOWLEDGE_MIN_REL = 0.60      # minimum relevance for knowledge
    COMBINE_BOTH_MIN = 0.66       # both must exceed this to combine
    DOMINANCE_DELTA = 0.08        # margin to declare a clear winner

    TOP_K = 8                     # smaller = faster; good enough in practice
    SCORE_THRESHOLD = 0.20        # early reject weak hits
    USE_GPT_FORMATTING = os.getenv('QA_USE_GPT_FORMATTING', '0') == '1'

    def __init__(self):
        """Initialize simple enhanced QA system"""
        # Pinecone Standard Plan - Multiple indexes
        self.indexes = {
            'video': 'qudemo-video-index',
            'knowledge': 'qudemo-knowledge-index',
            'web': 'qudemo-web-index',
            'legacy': 'qudemo-index'  # Fallback for existing content
        }
        
        # Simple LRU cache for question embeddings (capacity 100)
        self._embedding_cache = OrderedDict()
        self._max_cache_size = 100

    def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Ask a question and get an intelligent answer from video and knowledge sources"""
        try:
            print("🔥🔥🔥 UPDATED QA SYSTEM IS RUNNING - CONTENT RELEVANCE PRIORITIZATION ACTIVE 🔥🔥🔥")
            print(f"❓ Question for {company_name} qudemo {qudemo_id}: {question}")
            
            # Get question embedding (with caching)
            question_embedding = self._get_embedding(question)
            
            # Run video and knowledge searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both searches
                video_future = executor.submit(
                    self._search_video_transcripts, 
                    question, company_name, qudemo_id, question_embedding
                )
                knowledge_future = executor.submit(
                    self._search_knowledge_sources, 
                    question, company_name, qudemo_id, question_embedding
                )
                
                # Wait for both to complete
                video_result = video_future.result()
                knowledge_result = knowledge_future.result()
            
            # Select the best answer with strict gating
            final_answer = self._select_best_answer_strict(video_result, knowledge_result, question)
            
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
        """Search video transcripts for relevant content with Pinecone filters"""
        try:
            print(f"🎬 Searching video transcripts for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            print(f"🔍 Searching in namespace: {namespace}")
            
            # Video filter to reduce noise
            video_filter = {
                "$or": [
                    {"source_type": {"$in": ["video_transcript", "youtube", "loom", "video"]}},
                    {"source": {"$in": ["video", "youtube", "loom"]}}
                ]
            }
            
            # Try video index first, then legacy index as fallback
            for index_name in [self.indexes['video'], self.indexes['legacy']]:
                try:
                    index = pc.Index(index_name)
                    query_results = index.query(
                        vector=question_embedding,
                        top_k=self.TOP_K,
                        include_metadata=True,
                        namespace=namespace,
                        score_threshold=self.SCORE_THRESHOLD,
                        filter=video_filter
                    )
                    
                    if query_results.matches:
                        print(f"✅ Found {len(query_results.matches)} matches in {index_name}")
                        return self._process_video_matches(query_results, question, company_name, qudemo_id)
                        
                except Exception as index_error:
                    print(f"⚠️ {index_name} search failed: {index_error}")
                    continue
            
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
        """Process video matches and extract best answer with robust filtering"""
        try:
            # Filter for actual video content using robust detection
            video_matches = []
            for match in query_results.matches:
                if self._is_video_content(match.metadata):
                    video_matches.append(match)
                    print(f"✅ Accepted video match: {match.metadata.get('title', 'NO_TITLE')[:50]}...")
            
            if not video_matches:
                print("⚠️ No video content found after filtering")
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
            
            # Extract and validate timestamps
            start_time, end_time = self._extract_and_validate_timestamps(metadata, raw_text)
            
            # Clean text by removing timestamps
            clean_text = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', raw_text).strip()
            
            # Calculate relevance score using real Pinecone score
            relevance_score = self._calculate_relevance_score(question, clean_text, best_match.score)
            
            print(f"✅ Best video match - Score: {best_match.score:.3f}, Relevance: {relevance_score:.3f}")
            print(f"📹 Video URL: {video_url}")
            print(f"⏰ Timestamp: {start_time}s - {end_time}s")
            
            # Add source indicator
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
                'raw_text': raw_text,
                'title': metadata.get('title', '')
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
        """Search knowledge sources for relevant content with Pinecone filters"""
        try:
            print(f"📚 Searching knowledge sources for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            print(f"🔍 Searching in namespace: {namespace}")
            
            # Knowledge filter to reduce noise
            knowledge_filter = {
                "$or": [
                    {"source_type": {"$in": ["web_scraping", "knowledge"]}},
                    {"source": {"$in": ["web_scraping", "knowledge"]}}
                ]
            }
            
            # Try knowledge index first, then legacy index as fallback
            for index_name in [self.indexes['knowledge'], self.indexes['legacy']]:
                try:
                    index = pc.Index(index_name)
                    query_results = index.query(
                        vector=question_embedding,
                        top_k=self.TOP_K,
                        include_metadata=True,
                        namespace=namespace,
                        score_threshold=self.SCORE_THRESHOLD,
                        filter=knowledge_filter
                    )
                    
                    if query_results.matches:
                        print(f"✅ Found {len(query_results.matches)} matches in {index_name}")
                        return self._process_knowledge_matches(query_results, question)
                        
                except Exception as index_error:
                    print(f"⚠️ {index_name} search failed: {index_error}")
                    continue
            
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
        """Process knowledge matches and extract best answer with robust filtering"""
        try:
            # Filter for actual knowledge content using robust detection
            knowledge_matches = []
            for match in query_results.matches:
                if self._is_knowledge_content(match.metadata):
                    knowledge_matches.append(match)
                    print(f"✅ Accepted knowledge match: {match.metadata.get('title', 'NO_TITLE')[:50]}...")
            
            if not knowledge_matches:
                print("⚠️ No knowledge content found after filtering")
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'knowledge'
                }
            
            best_match = knowledge_matches[0]
            metadata = best_match.metadata
            content = metadata.get('text', '')
            title = metadata.get('title', '')
            
            print(f"🔍 Knowledge content debug:")
            print(f"   Raw content length: {len(content)}")
            print(f"   Raw content preview: '{content[:200]}...'")
            print(f"   Raw title: '{title}'")
            
            # Calculate relevance score using real Pinecone score
            relevance_score = self._calculate_relevance_score(question, content, best_match.score)
            
            print(f"✅ Best knowledge match - Score: {best_match.score:.3f}, Relevance: {relevance_score:.3f}")
            
            # Clean the content
            clean_content = self._clean_content(content)
            
            # Format the answer (with optional GPT formatting)
            if self.USE_GPT_FORMATTING:
                formatted_answer = self._format_knowledge_answer(question, clean_content)
            else:
                formatted_answer = clean_content + "\n\n[SOURCE: Scraped Data]"
            
            return {
                'success': True,
                'answer': formatted_answer,
                'raw_content': clean_content,  # Store cleaned content for title matching
                'raw_title': title,  # Store raw title for title matching
                'score': best_match.score,
                'relevance_score': relevance_score,
                'source': 'knowledge',
                'url': metadata.get('url', ''),
                'title': title
            }
            
        except Exception as e:
            print(f"❌ Error processing knowledge matches: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'knowledge'
            }

    def _select_best_answer_strict(self, video_result: Dict, knowledge_result: Dict, question: str) -> Dict:
        """Strict gating selection policy - prioritize content relevance over raw scores"""
        try:
            print(f"🤔 Selecting best answer with strict gating")
            
            video_score = video_result.get('relevance_score', 0) if video_result.get('success') else 0
            knowledge_score = knowledge_result.get('relevance_score', 0) if knowledge_result.get('success') else 0
            
            print(f"📊 Video score: {video_score:.3f}, Knowledge score: {knowledge_score:.3f}")
            print(f"📊 Min thresholds - Video: {self.VIDEO_MIN_REL}, Knowledge: {self.KNOWLEDGE_MIN_REL}")
            
            # Apply domain guard to video - must contain key phrases from question
            video_passes_domain_guard = True
            if video_result.get('success'):
                video_text = video_result.get('raw_text', video_result.get('answer', ''))
                video_title = video_result.get('title', '')
                video_passes_domain_guard = self._domain_guard(question, video_text or video_title)
            
            # Check if sources meet minimum relevance thresholds
            v_ok = video_result.get('success') and video_score >= self.VIDEO_MIN_REL and video_passes_domain_guard
            k_ok = knowledge_result.get('success') and knowledge_score >= self.KNOWLEDGE_MIN_REL
            
            print(f"🔍 Source qualification:")
            print(f"   Video: success={video_result.get('success')}, score={video_score:.3f} >= {self.VIDEO_MIN_REL}, domain_guard={video_passes_domain_guard} → {v_ok}")
            print(f"   Knowledge: success={knowledge_result.get('success')}, score={knowledge_score:.3f} >= {self.KNOWLEDGE_MIN_REL} → {k_ok}")
            
            # Strict gating logic
            if not v_ok and k_ok:
                # Knowledge-only => DO NOT include any video fields
                print(f"📚 Knowledge only (video failed: score={video_score:.3f} < {self.VIDEO_MIN_REL} or domain_guard={video_passes_domain_guard})")
                return self._knowledge_only_payload(knowledge_result)
            
            if v_ok and not k_ok:
                # Video-only => return video_url + start/end
                print(f"🎬 Video only (knowledge not relevant enough: {knowledge_score:.3f} < {self.KNOWLEDGE_MIN_REL})")
                return self._video_only_payload(video_result)
            
            if not v_ok and not k_ok:
                # Neither source is relevant enough
                print("❌ Neither source meets minimum relevance threshold")
                return self._no_sources_found()
            
            # Both qualify - check for content relevance first
            # Use title-first matching
            knowledge_title = knowledge_result.get('raw_title', '')
            knowledge_content = knowledge_result.get('raw_content', '')
            video_title = video_result.get('title', '')
            video_content = video_result.get('raw_text', video_result.get('answer', ''))
            
            knowledge_has_exact_match = self._has_exact_title_match(question, knowledge_title, knowledge_content)
            video_has_exact_match = self._has_exact_title_match(question, video_title, video_content)
            
            print(f"🔍 Content relevance check:")
            print(f"   Knowledge exact match: {knowledge_has_exact_match}")
            print(f"   Video exact match: {video_has_exact_match}")
            print(f"   Knowledge answer preview: {knowledge_result.get('answer', '')[:100]}...")
            
            if knowledge_has_exact_match and not video_has_exact_match:
                print(f"📚 Knowledge has exact title match, video doesn't - using knowledge")
                return self._knowledge_only_payload(knowledge_result)
            
            if video_has_exact_match and not knowledge_has_exact_match:
                print(f"🎬 Video has exact title match, knowledge doesn't - using video")
                return self._video_only_payload(video_result)
            
            # Check for dominance
            score_diff = video_score - knowledge_score
            if score_diff >= self.DOMINANCE_DELTA:
                print(f"🎬 Video dominates (diff: {score_diff:.3f} >= {self.DOMINANCE_DELTA})")
                return self._video_only_payload(video_result)
            
            if -score_diff >= self.DOMINANCE_DELTA:
                print(f"📚 Knowledge dominates (diff: {-score_diff:.3f} >= {self.DOMINANCE_DELTA})")
                return self._knowledge_only_payload(knowledge_result)
            
            # Close call - combine only if both are strong
            if video_score >= self.COMBINE_BOTH_MIN and knowledge_score >= self.COMBINE_BOTH_MIN:
                print(f"🔄 Both strong - combining (v: {video_score:.3f}, k: {knowledge_score:.3f} >= {self.COMBINE_BOTH_MIN})")
                return self._combined_payload(video_result, knowledge_result, question)
            
            # Otherwise pick the slightly better one, but keep gating guarantees
            if video_score >= knowledge_score:
                print(f"🎬 Video slightly better (v: {video_score:.3f} >= k: {knowledge_score:.3f})")
                return self._video_only_payload(video_result)
            else:
                print(f"📚 Knowledge slightly better (k: {knowledge_score:.3f} > v: {video_score:.3f})")
                return self._knowledge_only_payload(knowledge_result)
                
        except Exception as e:
            print(f"❌ Error in strict selection: {e}")
            return self._no_sources_found()

    def _knowledge_only_payload(self, knowledge_result: Dict) -> Dict:
        """Knowledge-only payload - NO video fields"""
        return {
            'success': True,
            'answer': knowledge_result['answer'],
            'start': 0,
            'end': 0,
            'video_url': None,  # CRITICAL: No video URL
            'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}],
            'total_sources': 1,
            'search_score': knowledge_result.get('score', 0),
            'content_types_found': ['knowledge'],
            'difficulty_level': 'intermediate',
            'estimated_time': '2-3 minutes'
        }

    def _video_only_payload(self, video_result: Dict) -> Dict:
        """Video-only payload with timestamps"""
        return {
            'success': True,
            'answer': video_result['answer'],
            'start': video_result.get('start', 0),
            'end': video_result.get('end', 0),
            'video_url': video_result.get('video_url', ''),
            'sources': [{'type': 'video', 'url': video_result.get('video_url', ''), 'title': 'Video Transcript'}],
            'total_sources': 1,
            'search_score': video_result.get('score', 0),
            'content_types_found': ['video'],
            'difficulty_level': 'intermediate',
            'estimated_time': '3-5 minutes'
        }

    def _combined_payload(self, video_result: Dict, knowledge_result: Dict, question: str) -> Dict:
        """Combined payload when both sources are strong"""
        # Simple combination - prefer knowledge content but include video timestamp
        combined_answer = f"{knowledge_result['answer']}\n\n{video_result['answer']}"
        
        return {
            'success': True,
            'answer': combined_answer,
            'start': video_result.get('start', 0),
            'end': video_result.get('end', 0),
            'video_url': video_result.get('video_url', ''),
            'sources': [
                {'type': 'video', 'url': video_result.get('video_url', ''), 'title': 'Video Transcript'},
                {'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}
            ],
            'total_sources': 2,
            'search_score': max(video_result.get('score', 0), knowledge_result.get('score', 0)),
            'content_types_found': ['video', 'knowledge'],
            'difficulty_level': 'intermediate',
            'estimated_time': '4-6 minutes'
        }

    def _no_sources_found(self) -> Dict:
        """No relevant sources found"""
        return {
            'success': False,
            'answer': "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic.",
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

    def _calculate_relevance_score(self, question: str, content: str, pinecone_score: float) -> float:
        """Calculate relevance score using real Pinecone score + lexical/context signals"""
        try:
            # 1) Semantic (dominant signal) - use actual Pinecone score
            sem = self._clamp(pinecone_score, 0, 1) * 0.60

            # 2) Keyword coverage: unique word overlap
            q_terms = set(self._tokenize(question))
            c_terms = set(self._tokenize(content))
            overlap = len(q_terms & c_terms)
            kw = (overlap / max(1, len(q_terms))) * 0.25
            
            # Debug keyword matching
            if overlap > 0:
                print(f"🔍 Keyword overlap: {overlap}/{len(q_terms)} words: {list(q_terms & c_terms)}")
                print(f"🔍 Keyword score calculation: {overlap}/{len(q_terms)} * 0.25 = {kw:.3f}")

            # 3) Exact title match boost (for knowledge sources)
            title_boost = 0.0
            question_clean = question.lower().replace('?', '').strip()
            content_clean = content.lower().replace('?', '').strip()
            
            # Check for exact question match in title/content
            if question_clean in content_clean or content_clean in question_clean:
                title_boost = 0.15
                print(f"🎯 Exact title match detected: '{question_clean}' in '{content_clean[:100]}...'")
            # Also check for partial matches with key terms
            elif 'how do i' in question_clean and 'how do i' in content_clean:
                # Check if the main action words match
                question_words = set(question_clean.split())
                content_words = set(content_clean.split())
                overlap = len(question_words & content_words)
                if overlap >= 3:  # At least 3 words overlap
                    title_boost = 0.10
                    print(f"🎯 Partial title match detected: {overlap} words overlap")

            # 4) Context boost for "how-to" verbs
            ctx = 0.0
            if re.search(r'\b(click|open|select|go to|create|configure|enable|disable|set up|install|pin|update)\b', content.lower()):
                ctx = 0.10
                print(f"🔧 How-to context detected in content")

            combined_score = sem + kw + title_boost + ctx
            print(f"📊 Relevance breakdown - Semantic: {sem:.3f}, Keyword: {kw:.3f}, Title: {title_boost:.3f}, Context: {ctx:.3f} = {combined_score:.3f}")
            return self._clamp(combined_score, 0, 1)
            
        except Exception as e:
            print(f"❌ Error calculating relevance score: {e}")
            return 0.0

    def _is_video_content(self, metadata: Dict) -> bool:
        """Robust video content detection"""
        st = (metadata.get('source_type') or '').lower()
        src = (metadata.get('source') or '').lower()
        has_url = bool(metadata.get('video_url') or metadata.get('url'))
        
        return (('video' in st) or ('youtube' in st) or ('loom' in st) or
                (src in ['video','youtube','loom']) or
                st in ['video_transcript','youtube','loom']) and has_url

    def _is_knowledge_content(self, metadata: Dict) -> bool:
        """Robust knowledge content detection"""
        st = (metadata.get('source_type') or '').lower()
        src = (metadata.get('source') or '').lower()
        looks_video = any(t in st for t in ['video','youtube','loom']) or src in ['video','youtube','loom']
        
        return (st in ['web_scraping','knowledge'] or src in ['web_scraping','knowledge'] or not looks_video)

    def _extract_and_validate_timestamps(self, metadata: Dict, raw_text: str) -> Tuple[int, int]:
        """Extract and validate timestamps with clamping"""
        start = int(metadata.get('start') or 0)
        end = int(metadata.get('end') or 0)

        if start == 0 and end == 0:
            # Fallback parse [MM:SS] or [HH:MM:SS]
            ts_match = re.search(r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]', raw_text)
            if ts_match:
                if len(ts_match.groups()) == 3 and ts_match.group(3):  # [HH:MM:SS]
                    start = int(ts_match.group(1)) * 3600 + int(ts_match.group(2)) * 60 + int(ts_match.group(3))
                else:  # [MM:SS]
                    start = int(ts_match.group(1)) * 60 + int(ts_match.group(2))
                end = start + 30

        # Order & clamp
        start = max(0, start)
        end = max(start, end)
        cap = 6 * 3600  # 6 hours max
        return min(start, cap), min(end, cap)

    def _clean_content(self, content: str) -> str:
        """Clean content by removing markdown and unwanted elements"""
        # Remove markdown formatting
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)      # Remove italic
        content = re.sub(r'`(.*?)`', r'\1', content)        # Remove code
        content = re.sub(r'#{1,6}\s*(.*)', r'\1', content)  # Remove headers
        content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # Remove links
        
        # Remove table formatting
        content = re.sub(r'\|.*?\|', '', content)  # Remove table rows
        content = re.sub(r'\|-+\|', '', content)   # Remove table separators
        
        # Remove bullet points and list markers
        content = re.sub(r'^\s*[\*\-+]\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
        
        # Remove extra whitespace and clean up
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Remove extra blank lines
        content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)  # Remove leading spaces
        content = content.strip()
        
        return content

    def _format_knowledge_answer(self, question: str, raw_content: str) -> str:
        """Format raw scraped content using GPT (optional)"""
        if not self.USE_GPT_FORMATTING:
            return raw_content + "\n\n[SOURCE: Scraped Data]"
            
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
            
            # Add source indicator
            formatted_answer += "\n\n[SOURCE: Scraped Data]"
            
            # Fallback to basic cleaning if GPT fails
            if not formatted_answer or len(formatted_answer) < 50:
                print("⚠️ GPT formatting failed, using basic cleaning")
                return raw_content + "\n\n[SOURCE: Scraped Data]"
            
            return formatted_answer
            
        except Exception as e:
            print(f"❌ Error formatting knowledge answer with GPT: {e}")
            return raw_content + "\n\n[SOURCE: Scraped Data]"

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with simple LRU cache"""
        # Check cache first
        if text in self._embedding_cache:
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(text)
            return self._embedding_cache[text]
        
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Add to cache
            self._embedding_cache[text] = embedding
            
            # Remove oldest if cache is full
            if len(self._embedding_cache) > self._max_cache_size:
                self._embedding_cache.popitem(last=False)
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            # Return a dummy embedding if OpenAI fails
            return [0.0] * 3072  # text-embedding-3-large uses 3072 dimensions

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for keyword matching"""
        return re.findall(r'\b\w+\b', text.lower())

    def _normalize(self, text: str) -> str:
        """Normalize text: lowercase, strip punctuation, collapse spaces"""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract noun phrases (bigrams) from text, filtering out stopwords"""
        stopwords = {'how', 'do', 'i', 'you', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = self._tokenize(text)
        phrases = []
        
        # Extract bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if words[i] not in stopwords and words[i+1] not in stopwords:
                phrases.append(bigram)
        
        return phrases

    def _domain_guard(self, question: str, text: str) -> bool:
        """Domain guard: video must contain key phrases from question to qualify"""
        q_phrases = self._extract_phrases(self._normalize(question))
        t = self._normalize(text)
        
        print(f"🔍 Domain guard check:")
        print(f"   Question phrases: {q_phrases}")
        print(f"   Text preview: '{t[:100]}...'")
        
        # Require all multiword phrases from question to appear in text
        for phrase in q_phrases:
            if len(phrase.split()) >= 2 and phrase not in t:
                print(f"   ❌ Missing phrase: '{phrase}'")
                return False
        
        print(f"   ✅ All phrases found in text")
        return True

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))

    def _has_exact_title_match(self, question: str, title: str, content: str = "") -> bool:
        """Check if title or content has exact title match with question"""
        q = self._normalize(question)
        t = self._normalize(title)
        
        print(f"🔍 Title match check:")
        print(f"   Question: '{q}'")
        print(f"   Title: '{t}'")
        print(f"   Content: '{content[:100] if content else 'N/A'}...'")
        
        if not t:
            print(f"   ❌ No title provided")
            return False

        # Strong checks on title first
        if q in t:
            print(f"   ✅ Exact title match found!")
            return True

        # Phrase-level: require key noun phrases from question
        phrases = self._extract_phrases(q)
        if phrases:
            phrase_matches = [p for p in phrases if p in t]
            if phrase_matches:
                print(f"   ✅ Phrase match found: {phrase_matches}")
                return True

        # Optional: backoff to content if title didn't hit
        if content:
            c = self._normalize(content)
            if q in c:
                print(f"   ✅ Content match found!")
                return True
            
        print(f"   ❌ No match found")
        return False


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
    global simple_enhanced_qa
    if simple_enhanced_qa is None:
        initialize_simple_enhanced_qa()
    return simple_enhanced_qa