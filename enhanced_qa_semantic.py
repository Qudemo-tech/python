#!/usr/bin/env python3
"""
Enhanced Semantic Q&A System
Focuses on semantic understanding, context-aware relevance, and prevents random video suggestions
"""

import os
import re
import time
import json
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import openai
from pinecone import Pinecone
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedSemanticQA:
    """Enhanced Q&A system with semantic understanding and context-aware relevance"""
    
    # Enhanced thresholds for better quality control (relaxed for better coverage)
    KNOWLEDGE_MIN_RELEVANCE = 0.3   # Lowered threshold for knowledge
    VIDEO_MIN_RELEVANCE = 0.2       # Much lower threshold for videos
    SEMANTIC_SIMILARITY_THRESHOLD = 0.1  # Very low threshold to allow more content
    CONTEXT_UNDERSTANDING_THRESHOLD = 0.3  # Lowered context understanding threshold
    
    # Retrieval parameters
    TOP_K_RECALL = 25
    TOP_K_RERANK = 10
    SCORE_THRESHOLD = 0.05  # Much lower initial threshold
    
    def __init__(self):
        """Initialize enhanced semantic QA system"""
        # Pinecone indexes
        self.indexes = {
            'video': 'qudemo-video-index',
            'knowledge': 'qudemo-knowledge-index',
            'legacy': 'qudemo-index'
        }
        
        # OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Pinecone client
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Enhanced embedding cache
        self._embedding_cache = OrderedDict()
        self._cache_timestamps = {}
        self.CACHE_SIZE = 200  # Increased cache size
        self.CACHE_TTL = 24 * 60 * 60  # 24 hours
        self._cache_hits = 0
        self._cache_misses = 0
    
    def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Main QA entry point with enhanced semantic understanding"""
        try:
            start_time = time.time()
            print(f"🧠 Enhanced Semantic QA: {question}")
            
            # Stage 1: Question analysis and intent understanding
            question_analysis = self._analyze_question_intent(question)
            
            # Stage 2: Enhanced semantic retrieval
            candidates = self._enhanced_semantic_retrieval(question, question_analysis, company_name, qudemo_id)
            
            # Stage 3: Context-aware relevance scoring
            scored_candidates = self._context_aware_scoring(question, question_analysis, candidates)
            
            # Stage 4: Intelligent answer selection
            final_answer = self._intelligent_answer_selection(question, question_analysis, scored_candidates)
            
            # Stage 5: Quality assurance and formatting
            formatted_answer = self._quality_assurance_formatting(final_answer, question, question_analysis)
            
            # Observability logging
            latency = int((time.time() - start_time) * 1000)
            self._log_enhanced_observability(question, question_analysis, candidates, scored_candidates, final_answer, latency)
            
            return formatted_answer
            
        except Exception as e:
            print(f"❌ Enhanced Semantic QA Error: {e}")
            return self._safe_fallback()
    
    def _analyze_question_intent(self, question: str) -> Dict:
        """Analyze question intent and extract key concepts"""
        try:
            # Use GPT to analyze question intent
            prompt = f"""
            Analyze this question and extract key information:
            Question: "{question}"
            
            Provide a JSON response with:
            1. intent_type: "create", "edit", "delete", "view", "how_to", "what_is", "troubleshoot"
            2. main_action: The primary action being asked for
            3. key_entities: List of important entities/objects mentioned
            4. context_clues: List of context clues that help understand the question
            5. expected_answer_type: "procedural", "explanatory", "troubleshooting", "reference"
            6. complexity_level: "simple", "moderate", "complex"
            
            Example:
            {{
                "intent_type": "how_to",
                "main_action": "create disqualified leads",
                "key_entities": ["disqualified leads", "leads"],
                "context_clues": ["create", "how to"],
                "expected_answer_type": "procedural",
                "complexity_level": "moderate"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            try:
                analysis = json.loads(result)
                print(f"🔍 Question Analysis: {analysis}")
                return analysis
            except:
                # Fallback analysis
                return self._fallback_question_analysis(question)
                
        except Exception as e:
            print(f"❌ Question analysis error: {e}")
            return self._fallback_question_analysis(question)
    
    def _fallback_question_analysis(self, question: str) -> Dict:
        """Fallback question analysis using simple pattern matching"""
        question_lower = question.lower()
        
        # Intent detection
        if any(word in question_lower for word in ['how to', 'how do', 'how can', 'how should']):
            intent_type = "how_to"
        elif any(word in question_lower for word in ['create', 'make', 'add', 'new']):
            intent_type = "create"
        elif any(word in question_lower for word in ['edit', 'modify', 'change', 'update']):
            intent_type = "edit"
        elif any(word in question_lower for word in ['delete', 'remove', 'delete']):
            intent_type = "delete"
        else:
            intent_type = "view"
        
        # Extract key entities (simple approach)
        key_entities = []
        for word in question_lower.split():
            if len(word) > 3 and word not in ['how', 'to', 'do', 'can', 'should', 'what', 'is', 'are']:
                key_entities.append(word)
        
        return {
            "intent_type": intent_type,
            "main_action": question_lower,
            "key_entities": key_entities,
            "context_clues": question_lower.split(),
            "expected_answer_type": "procedural" if intent_type == "how_to" else "explanatory",
            "complexity_level": "moderate"
        }
    
    def _enhanced_semantic_retrieval(self, question: str, question_analysis: Dict, company_name: str, qudemo_id: str) -> Dict:
        """Enhanced semantic retrieval with BEST CHUNK ONLY strategy"""
        try:
            # Get question embedding
            question_embedding = self._get_embedding(question, model="text-embedding-3-large")
            
            # Create enhanced query with intent context
            enhanced_query = self._create_enhanced_query(question, question_analysis)
            enhanced_embedding = self._get_embedding(enhanced_query, model="text-embedding-3-large")
            
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Search both indexes with enhanced parameters - but limit to top results
            video_results = self._search_with_intent('video', namespace, question_embedding, enhanced_embedding, question_analysis)
            knowledge_results = self._search_with_intent('knowledge', namespace, question_embedding, enhanced_embedding, question_analysis)
            
            # Note: Filtering moved to after scoring to ensure enhanced_score is available
            
            print(f"🔍 Retrieved: {len(video_results)} video chunks, {len(knowledge_results)} knowledge chunks")
            
            return {
                'video': video_results,
                'knowledge': knowledge_results
            }
            
        except Exception as e:
            print(f"❌ Enhanced retrieval error: {e}")
            return {'video': [], 'knowledge': []}
    
    def _create_enhanced_query(self, question: str, question_analysis: Dict) -> str:
        """Create enhanced query with intent context"""
        intent_type = question_analysis.get('intent_type', '')
        main_action = question_analysis.get('main_action', '')
        key_entities = question_analysis.get('key_entities', [])
        
        # Create context-enhanced query
        enhanced_parts = [question]
        
        if intent_type == "create":
            enhanced_parts.append(f"how to create {main_action}")
            enhanced_parts.append(f"steps to create {main_action}")
        elif intent_type == "how_to":
            enhanced_parts.append(f"tutorial {main_action}")
            enhanced_parts.append(f"guide {main_action}")
        
        # Add entity context
        for entity in key_entities[:3]:  # Limit to top 3 entities
            enhanced_parts.append(f"{entity} process")
            enhanced_parts.append(f"{entity} workflow")
        
        return " ".join(enhanced_parts)
    
    def _search_with_intent(self, index_type: str, namespace: str, question_embedding: List[float], enhanced_embedding: List[float], question_analysis: Dict) -> List[Dict]:
        """Search with intent-aware parameters"""
        try:
            # Get index
            if index_type in self.indexes:
                index = self.pc.Index(self.indexes[index_type])
            else:
                return []
            
            # Set filters based on index type and intent
            filter_query = self._create_intent_filter(index_type, question_analysis)
            
            # Use enhanced embedding for better semantic matching
            results = index.query(
                vector=enhanced_embedding,
                top_k=self.TOP_K_RECALL,
                include_metadata=True,
                namespace=namespace,
                filter=filter_query,
                score_threshold=self.SCORE_THRESHOLD
            )
            
            # Convert to candidate format
            candidates = []
            for match in results.matches:
                candidate = self._build_enhanced_candidate(match, index_type, question_analysis)
                if candidate:
                    candidates.append(candidate)
            
            print(f"✅ {index_type}: {len(candidates)} candidates found")
            return candidates
            
        except Exception as e:
            print(f"❌ {index_type} search error: {e}")
            return []
    
    def _create_intent_filter(self, index_type: str, question_analysis: Dict) -> Dict:
        """Create intent-aware filter for search"""
        base_filter = {}
        
        if index_type == 'knowledge':
            base_filter = {
                "source_type": "web_scraping",
                "content_has_text": True
            }
        elif index_type == 'video':
            base_filter = {
                "source_type": "video_transcript"
            }
        
        # Add intent-specific filters
        intent_type = question_analysis.get('intent_type', '')
        if intent_type == "how_to" and index_type == 'knowledge':
            base_filter["has_steps"] = True
        
        return base_filter
    
    def _build_enhanced_candidate(self, match, index_type: str, question_analysis: Dict) -> Optional[Dict]:
        """Build enhanced candidate with intent context"""
        try:
            metadata = match.metadata
            text = metadata.get('text', '') or metadata.get('content', '')
            
            if not text or len(text.strip()) < 50:
                return None
            
            # Base candidate structure
            candidate = {
                'id': match.id,
                'score': match.score,
                'text': text,
                'title': metadata.get('title', ''),
                'summary': metadata.get('summary', ''),
                'index_type': index_type,
                'metadata': metadata
            }
            
            # Add index-specific fields
            if index_type == 'video':
                candidate.update({
                    'seekable': metadata.get('seekable', True),  # Default to True for video content
                    'has_timestamps': metadata.get('has_timestamps', True),  # Default to True for video content
                    'start': metadata.get('start_timestamp', metadata.get('start', 0)),
                    'end': metadata.get('end_timestamp', metadata.get('end', 0)),
                    'duration': metadata.get('duration', metadata.get('chunk_duration', 0)),
                    'video_url': metadata.get('video_url', '') or metadata.get('url', '')
                })
            elif index_type == 'knowledge':
                candidate.update({
                    'content_has_text': metadata.get('content_has_text', True),
                    'url': metadata.get('url', ''),
                    'has_steps': metadata.get('has_steps', False),
                    'difficulty_level': metadata.get('difficulty_level', 'intermediate')
                })
            
            # Truncate text for processing
            if len(text) > 800:
                candidate['text_truncated'] = text[:400] + "..." + text[-400:]
            else:
                candidate['text_truncated'] = text
            
            return candidate
            
        except Exception as e:
            print(f"❌ Enhanced candidate build error: {e}")
            return None
    
    def _context_aware_scoring(self, question: str, question_analysis: Dict, candidates: Dict) -> Dict:
        """Context-aware relevance scoring with semantic understanding"""
        try:
            all_candidates = candidates['video'] + candidates['knowledge']
            
            if not all_candidates:
                return {'video': [], 'knowledge': []}
            
            # Calculate enhanced scores for each candidate
            for candidate in all_candidates:
                # Use existing Pinecone score as semantic similarity (already computed)
                semantic_score = candidate.get('score', 0)
                
                # Intent alignment score
                intent_score = self._calculate_intent_alignment(candidate, question_analysis)
                
                # Context relevance score
                context_score = self._calculate_context_relevance(candidate, question_analysis)
                
                # Quality score
                quality_score = self._calculate_quality_score(candidate)
                
                # Removed complex timestamp scoring to keep it simple
                
                # Final enhanced score (restored to simple version)
                final_score = (
                    0.35 * semantic_score +
                    0.25 * intent_score +
                    0.25 * context_score +
                    0.15 * quality_score
                )
                
                candidate['enhanced_score'] = final_score
                candidate['semantic_score'] = semantic_score
                candidate['intent_score'] = intent_score
                candidate['context_score'] = context_score
                candidate['quality_score'] = quality_score
            
            # Sort by enhanced score
            all_candidates.sort(key=lambda x: x['enhanced_score'], reverse=True)
            
            # Apply quality thresholds (relaxed for better coverage)
            filtered_candidates = [c for c in all_candidates if c['enhanced_score'] >= 0.1]
            
            # No temporal re-ranking - use enhanced scores directly
            # This ensures only the most relevant content is selected regardless of timestamp
            
            # Separate back into video/knowledge
            video_candidates = [c for c in filtered_candidates if c['index_type'] == 'video']
            knowledge_candidates = [c for c in filtered_candidates if c['index_type'] == 'knowledge']
            
            return {
                'video': video_candidates,
                'knowledge': knowledge_candidates
            }
            
        except Exception as e:
            print(f"❌ Context-aware scoring error: {e}")
            return candidates
    
    def _calculate_intent_alignment(self, candidate: Dict, question_analysis: Dict) -> float:
        """Calculate how well the candidate aligns with question intent"""
        try:
            intent_type = question_analysis.get('intent_type', '')
            main_action = question_analysis.get('main_action', '')
            key_entities = question_analysis.get('key_entities', [])
            
            candidate_text = f"{candidate['title']} {candidate['text_truncated']}".lower()
            
            # Intent type alignment
            intent_score = 0.0
            if intent_type == "create" and any(word in candidate_text for word in ['create', 'make', 'add', 'new']):
                intent_score += 0.4
            elif intent_type == "how_to" and any(word in candidate_text for word in ['how', 'step', 'process', 'tutorial']):
                intent_score += 0.4
            elif intent_type == "edit" and any(word in candidate_text for word in ['edit', 'modify', 'change', 'update']):
                intent_score += 0.4
            
            # Entity alignment
            entity_score = 0.0
            for entity in key_entities:
                if entity in candidate_text:
                    entity_score += 0.2
            
            # Action alignment
            action_score = 0.0
            if main_action and any(word in candidate_text for word in main_action.split()):
                action_score += 0.2
            
            return min(intent_score + entity_score + action_score, 1.0)
            
        except Exception as e:
            print(f"❌ Intent alignment error: {e}")
            return 0.0
    
    def _calculate_context_relevance(self, candidate: Dict, question_analysis: Dict) -> float:
        """Calculate context relevance score"""
        try:
            expected_type = question_analysis.get('expected_answer_type', '')
            complexity = question_analysis.get('complexity_level', '')
            
            candidate_text = f"{candidate['title']} {candidate['text_truncated']}".lower()
            
            # Answer type alignment
            type_score = 0.0
            if expected_type == "procedural" and any(word in candidate_text for word in ['step', 'process', 'procedure', 'workflow']):
                type_score += 0.5
            elif expected_type == "explanatory" and any(word in candidate_text for word in ['explain', 'description', 'overview', 'introduction']):
                type_score += 0.5
            
            # Complexity alignment
            complexity_score = 0.0
            if complexity == "simple" and len(candidate_text) < 500:
                complexity_score += 0.3
            elif complexity == "complex" and len(candidate_text) > 800:
                complexity_score += 0.3
            else:
                complexity_score += 0.2  # Moderate complexity
            
            return min(type_score + complexity_score, 1.0)
            
        except Exception as e:
            print(f"❌ Context relevance error: {e}")
            return 0.0
    
    def _calculate_quality_score(self, candidate: Dict) -> float:
        """Calculate content quality score"""
        try:
            quality_score = 0.0
            
            # Text length quality
            text_length = len(candidate['text'])
            if 100 <= text_length <= 2000:
                quality_score += 0.3
            elif text_length > 2000:
                quality_score += 0.2
            
            # Title quality
            if candidate['title'] and len(candidate['title']) > 10:
                quality_score += 0.2
            
            # Structure quality
            if candidate['index_type'] == 'knowledge' and candidate.get('has_steps', False):
                quality_score += 0.2
            elif candidate['index_type'] == 'video' and candidate.get('has_timestamps', False):
                quality_score += 0.2
            
            # Content completeness
            if candidate['text'] and not candidate['text'].strip().startswith('['):
                quality_score += 0.3
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            print(f"❌ Quality score error: {e}")
            return 0.0
    
    def _calculate_timestamp_relevance(self, candidate: Dict, question: str, question_analysis: Dict) -> float:
        """Calculate how relevant the chunk's timestamps are to the question"""
        try:
            # Only apply to video candidates
            if candidate.get('index_type') != 'video':
                return 0.5  # Neutral score for knowledge candidates
            
            # Extract key terms from question
            question_lower = question.lower()
            key_terms = []
            
            # Extract important words (longer than 3 chars, not common words)
            common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
            words = question_lower.split()
            key_terms = [word for word in words if len(word) > 3 and word not in common_words]
            
            # Get chunk content
            content = candidate.get('content', '').lower()
            
            # Check if key terms appear in the chunk content
            term_matches = 0
            for term in key_terms:
                if term in content:
                    term_matches += 1
            
            # Calculate relevance score based on term matches
            if not key_terms:
                relevance_score = 0.5  # No key terms to match
            else:
                relevance_score = term_matches / len(key_terms)
            
            # Boost score if question intent matches content type
            intent_type = question_analysis.get('intent_type', '')
            if intent_type == 'what_is' and any(word in content for word in ['definition', 'means', 'stands for', 'is a']):
                relevance_score += 0.2
            elif intent_type == 'how_to' and any(word in content for word in ['steps', 'process', 'how to', 'tutorial']):
                relevance_score += 0.2
            elif intent_type == 'view' and any(word in content for word in ['example', 'instance', 'case', 'scenario']):
                relevance_score += 0.2
            
            # NEW: Prefer earlier timestamps for better topic positioning
            start_time = candidate.get('start', 0)
            if start_time > 0:
                # Boost score for earlier timestamps (prefer chunks that start earlier in the video)
                # This helps avoid chunks at the end of topic discussions
                if start_time < 300:  # First 5 minutes
                    relevance_score += 0.4  # Increased boost
                elif start_time < 600:  # First 10 minutes
                    relevance_score += 0.3  # Increased boost
                elif start_time < 900:  # First 15 minutes
                    relevance_score += 0.2  # Increased boost
                else:  # After 15 minutes - penalize
                    relevance_score -= 0.2  # Penalty for late timestamps
            
            # Ensure score is between 0 and 1
            return min(1.0, max(0.0, relevance_score))
            
        except Exception as e:
            print(f"❌ Timestamp relevance error: {e}")
            return 0.5
    
    def _rerank_with_time_bias(self, candidates: list, query: str) -> list:
        """Re-rank candidates with smart temporal bias - only apply when relevance scores are close"""
        try:
            import math
            
            if not candidates:
                return candidates
            
            K = len(candidates)
            
            # 1) Calculate rank normalization
            ranked = sorted(candidates, key=lambda c: c['enhanced_score'], reverse=True)
            for r, c in enumerate(ranked, start=1):
                c['ranknorm'] = (K - r + 1) / K
            
            # 2) Check if temporal bias should be applied
            # Only apply temporal bias if top candidates have similar relevance scores
            if len(ranked) >= 2:
                top_score = ranked[0]['enhanced_score']
                second_score = ranked[1]['enhanced_score']
                score_diff = top_score - second_score
                
                # Only apply temporal bias if scores are close (within 0.1)
                apply_temporal_bias = score_diff < 0.1
            else:
                apply_temporal_bias = False
            
            # 3) No temporal bias - only use relevance scores
            # Completely remove early timestamp selection logic
            lam, gamma, reverse = 0.0, 1.0, False  # No temporal bias at all
            
            # 4) Apply temporal prior to each candidate
            for c in ranked:
                if c.get('index_type') == 'video' and c.get('start', 0) > 0:
                    # Get video duration (estimate from max timestamp or use default)
                    max_duration = max(c.get('end', 0) for c in candidates if c.get('end', 0) > 0)
                    if max_duration == 0:
                        max_duration = 1200  # Default 20 minutes
                    
                    p = c['start'] / max_duration
                    prior_arg = (1 - p) if reverse else p
                    time_prior = math.exp(-lam * (prior_arg ** gamma))
                    
                    # Get unified chunking metadata for quality boost
                    metadata = c.get('metadata', {})
                    content_quality = metadata.get('content_quality', 0.5)
                    is_complete_sentence = metadata.get('is_complete_sentence', False)
                    
                    # Enhanced quality boost for high-quality, complete chunks
                    quality_boost = 1.0
                    if content_quality > 0.8:
                        quality_boost = 1.15  # 15% boost for high quality
                    elif content_quality > 0.6:
                        quality_boost = 1.10  # 10% boost for good quality
                    elif content_quality > 0.4:
                        quality_boost = 1.05  # 5% boost for decent quality
                    
                    if is_complete_sentence:
                        quality_boost *= 1.08  # 8% boost for complete sentences
                    
                    # Combine enhanced score with rank normalization, time prior, and quality boost
                    base = c['enhanced_score'] * 0.9 + c['ranknorm'] * 0.1
                    c['final_score'] = base * time_prior * quality_boost
                else:
                    # For knowledge candidates or video without timestamps, use original score
                    c['final_score'] = c['enhanced_score']
            
            # 4) Sort by final score
            ranked.sort(key=lambda x: x['final_score'], reverse=True)
            
            print(f"🕒 Temporal re-ranking applied (intent: {intent}, λ={lam}, γ={gamma})")
            if ranked:
                best = ranked[0]
                print(f"🕒 Best chunk: {best.get('start', 0)}s (enhanced: {best.get('enhanced_score', 0):.3f}, final: {best['final_score']:.3f})")
                
                # Show top 3 for debugging with unified metadata
                print(f"🕒 Top 3 candidates:")
                for i, c in enumerate(ranked[:3]):
                    metadata = c.get('metadata', {})
                    quality = metadata.get('content_quality', 0)
                    complete = metadata.get('is_complete_sentence', False)
                    print(f"   {i+1}. {c.get('start', 0)}s (enhanced: {c.get('enhanced_score', 0):.3f}, final: {c.get('final_score', 0):.3f}, quality: {quality:.2f}, complete: {complete})")
            
            return ranked
            
        except Exception as e:
            print(f"❌ Temporal re-ranking error: {e}")
            return candidates
    
    def _classify_temporal_intent(self, query: str) -> str:
        """Classify query intent for temporal bias (early, neutral, late)"""
        query_lower = query.lower()
        
        # Early-preferring intents (tips, hacks, how-to, setup, beginners, introduction, what is)
        early_keywords = [
            'tips', 'hacks', 'how to', 'setup', 'beginners', 'introduction', 'what is',
            'guide', 'tutorial', 'steps', 'process', 'create', 'build', 'make',
            'suggest', 'recommend', 'advice', 'help', 'start', 'begin'
        ]
        
        # Late-preferring intents (summary, conclusion, final thoughts, recap, TL;DR)
        late_keywords = [
            'summary', 'conclusion', 'final thoughts', 'recap', 'tl;dr', 'tldr',
            'key takeaways', 'main points', 'overview', 'wrap up', 'ending',
            'final verdict', 'bottom line', 'in summary'
        ]
        
        # Check for early-preferring keywords
        if any(keyword in query_lower for keyword in early_keywords):
            return 'early'
        
        # Check for late-preferring keywords
        if any(keyword in query_lower for keyword in late_keywords):
            return 'late'
        
        # Default to neutral
        return 'neutral'
    
    def _validate_timestamps(self, candidate: Dict, question: str, question_analysis: Dict) -> bool:
        """Validate that the chunk's timestamps are relevant to the question"""
        try:
            # Get timestamps
            start = candidate.get('start', 0)
            end = candidate.get('end', 0)
            
            # Basic timestamp validation
            if start >= end or start < 0:
                print(f"❌ Invalid timestamp range: {start}-{end}")
                return False
            
            # Check if timestamps are reasonable (not too long or too short)
            duration = end - start
            if duration < 3:  # Too short (lowered from 5s)
                print(f"❌ Timestamp too short: {duration}s")
                return False
            if duration > 600:  # Too long (10 minutes, increased from 5)
                print(f"❌ Timestamp too long: {duration}s")
                return False
            
            # NEW: Penalize chunks that are too late in the video (likely end of topics)
            if start > 900:  # After 15 minutes
                print(f"⚠️ Timestamp too late in video: {start}s (likely end of topic)")
                # Don't fail validation, but warn - this helps avoid end-of-topic chunks
            
            # Check content relevance
            content = candidate.get('content', '').lower()
            question_lower = question.lower()
            
            # Extract key terms from question
            key_terms = []
            common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
            words = question_lower.split()
            key_terms = [word for word in words if len(word) > 3 and word not in common_words]
            
            # Check if key terms appear in content (optional validation)
            if key_terms:
                term_matches = sum(1 for term in key_terms if term in content)
                relevance_ratio = term_matches / len(key_terms)
                
                # Only warn if very low relevance, don't fail validation
                if relevance_ratio < 0.2:
                    print(f"⚠️ Low content relevance: {relevance_ratio:.2f} (continuing anyway)")
                    # Don't return False, just warn
            
            # Check intent-specific validation (made optional to avoid being too strict)
            intent_type = question_analysis.get('intent_type', '')
            if intent_type == 'what_is':
                # For definition questions, look for definition indicators (optional check)
                definition_words = ['definition', 'means', 'stands for', 'is a', 'refers to', 'mvp', 'minimum viable product']
                if not any(word in content for word in definition_words):
                    print(f"⚠️ No definition indicators found for 'what_is' question (continuing anyway)")
                    # Don't return False, just warn
            
            elif intent_type == 'how_to':
                # For how-to questions, look for procedural indicators (optional check)
                procedural_words = ['steps', 'process', 'how to', 'tutorial', 'guide', 'method']
                if not any(word in content for word in procedural_words):
                    print(f"⚠️ No procedural indicators found for 'how_to' question (continuing anyway)")
                    # Don't return False, just warn
            
            elif intent_type == 'view':
                # For example questions, look for example indicators (optional check)
                example_words = ['example', 'instance', 'case', 'scenario', 'for instance', 'such as']
                if not any(word in content for word in example_words):
                    print(f"⚠️ No example indicators found for 'view' question (continuing anyway)")
                    # Don't return False, just warn
            
            print(f"✅ Timestamp validation passed: {start}-{end}s")
            return True
            
        except Exception as e:
            print(f"❌ Timestamp validation error: {e}")
            return False
    
    def _intelligent_answer_selection(self, question: str, question_analysis: Dict, scored_candidates: Dict) -> Dict:
        """Intelligent answer selection with BEST CHUNK ONLY strategy"""
        try:
            video_candidates = scored_candidates['video']
            knowledge_candidates = scored_candidates['knowledge']
            
            # Get best candidates (single best, not combined) - use final_score if available
            best_video = video_candidates[0] if video_candidates and len(video_candidates) > 0 else None
            best_knowledge = knowledge_candidates[0] if knowledge_candidates and len(knowledge_candidates) > 0 else None
            
            # Use enhanced_score directly (no temporal re-ranking)
            video_score = best_video.get('enhanced_score', 0) if best_video else 0
            knowledge_score = best_knowledge.get('enhanced_score', 0) if best_knowledge else 0
            print(f"🔍 Best video score: {video_score:.3f}")
            print(f"🔍 Best knowledge score: {knowledge_score:.3f}")
            if best_video:
                print(f"🔍 Video URL: {best_video.get('video_url', 'EMPTY')}")
                print(f"🔍 Video URL (alt): {best_video.get('url', 'EMPTY')}")
            
            # Apply strict quality gates
            video_passes_gate = self._video_passes_quality_gate(best_video, question_analysis)
            knowledge_passes_gate = self._knowledge_passes_quality_gate(best_knowledge, question_analysis)
            
            # Simple timestamp validation (restored to basic version)
            video_timestamp_valid = True
            
            print(f"🔍 Video passes gate: {video_passes_gate}")
            print(f"🔍 Knowledge passes gate: {knowledge_passes_gate}")
            
            # NEW STRATEGY: Choose the SINGLE BEST chunk, don't combine
            # Priority: Video > Knowledge (if both pass gates)
            # But only if video is significantly better or question is video-specific
            
            # Check if question is video-specific
            is_video_specific = self._detect_video_intent(question)
            print(f"🔍 Video-specific question: {is_video_specific}")
            
            # Decision logic with BEST CHUNK ONLY strategy (restored to simple version)
            if video_passes_gate and is_video_specific:
                # Video-specific question and video passes gate
                return {
                    'decision': 'video_only',
                    'candidate': best_video,
                    'reason': 'video_specific_question'
                }
            
            if knowledge_passes_gate and not video_passes_gate:
                # Only knowledge passes gate
                return {
                    'decision': 'knowledge_only',
                    'candidate': best_knowledge,
                    'reason': 'video_failed_quality_gate'
                }
            
            if video_passes_gate and not knowledge_passes_gate:
                # Only video passes gate
                return {
                    'decision': 'video_only',
                    'candidate': best_video,
                    'reason': 'knowledge_failed_quality_gate'
                }
            
            if not knowledge_passes_gate and not video_passes_gate:
                # Neither passes gate - use fallback
                return {
                    'decision': 'fallback',
                    'candidate': best_knowledge or best_video,
                    'reason': 'both_failed_quality_gates'
                }
            
            # Both pass gates - choose the SINGLE BEST one (use enhanced_score directly)
            video_score = best_video.get('enhanced_score', 0) if best_video else 0
            knowledge_score = best_knowledge.get('enhanced_score', 0) if best_knowledge else 0
            
            # Choose the significantly better one (minimum 0.15 difference)
            if video_score >= knowledge_score + 0.15:
                return {
                    'decision': 'video_only',
                    'candidate': best_video,
                    'reason': 'video_significantly_better'
                }
            
            if knowledge_score >= video_score + 0.15:
                return {
                    'decision': 'knowledge_only',
                    'candidate': best_knowledge,
                    'reason': 'knowledge_significantly_better'
                }
            
            # Close scores - prefer knowledge for better answer quality (unless video-specific)
            if is_video_specific:
                return {
                    'decision': 'video_only',
                    'candidate': best_video,
                    'reason': 'video_specific_close_scores'
                }
            else:
                return {
                    'decision': 'knowledge_only',
                    'candidate': best_knowledge,
                    'reason': 'knowledge_preferred_close_scores'
                }
            
        except Exception as e:
            print(f"❌ Intelligent selection error: {e}")
            return {
                'decision': 'fallback',
                'candidate': None,
                'reason': 'selection_error'
            }
    
    def _video_passes_quality_gate(self, candidate: Optional[Dict], question_analysis: Dict) -> bool:
        """Enhanced quality gate for video candidates with improved validation"""
        if not candidate:
            return False
        
        # Enhanced score threshold
        if candidate['enhanced_score'] < self.VIDEO_MIN_RELEVANCE:
            print(f"❌ Video failed: enhanced score {candidate['enhanced_score']:.3f} < {self.VIDEO_MIN_RELEVANCE}")
            return False
        
        # Semantic similarity threshold
        if candidate['semantic_score'] < self.SEMANTIC_SIMILARITY_THRESHOLD:
            print(f"❌ Video failed: semantic score {candidate['semantic_score']:.3f} < {self.SEMANTIC_SIMILARITY_THRESHOLD}")
            return False
        
        # Intent alignment threshold (very low to allow more candidates)
        if candidate['intent_score'] < 0.0:
            print(f"❌ Video failed: intent score {candidate['intent_score']:.3f} < 0.0")
            return False
        
        # Video-specific requirements (relaxed - we have timestamps)
        if not candidate.get('start', 0) and not candidate.get('end', 0):
            print(f"❌ Video failed: no timestamps")
            return False
        
        # Enhanced timestamp validation
        start = candidate.get('start', 0)
        end = candidate.get('end', 0)
        duration = candidate.get('duration', 0)
        
        # Validate timestamp ranges (relaxed - duration can be 0)
        if not (0 <= start < end):
            print(f"❌ Video failed: invalid timestamps (start={start}, end={end})")
            return False
        
        # Validate timestamp duration (not too short, not too long)
        timestamp_duration = end - start
        if timestamp_duration < 5:  # At least 5 seconds
            print(f"❌ Video failed: timestamp too short ({timestamp_duration}s)")
            return False
        
        if timestamp_duration > 300:  # Not more than 5 minutes
            print(f"❌ Video failed: timestamp too long ({timestamp_duration}s)")
            return False
        
        # Check for meaningful content
        text_content = candidate.get('text', '')
        if len(text_content) < 50:
            print(f"❌ Video failed: insufficient content ({len(text_content)} chars)")
            return False
        
        print(f"✅ Video passes all enhanced quality gates")
        return True
    
    def _knowledge_passes_quality_gate(self, candidate: Optional[Dict], question_analysis: Dict) -> bool:
        """Enhanced quality gate for knowledge candidates with improved validation"""
        if not candidate:
            return False
        
        # Enhanced score threshold
        if candidate['enhanced_score'] < self.KNOWLEDGE_MIN_RELEVANCE:
            print(f"❌ Knowledge failed: enhanced score {candidate['enhanced_score']:.3f} < {self.KNOWLEDGE_MIN_RELEVANCE}")
            return False
        
        # Semantic similarity threshold
        if candidate['semantic_score'] < self.SEMANTIC_SIMILARITY_THRESHOLD:
            print(f"❌ Knowledge failed: semantic score {candidate['semantic_score']:.3f} < {self.SEMANTIC_SIMILARITY_THRESHOLD}")
            return False
        
        # Intent alignment threshold
        if candidate['intent_score'] < 0.5:
            print(f"❌ Knowledge failed: intent score {candidate['intent_score']:.3f} < 0.5")
            return False
        
        # Content quality requirements
        if not candidate.get('content_has_text', True):
            print(f"❌ Knowledge failed: no text content")
            return False
        
        # Enhanced text length and quality requirements
        text_content = candidate['text']
        if len(text_content) < 100:
            print(f"❌ Knowledge failed: text too short ({len(text_content)} chars)")
            return False
        
        # Check for meaningful content (not just metadata or timestamps)
        meaningful_words = [word for word in text_content.split() if len(word) > 3]
        if len(meaningful_words) < 10:
            print(f"❌ Knowledge failed: insufficient meaningful content ({len(meaningful_words)} words)")
            return False
        
        # Check for content structure (not just a list of timestamps)
        if text_content.count('[') > len(text_content) / 20:  # Too many timestamp-like brackets
            print(f"❌ Knowledge failed: too many timestamp-like brackets")
            return False
        
        # Check for title quality
        title = candidate.get('title', '')
        if not title or len(title) < 5:
            print(f"❌ Knowledge failed: poor title quality")
            return False
        
        print(f"✅ Knowledge passes all enhanced quality gates")
        return True
    
    def _quality_assurance_formatting(self, decision: Dict, question: str, question_analysis: Dict) -> Dict:
        """Quality assurance and final formatting"""
        try:
            if decision['decision'] == 'fallback':
                return self._safe_fallback()
            
            candidate = decision['candidate']
            if not candidate:
                return self._safe_fallback()
            
            # Prepare context for LLM
            if decision['decision'] == 'knowledge_only':
                context = self._prepare_knowledge_context(candidate)
                answer_type = 'knowledge'
            elif decision['decision'] == 'video_only':
                context = self._prepare_video_context(candidate)
                answer_type = 'video'
            else:
                return self._safe_fallback()
            
            # Enhanced LLM formatting with intent awareness
            formatted_answer = self._format_with_intent_awareness(question, context, answer_type, question_analysis)
            
            # Calculate enhanced confidence score
            confidence_score = self._calculate_enhanced_confidence_score(candidate, question)
            
            # Build response payload
            response = {
                'success': True,
                'answer': formatted_answer,
                'sources': self._build_sources(decision),
                'total_sources': len(self._build_sources(decision)),
                'search_score': candidate['enhanced_score'],
                'confidence': confidence_score,
                'confidence_score': confidence_score,
                'content_types_found': [answer_type],
                'difficulty_level': candidate.get('difficulty_level', 'intermediate'),
                'estimated_time': '2-3 minutes'
            }
            
            # Add video fields only if appropriate
            if answer_type == 'video' and decision.get('candidate'):
                video_candidate = decision['candidate']
                response.update({
                    'start': video_candidate.get('start', 0),
                    'end': video_candidate.get('end', 0),
                    'video_url': video_candidate.get('video_url', '') or video_candidate.get('url', ''),
                    'timestamp': self._format_timestamp(video_candidate.get('start', 0), video_candidate.get('end', 0)),
                    'formatted_timestamp': self._format_timestamp(video_candidate.get('start', 0), video_candidate.get('end', 0))
                })
            else:
                response.update({
                    'start': 0,
                    'end': 0,
                    'video_url': None
                })
            
            return response
            
        except Exception as e:
            print(f"❌ Quality assurance error: {e}")
            return self._safe_fallback()
    
    def _prepare_knowledge_context(self, candidate: Dict) -> str:
        """Prepare knowledge context for LLM with better structure"""
        try:
            # Clean and structure the content
            title = candidate.get('title', 'Untitled')
            content = candidate.get('text', '')
            url = candidate.get('url', '')
            
            # Clean up the content
            content = self._clean_content(content)
            
            # Structure the context
            context = f"""Title: {title}

Content: {content}"""
            
            if url:
                context += f"\n\nSource URL: {url}"
            
            return context
            
        except Exception as e:
            print(f"❌ Error preparing knowledge context: {e}")
            return f"Title: {candidate.get('title', 'Untitled')}\nContent: {candidate.get('text', '')}"
    
    def _prepare_video_context(self, candidate: Dict) -> str:
        """Prepare video context for LLM with unified chunking metadata"""
        try:
            # Clean and structure the content
            title = candidate.get('title', 'Untitled')
            content = candidate.get('text', '')
            video_url = candidate.get('video_url', '') or candidate.get('url', '')
            start = candidate.get('start', 0)
            end = candidate.get('end', 0)
            
            # Get unified chunking metadata
            metadata = candidate.get('metadata', {})
            content_quality = metadata.get('content_quality', 0)
            is_complete_sentence = metadata.get('is_complete_sentence', False)
            word_count = metadata.get('word_count', 0)
            topic_keywords = metadata.get('topic_keywords', [])
            has_question = metadata.get('has_question', False)
            has_instruction = metadata.get('has_instruction', False)
            has_explanation = metadata.get('has_explanation', False)
            
            # Clean up the content
            content = self._clean_content(content)
            
            # Format timestamp info
            timestamp_info = self._format_timestamp(start, end)
            
            # Structure the context with enhanced metadata
            context = f"""Title: {title}

Content: {content}

Video Information:
- Timestamp: {timestamp_info}
- Duration: {end - start:.1f} seconds
- Word Count: {word_count} words"""
            
            # Add quality indicators
            if content_quality > 0.7:
                context += "\n- Quality: High-quality content"
            elif content_quality > 0.4:
                context += "\n- Quality: Good content"
            
            if is_complete_sentence:
                context += "\n- Structure: Complete sentences"
            
            if topic_keywords:
                context += f"\n- Keywords: {', '.join(topic_keywords[:3])}"
            
            # Add content type indicators
            content_types = []
            if has_question:
                content_types.append("question")
            if has_instruction:
                content_types.append("instruction")
            if has_explanation:
                content_types.append("explanation")
            
            if content_types:
                context += f"\n- Content Type: {', '.join(content_types)}"
            
            if video_url:
                context += f"\n- Video URL: {video_url}"
            
            return context
            
        except Exception as e:
            print(f"❌ Error preparing video context: {e}")
            return f"Title: {candidate.get('title', 'Untitled')}\nContent: {candidate.get('text', '')}"
    
    def _clean_content(self, content: str) -> str:
        """Clean and structure content for better LLM processing"""
        try:
            if not content:
                return ""
            
            # Remove excessive whitespace
            content = ' '.join(content.split())
            
            # Remove timestamp-like patterns that are just noise
            import re
            # Remove patterns like [00:01] or [1:23] that appear frequently
            content = re.sub(r'\[\d{1,2}:\d{2}\]', '', content)
            
            # Remove excessive punctuation
            content = re.sub(r'[.]{3,}', '...', content)
            
            # Clean up sentence boundaries
            content = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', content)
            
            # Limit length to prevent overwhelming the LLM
            if len(content) > 2000:
                content = content[:2000] + "..."
            
            return content.strip()
            
        except Exception as e:
            print(f"❌ Error cleaning content: {e}")
            return content
    
    def _format_with_intent_awareness(self, question: str, context: str, answer_type: str, question_analysis: Dict) -> str:
        """Format answer with intent awareness and conversational tone"""
        try:
            intent_type = question_analysis.get('intent_type', '')
            main_action = question_analysis.get('main_action', '')
            
            # Enhanced prompt for better formatting
            if answer_type == 'knowledge':
                if intent_type == 'create':
                    prompt = f"""You are a helpful AI assistant. The user wants to CREATE something. Based on the following knowledge base content, provide a clear, conversational, and well-formatted response.

Question: {question}
Main Action: {main_action}

Knowledge Base Content:
{context}

Instructions:
1. Write in a conversational, helpful tone
2. Structure your response with clear sections
3. Use bullet points or numbered steps where appropriate
4. Make it easy to read and follow
5. Don't just copy the content - synthesize and explain it clearly
6. Start with a brief introduction, then provide the steps/instructions
7. End with a helpful tip or summary

Format your response like a helpful bot would:"""
                elif intent_type == 'how_to':
                    prompt = f"""You are a helpful AI assistant. The user is asking HOW TO do something. Based on the following knowledge base content, provide a clear, step-by-step guide.

Question: {question}
Main Action: {main_action}

Knowledge Base Content:
{context}

Instructions:
1. Write in a conversational, helpful tone
2. Structure your response with clear sections
3. Use numbered steps for procedures
4. Make it easy to read and follow
5. Don't just copy the content - synthesize and explain it clearly
6. Start with a brief introduction, then provide the steps
7. End with a helpful tip or summary

Format your response like a helpful bot would:"""
                else:
                    prompt = f"""You are a helpful AI assistant. Based on the following knowledge base content, provide a clear, conversational answer to the user's question.

Question: {question}

Knowledge Base Content:
{context}

Instructions:
1. Write in a conversational, helpful tone
2. Structure your response with clear sections
3. Use bullet points or formatting where appropriate
4. Make it easy to read and follow
5. Don't just copy the content - synthesize and explain it clearly
6. Start with a brief introduction, then provide the main information
7. End with a helpful summary or next steps

Format your response like a helpful bot would:"""
            
            elif answer_type == 'video':
                if intent_type == 'create':
                    prompt = f"""You are a knowledgeable sales manager. The user wants to CREATE something. Give them a short, actionable response under 700 characters.

Question: {question}
Main Action: {main_action}

Information:
{context}

Instructions:
1. Keep your answer UNDER 700 CHARACTERS
2. Focus on the key steps they need to take
3. Write in a confident, professional tone
4. Don't reference "video" or "transcript" - just explain directly
5. Be concise and actionable
6. One or two sentences maximum

Answer:"""
                elif intent_type == 'how_to':
                    prompt = f"""You are a helpful AI assistant. The user is asking HOW TO do something. Based on the following video transcript, provide a clear, step-by-step guide.

Question: {question}
Main Action: {main_action}

Video Transcript:
{context}

Instructions:
1. Write in a conversational, helpful tone
2. Structure your response with clear sections
3. Use numbered steps for procedures
4. Make it easy to read and follow
5. Don't just copy the transcript - synthesize and explain it clearly
6. Start with a brief introduction, then provide the steps
7. End with a helpful tip or summary
8. Mention that this information comes from a video

Format your response like a helpful bot would:"""
                else:
                    prompt = f"""You are a knowledgeable sales manager. Answer the user's question with a well-structured, detailed response.

Question: {question}

Information:
{context}

Instructions:
1. Write a comprehensive answer (3-5 sentences, 600-800 characters)
2. Write in a confident, professional tone
3. Don't reference "video" or "transcript" - just explain directly
4. Structure your answer with clear points
5. Be conversational and informative
6. Provide actionable insights
7. Keep your answer between 600-800 characters for optimal readability

Answer:"""
            
            else:
                return context  # Fallback to raw content
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800  # Increased for better formatting
            )
            
            formatted_answer = response.choices[0].message.content.strip()
            
            # Post-process to ensure good formatting
            formatted_answer = self._post_process_answer(formatted_answer, answer_type)
            
            return formatted_answer
            
        except Exception as e:
            print(f"❌ Intent-aware formatting error: {e}")
            return self._create_fallback_formatted_answer(context, answer_type)
    
    def _post_process_answer(self, answer: str, answer_type: str) -> str:
        """Post-process the answer to ensure good formatting"""
        try:
            # Clean up the answer
            answer = answer.strip()
            
            # Ensure it starts with a proper introduction
            if not answer.startswith(('I', 'Based on', 'Here', 'To', 'You', 'An', 'A', 'The')):
                if answer_type == 'video':
                    answer = "Let me explain: " + answer.lower()
                else:
                    answer = "Here's what I can tell you: " + answer.lower()
            
            # Ensure proper capitalization
            answer = answer[0].upper() + answer[1:] if len(answer) > 1 else answer
            
            # Add source attribution if not present (removed video references)
            if answer_type == 'knowledge' and 'knowledge' not in answer.lower():
                answer += "\n\n*This information comes from your knowledge base.*"
            
            return answer
            
        except Exception as e:
            print(f"❌ Post-processing error: {e}")
            return answer
    
    def _create_fallback_formatted_answer(self, context: str, answer_type: str) -> str:
        """Create a fallback formatted answer when LLM processing fails"""
        try:
            # Extract key information from context
            lines = context.split('\n')
            title = ""
            content = ""
            
            for line in lines:
                if line.startswith('Title:'):
                    title = line.replace('Title:', '').strip()
                elif line.startswith('Content:'):
                    content = line.replace('Content:', '').strip()
            
            # Create a basic formatted response
            if answer_type == 'video':
                response = f"Based on the video content about '{title}', here's what I found:\n\n"
            else:
                response = f"Based on the knowledge base content about '{title}', here's what I found:\n\n"
            
            # Add the content with basic formatting
            if content:
                # Limit content length and add basic structure
                if len(content) > 500:
                    content = content[:500] + "..."
                
                response += content
            
            return response
            
        except Exception as e:
            print(f"❌ Fallback formatting error: {e}")
            return "I found some relevant information, but I'm having trouble formatting it properly. Please try rephrasing your question."
    
    def _build_sources(self, decision: Dict) -> List[Dict]:
        """Build sources list for response"""
        sources = []
        
        candidate = decision['candidate']
        if candidate:
            source_type = 'knowledge' if candidate['index_type'] == 'knowledge' else 'video'
            sources.append({
                'type': source_type,
                'url': candidate.get('url', candidate.get('video_url', '')),
                'title': candidate['title']
            })
        
        return sources
    
    def _format_timestamp(self, start: float, end: float) -> str:
        """Format timestamp for display"""
        if start == 0 and end == 0:
            return ""
        
        start_str = self._seconds_to_timestamp(start)
        end_str = self._seconds_to_timestamp(end)
        
        return f"{start_str} - {end_str}"
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS or HH:MM:SS format"""
        if seconds < 60:
            return f"{int(seconds):02d}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}:{minutes:02d}:{secs:02d}"
    
    def _get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """Get embedding with caching"""
        # Normalize text for caching
        cache_key = f"{model}:{text.lower().strip()}"
        
        # Check cache
        if cache_key in self._embedding_cache:
            # Check TTL
            if time.time() - self._cache_timestamps[cache_key] < self.CACHE_TTL:
                self._embedding_cache.move_to_end(cache_key)
                self._cache_hits += 1
                return self._embedding_cache[cache_key]
            else:
                # Expired, remove from cache
                del self._embedding_cache[cache_key]
                del self._cache_timestamps[cache_key]
        
        # Generate new embedding
        try:
            self._cache_misses += 1
            response = self.openai_client.embeddings.create(
                model=model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Add to cache
            self._embedding_cache[cache_key] = embedding
            self._cache_timestamps[cache_key] = time.time()
            
            # Remove oldest if cache is full
            if len(self._embedding_cache) > self.CACHE_SIZE:
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
                del self._cache_timestamps[oldest_key]
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return []
    
    def _log_enhanced_observability(self, question: str, question_analysis: Dict, candidates: Dict, scored_candidates: Dict, decision: Dict, latency: int):
        """Enhanced observability logging"""
        try:
            video_candidates = scored_candidates['video']
            knowledge_candidates = scored_candidates['knowledge']
            
            best_video = video_candidates[0] if video_candidates else None
            best_knowledge = knowledge_candidates[0] if knowledge_candidates else None
            
            # Extract metrics
            kn_hits = len(candidates['knowledge'])
            vid_hits = len(candidates['video'])
            
            kn_score = best_knowledge.get('enhanced_score', 0) if best_knowledge else 0
            vid_score = best_video.get('enhanced_score', 0) if best_video else 0
            
            kn_semantic = best_knowledge.get('semantic_score', 0) if best_knowledge else 0
            vid_semantic = best_video.get('semantic_score', 0) if best_video else 0
            
            kn_intent = best_knowledge.get('intent_score', 0) if best_knowledge else 0
            vid_intent = best_video.get('intent_score', 0) if best_video else 0
            
            # Log comprehensive observability
            cache_hit_rate = (self._cache_hits / (self._cache_hits + self._cache_misses)) * 100 if (self._cache_hits + self._cache_misses) > 0 else 0
            print(f"📊 [Enhanced QA] q_ms={latency}")
            print(f"    intent={question_analysis.get('intent_type', 'unknown')}")
            print(f"    kn: hits={kn_hits} S={kn_score:.3f} sem={kn_semantic:.3f} intent={kn_intent:.3f}")
            print(f"    vid: hits={vid_hits} S={vid_score:.3f} sem={vid_semantic:.3f} intent={vid_intent:.3f}")
            print(f"    decision={decision['decision']} reason={decision['reason']}")
            print(f"    cache: hit_rate={cache_hit_rate:.1f}% hits={self._cache_hits} misses={self._cache_misses}")
            
        except Exception as e:
            print(f"❌ Enhanced observability logging error: {e}")
    
    def _calculate_enhanced_confidence_score(self, candidate: Dict, question: str) -> float:
        """Calculate enhanced confidence score with unified chunking metadata integration"""
        try:
            # Base confidence from enhanced score
            base_confidence = candidate.get('enhanced_score', 0)
            
            # Get unified chunking metadata for enhanced scoring
            metadata = candidate.get('metadata', {})
            content_quality = metadata.get('content_quality', 0.5)
            is_complete_sentence = metadata.get('is_complete_sentence', False)
            word_count = metadata.get('word_count', 0)
            topic_keywords = metadata.get('topic_keywords', [])
            has_question = metadata.get('has_question', False)
            has_instruction = metadata.get('has_instruction', False)
            has_explanation = metadata.get('has_explanation', False)
            
            # Enhanced content quality boost using unified metadata
            if content_quality > 0.8:
                base_confidence += 0.15  # 15% boost for high quality
            elif content_quality > 0.6:
                base_confidence += 0.10  # 10% boost for good quality
            elif content_quality > 0.4:
                base_confidence += 0.05  # 5% boost for decent quality
            
            # Complete sentence boost
            if is_complete_sentence:
                base_confidence += 0.08  # 8% boost for complete sentences
            
            # Word count quality boost
            if word_count > 50:
                base_confidence += 0.05
            if word_count > 100:
                base_confidence += 0.05
            if word_count > 200:
                base_confidence += 0.03
            
            # Content type boost
            content_type_score = 0
            if has_question:
                content_type_score += 0.03
            if has_instruction:
                content_type_score += 0.05
            if has_explanation:
                content_type_score += 0.05
            
            base_confidence += content_type_score
            
            # Topic keywords relevance boost
            if topic_keywords:
                question_lower = question.lower()
                keyword_matches = sum(1 for keyword in topic_keywords if keyword.lower() in question_lower)
                if keyword_matches > 0:
                    base_confidence += min(keyword_matches * 0.02, 0.08)
            
            # Metadata completeness boost
            metadata_score = 0
            if candidate.get('title'): metadata_score += 0.05
            if candidate.get('summary'): metadata_score += 0.05
            if candidate.get('url') or candidate.get('video_url'): metadata_score += 0.05
            if candidate.get('created_at'): metadata_score += 0.05
            
            base_confidence += metadata_score
            
            # Video content boost
            if candidate.get('index_type') == 'video' and candidate.get('has_timestamps', False):
                base_confidence += 0.1
                # Additional boost for good timestamp range
                start = candidate.get('start', 0)
                end = candidate.get('end', 0)
                if 5 <= (end - start) <= 300:  # Good timestamp duration
                    base_confidence += 0.05
            
            # Knowledge content boost
            if candidate.get('index_type') == 'knowledge':
                if candidate.get('has_steps', False):
                    base_confidence += 0.1
                if candidate.get('content_has_text', True):
                    base_confidence += 0.05
            
            # Question relevance boost
            question_words = set(question.lower().split())
            candidate_text = f"{candidate.get('title', '')} {candidate.get('text', '')}".lower()
            candidate_words = set(candidate_text.split())
            common_words = question_words.intersection(candidate_words)
            if len(common_words) > 0:
                base_confidence += min(len(common_words) * 0.02, 0.1)
            
            return min(base_confidence, 1.0)
            
        except Exception as e:
            print(f"❌ Error calculating enhanced confidence score: {e}")
            return 0.5
    
    def _detect_video_intent(self, question: str) -> bool:
        """Detect if question is video-specific"""
        video_keywords = ['video', 'show me', 'watch', 'play', 'timestamp', 'demo', 'recording']
        return any(keyword in question.lower() for keyword in video_keywords)
    
    def _filter_similar_chunks(self, chunks: List[Dict], question: str) -> List[Dict]:
        """Filter out similar chunks to avoid repetitive content"""
        try:
            if len(chunks) <= 1:
                return chunks
            
            # Keep only the best chunk (highest enhanced score)
            # This ensures we don't combine similar chunks
            best_chunk = max(chunks, key=lambda x: x.get('enhanced_score', 0))
            
            # Additional filtering: remove chunks that are too similar to the best one
            filtered_chunks = [best_chunk]
            
            for chunk in chunks:
                if chunk == best_chunk:
                    continue
                
                # Check similarity with best chunk
                similarity = self._calculate_chunk_similarity(best_chunk, chunk)
                
                # Only keep if significantly different (less than 80% similar)
                if similarity < 0.8:
                    filtered_chunks.append(chunk)
                    break  # Only keep one additional chunk max
            
            print(f"🔍 Filtered {len(chunks)} chunks down to {len(filtered_chunks)} unique chunks")
            return filtered_chunks[:1]  # Return only the best chunk
            
        except Exception as e:
            print(f"❌ Error filtering similar chunks: {e}")
            return chunks[:1] if chunks else []  # Return only first chunk as fallback
    
    def _calculate_chunk_similarity(self, chunk1: Dict, chunk2: Dict) -> float:
        """Calculate similarity between two chunks"""
        try:
            # Simple similarity based on text content
            text1 = chunk1.get('text', '').lower()
            text2 = chunk2.get('text', '').lower()
            
            if not text1 or not text2:
                return 0.0
            
            # Calculate word overlap
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            similarity = len(intersection) / len(union) if union else 0.0
            
            return similarity
            
        except Exception as e:
            print(f"❌ Error calculating chunk similarity: {e}")
            return 0.0
    
    def _safe_fallback(self) -> Dict:
        """Enhanced safe fallback response"""
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
            'difficulty_level': 'unknown',
            'estimated_time': 'unknown',
            'confidence_score': 0.0,
            'fallback_reason': 'no_relevant_content_found'
        }


# Global instance for singleton pattern
_enhanced_semantic_qa_instance = None

def initialize_enhanced_semantic_qa():
    """Initialize the enhanced semantic QA system"""
    global _enhanced_semantic_qa_instance
    try:
        _enhanced_semantic_qa_instance = EnhancedSemanticQA()
        print("✅ Enhanced Semantic QA system initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced Semantic QA: {e}")
        return False

def get_enhanced_semantic_qa():
    """Get the enhanced semantic QA system instance"""
    global _enhanced_semantic_qa_instance
    if _enhanced_semantic_qa_instance is None:
        print("⚠️ Enhanced Semantic QA not initialized, initializing now...")
        initialize_enhanced_semantic_qa()
    return _enhanced_semantic_qa_instance
