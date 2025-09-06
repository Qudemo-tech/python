#!/usr/bin/env python3
"""
Context-First QA System
Production-safe, semantic-focused Q&A with two-stage retrieval and neural re-ranking
"""

import os
import re
import time
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import openai
from pinecone import Pinecone
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContextFirstQA:
    """Context-first QA system with semantic retrieval and neural re-ranking"""
    
    # Production thresholds
    KNOWLEDGE_MIN = 0.20
    VIDEO_MIN = 0.20
    DOMINANCE_DELTA = 0.06
    COMPATIBILITY_THRESHOLD = 0.55
    
    # Retrieval parameters
    TOP_K_RECALL = 30
    TOP_K_RERANK = 8
    SCORE_THRESHOLD = 0.05
    
    # Performance settings
    BATCH_SIZE = 100
    CACHE_SIZE = 100
    CACHE_TTL = 24 * 60 * 60  # 24 hours
    
    def __init__(self):
        """Initialize context-first QA system"""
        # Pinecone indexes
        self.indexes = {
            'video': 'qudemo-video-index',
            'knowledge': 'qudemo-knowledge-index',
            'legacy': 'qudemo-index'
        }
        
        # Embedding cache
        self._embedding_cache = OrderedDict()
        self._cache_timestamps = {}
        
        # OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Pinecone client
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
    def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Main QA entry point with context-first approach"""
        try:
            start_time = time.time()
            print(f"🧠 Context-First QA: {question}")
            
            # Stage 1: Two-stage retrieval
            candidates = self._two_stage_retrieval(question, company_name, qudemo_id)
            
            # Stage 2: Neural re-ranking
            reranked = self._neural_rerank(question, candidates)
            
            # Stage 3: Context-aware decision
            final_answer = self._context_aware_decision(question, reranked)
            
            # Stage 4: Fast answer building
            formatted_answer = self._build_final_answer(final_answer, question)
            
            # Observability logging
            latency = int((time.time() - start_time) * 1000)
            self._log_observability(question, candidates, reranked, final_answer, latency)
            
            return formatted_answer
            
        except Exception as e:
            print(f"❌ Context-First QA Error: {e}")
            return self._safe_fallback()
    
    def _two_stage_retrieval(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Stage A: Broad vector recall from both indexes"""
        try:
            # Get question embedding (cached)
            question_embedding = self._get_embedding(question, model="text-embedding-3-small")
            
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Parallel queries to both indexes
            with ThreadPoolExecutor(max_workers=2) as executor:
                video_future = executor.submit(
                    self._query_index, 
                    'video', namespace, question_embedding
                )
                knowledge_future = executor.submit(
                    self._query_index, 
                    'knowledge', namespace, question_embedding
                )
                
                video_results = video_future.result()
                knowledge_results = knowledge_future.result()
            
            return {
                'video': video_results,
                'knowledge': knowledge_results
            }
            
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return {'video': [], 'knowledge': []}
    
    def _query_index(self, index_type: str, namespace: str, embedding: List[float]) -> List[Dict]:
        """Query specific index with appropriate filters"""
        try:
            # Get index
            if index_type in self.indexes:
                index = self.pc.Index(self.indexes[index_type])
            else:
                return []
            
            # Set filters based on index type
            if index_type == 'knowledge':
                filter_query = {
                    "source_type": "web_scraping",
                    "content_has_text": True
                }
            elif index_type == 'video':
                filter_query = {
                    "source_type": "video_transcript"
                }
            else:
                filter_query = {}
            
            # Query with broad parameters
            results = index.query(
                vector=embedding,
                top_k=self.TOP_K_RECALL,
                include_metadata=True,
                namespace=namespace,
                filter=filter_query,
                score_threshold=self.SCORE_THRESHOLD
            )
            
            # Convert to candidate format
            candidates = []
            for match in results.matches:
                candidate = self._build_candidate_view(match, index_type)
                if candidate:
                    candidates.append(candidate)
            
            print(f"✅ {index_type}: {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            print(f"❌ {index_type} query error: {e}")
            return []
    
    def _build_candidate_view(self, match, index_type: str) -> Optional[Dict]:
        """Build candidate view for re-ranking"""
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
                    'seekable': metadata.get('seekable', False),
                    'has_timestamps': metadata.get('has_timestamps', False),
                    'start': metadata.get('start', 0),
                    'end': metadata.get('end', 0),
                    'duration': metadata.get('duration', 0),
                    'video_url': metadata.get('video_url', '')
                })
            elif index_type == 'knowledge':
                candidate.update({
                    'content_has_text': metadata.get('content_has_text', True),
                    'url': metadata.get('url', ''),
                    'has_steps': metadata.get('has_steps', False),
                    'difficulty_level': metadata.get('difficulty_level', 'intermediate')
                })
            
            # Truncate text for re-ranking (first/last 300 chars)
            if len(text) > 600:
                candidate['text_truncated'] = text[:300] + "..." + text[-300:]
            else:
                candidate['text_truncated'] = text
            
            return candidate
            
        except Exception as e:
            print(f"❌ Candidate build error: {e}")
            return None
    
    def _neural_rerank(self, question: str, candidates: Dict) -> Dict:
        """Stage B: Neural re-ranking with semantic cross-check"""
        try:
            all_candidates = candidates['video'] + candidates['knowledge']
            
            if not all_candidates:
                return {'video': [], 'knowledge': []}
            
            # Limit to top candidates for re-ranking
            all_candidates = all_candidates[:20]  # Reasonable limit
            
            # Get embeddings for semantic comparison
            question_embedding = self._get_embedding(question, model="text-embedding-3-small")
            
            # Calculate semantic scores
            semantic_scores = []
            for candidate in all_candidates:
                # Combine title + summary + text for embedding
                candidate_text = f"{candidate['title']} {candidate['summary']} {candidate['text_truncated']}"
                candidate_embedding = self._get_embedding(candidate_text, model="text-embedding-3-small")
                
                # Cosine similarity
                similarity = cosine_similarity([question_embedding], [candidate_embedding])[0][0]
                semantic_scores.append(similarity)
            
            # Get entailment scores (batched)
            entailment_scores = self._batch_entailment_check(question, all_candidates)
            
            # Calculate final scores
            for i, candidate in enumerate(all_candidates):
                semantic_score = semantic_scores[i]
                entailment_score = entailment_scores[i]
                
                # Structure priors (small boosts)
                structure_prior = self._calculate_structure_prior(question, candidate)
                
                # Final score
                final_score = (
                    0.65 * semantic_score +
                    0.25 * entailment_score +
                    0.10 * structure_prior
                )
                
                candidate['final_score'] = final_score
                candidate['semantic_score'] = semantic_score
                candidate['entailment_score'] = entailment_score
                candidate['structure_prior'] = structure_prior
            
            # Sort by final score
            all_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            # MMR diversification on top 8
            diversified = self._mmr_diversify(all_candidates[:self.TOP_K_RERANK])
            
            # Separate back into video/knowledge
            video_candidates = [c for c in diversified if c['index_type'] == 'video']
            knowledge_candidates = [c for c in diversified if c['index_type'] == 'knowledge']
            
            return {
                'video': video_candidates,
                'knowledge': knowledge_candidates
            }
            
        except Exception as e:
            print(f"❌ Re-ranking error: {e}")
            return candidates
    
    def _batch_entailment_check(self, question: str, candidates: List[Dict]) -> List[float]:
        """Batched entailment check using small LLM"""
        try:
            if not candidates:
                return []
            
            # Build batch prompt
            candidate_texts = []
            for candidate in candidates:
                text = f"Title: {candidate['title']}\nContent: {candidate['text_truncated']}"
                candidate_texts.append(text)
            
            # Create batch prompt
            prompt = f"""Question: {question}

For each passage below, rate how directly it helps answer the question (0.0 = not helpful, 1.0 = directly answers).

Passages:
"""
            for i, text in enumerate(candidate_texts):
                prompt += f"\n{i+1}. {text}\n"
            
            prompt += "\nRespond with JSON array of scores: [0.8, 0.3, 0.9, ...]"
            
            # Call small model for entailment
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse JSON response
            result = response.choices[0].message.content.strip()
            try:
                scores = json.loads(result)
                if isinstance(scores, list) and len(scores) == len(candidates):
                    return scores
            except:
                pass
            
            # Fallback: return semantic scores as entailment
            return [0.5] * len(candidates)
            
        except Exception as e:
            print(f"❌ Entailment check error: {e}")
            return [0.5] * len(candidates)
    
    def _calculate_structure_prior(self, question: str, candidate: Dict) -> float:
        """Calculate structure prior (small boosts)"""
        try:
            prior = 0.0
            
            # Check for procedural intent
            is_procedural = self._is_procedural_question(question)
            
            # Steps boost
            if candidate.get('has_steps', False) and is_procedural:
                prior += 0.05
            
            # Video seekability boost
            if candidate['index_type'] == 'video':
                if candidate.get('seekable', False) and candidate.get('has_timestamps', False):
                    prior += 0.05
            
            return min(prior, 0.10)  # Cap at 0.10
            
        except Exception as e:
            print(f"❌ Structure prior error: {e}")
            return 0.0
    
    def _is_procedural_question(self, question: str) -> bool:
        """Quick procedural intent detection"""
        procedural_indicators = [
            'how to', 'how do', 'how can', 'how should',
            'step', 'steps', 'process', 'procedure',
            'tutorial', 'guide', 'walkthrough'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in procedural_indicators)
    
    def _mmr_diversify(self, candidates: List[Dict]) -> List[Dict]:
        """MMR diversification to avoid near-duplicates"""
        try:
            if len(candidates) <= 1:
                return candidates
            
            diversified = [candidates[0]]  # Start with best candidate
            remaining = candidates[1:]
            
            while remaining and len(diversified) < self.TOP_K_RERANK:
                best_candidate = None
                best_score = -1
                best_index = -1
                
                for i, candidate in enumerate(remaining):
                    # Calculate MMR score
                    relevance = candidate['final_score']
                    
                    # Calculate max similarity to already selected
                    max_similarity = 0
                    for selected in diversified:
                        # Simple similarity based on title overlap
                        similarity = self._calculate_similarity(candidate['title'], selected['title'])
                        max_similarity = max(max_similarity, similarity)
                    
                    # MMR score (lambda = 0.7 for diversity)
                    mmr_score = 0.7 * relevance - 0.3 * max_similarity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_candidate = candidate
                        best_index = i
                
                if best_candidate:
                    diversified.append(best_candidate)
                    remaining.pop(best_index)
                else:
                    break
            
            return diversified
            
        except Exception as e:
            print(f"❌ MMR diversification error: {e}")
            return candidates
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity for MMR"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _context_aware_decision(self, question: str, reranked: Dict) -> Dict:
        """Context-aware decision policy with soft gating"""
        try:
            video_candidates = reranked['video']
            knowledge_candidates = reranked['knowledge']
            
            # Get best candidates
            best_video = video_candidates[0] if video_candidates else None
            best_knowledge = knowledge_candidates[0] if knowledge_candidates else None
            
            # Apply gates
            video_valid = self._is_video_valid(best_video)
            knowledge_valid = self._is_knowledge_valid(best_knowledge)
            
            print(f"🔍 Gates: video_valid={video_valid}, knowledge_valid={knowledge_valid}")
            
            # Decision logic
            if knowledge_valid and not video_valid:
                return {
                    'decision': 'knowledge_only',
                    'candidate': best_knowledge,
                    'reason': 'video_failed_gates'
                }
            
            if video_valid and not knowledge_valid:
                return {
                    'decision': 'video_only',
                    'candidate': best_video,
                    'reason': 'knowledge_failed_gates'
                }
            
            if not knowledge_valid and not video_valid:
                return {
                    'decision': 'fallback',
                    'candidate': best_knowledge or best_video,
                    'reason': 'both_failed_gates'
                }
            
            # Both valid - check dominance
            video_score = best_video['final_score']
            knowledge_score = best_knowledge['final_score']
            
            if video_score >= knowledge_score + self.DOMINANCE_DELTA:
                return {
                    'decision': 'video_only',
                    'candidate': best_video,
                    'reason': 'video_dominates'
                }
            
            if knowledge_score >= video_score + self.DOMINANCE_DELTA:
                return {
                    'decision': 'knowledge_only',
                    'candidate': best_knowledge,
                    'reason': 'knowledge_dominates'
                }
            
            # Close scores - check compatibility
            compatibility = self._calculate_compatibility(best_knowledge, best_video)
            
            if compatibility >= self.COMPATIBILITY_THRESHOLD:
                return {
                    'decision': 'combine',
                    'candidate': best_knowledge,
                    'video_candidate': best_video,
                    'compatibility': compatibility,
                    'reason': 'high_compatibility'
                }
            else:
                return {
                    'decision': 'knowledge_only',
                    'candidate': best_knowledge,
                    'reason': 'low_compatibility'
                }
            
        except Exception as e:
            print(f"❌ Decision error: {e}")
            return {
                'decision': 'fallback',
                'candidate': None,
                'reason': 'decision_error'
            }
    
    def _is_video_valid(self, candidate: Optional[Dict]) -> bool:
        """Check if video candidate is valid and eligible to play"""
        if not candidate:
            return False
        
        # Check score threshold
        if candidate['final_score'] < self.VIDEO_MIN:
            return False
        
        # Check seekability and timestamps
        if not candidate.get('seekable', False) or not candidate.get('has_timestamps', False):
            return False
        
        # Validate timestamps
        start = candidate.get('start', 0)
        end = candidate.get('end', 0)
        duration = candidate.get('duration', 0)
        
        if not (0 <= start < end <= duration):
            return False
        
        return True
    
    def _is_knowledge_valid(self, candidate: Optional[Dict]) -> bool:
        """Check if knowledge candidate is valid"""
        if not candidate:
            return False
        
        # Check score threshold
        if candidate['final_score'] < self.KNOWLEDGE_MIN:
            return False
        
        # Check content quality
        if not candidate.get('content_has_text', True):
            return False
        
        return True
    
    def _calculate_compatibility(self, knowledge_candidate: Dict, video_candidate: Dict) -> float:
        """Calculate chunk-to-chunk compatibility"""
        try:
            # Get embeddings for both candidates
            kn_embedding = self._get_embedding(knowledge_candidate['text'], model="text-embedding-3-small")
            vid_embedding = self._get_embedding(video_candidate['text'], model="text-embedding-3-small")
            
            # Calculate cosine similarity
            similarity = cosine_similarity([kn_embedding], [vid_embedding])[0][0]
            
            return similarity
            
        except Exception as e:
            print(f"❌ Compatibility calculation error: {e}")
            return 0.0
    
    def _build_final_answer(self, decision: Dict, question: str) -> Dict:
        """Build final answer with single LLM pass"""
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
            elif decision['decision'] == 'combine':
                context = self._prepare_combined_context(candidate, decision['video_candidate'])
                answer_type = 'combined'
            else:
                return self._safe_fallback()
            
            # Single LLM pass for formatting
            formatted_answer = self._format_with_llm(question, context, answer_type)
            
            # Build response payload
            response = {
                'success': True,
                'answer': formatted_answer,
                'sources': self._build_sources(decision),
                'total_sources': len(self._build_sources(decision)),
                'search_score': candidate['final_score'],
                'content_types_found': [answer_type],
                'difficulty_level': candidate.get('difficulty_level', 'intermediate'),
                'estimated_time': '2-3 minutes'
            }
            
            # Add video fields only if appropriate
            if answer_type in ['video', 'combined'] and decision.get('video_candidate'):
                video_candidate = decision['video_candidate']
                response.update({
                    'start': video_candidate.get('start', 0),
                    'end': video_candidate.get('end', 0),
                    'video_url': video_candidate.get('video_url', ''),
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
            print(f"❌ Answer building error: {e}")
            return self._safe_fallback()
    
    def _prepare_knowledge_context(self, candidate: Dict) -> str:
        """Prepare knowledge context for LLM"""
        return f"""
Title: {candidate['title']}
Content: {candidate['text']}
URL: {candidate.get('url', '')}
"""
    
    def _prepare_video_context(self, candidate: Dict) -> str:
        """Prepare video context for LLM"""
        timestamp_info = f" (Video timestamp: {self._format_timestamp(candidate.get('start', 0), candidate.get('end', 0))})"
        return f"""
Title: {candidate['title']}
Content: {candidate['text']}{timestamp_info}
Video URL: {candidate.get('video_url', '')}
"""
    
    def _prepare_combined_context(self, knowledge_candidate: Dict, video_candidate: Dict) -> str:
        """Prepare combined context for LLM"""
        timestamp_info = f" (Video timestamp: {self._format_timestamp(video_candidate.get('start', 0), video_candidate.get('end', 0))})"
        return f"""
Knowledge:
Title: {knowledge_candidate['title']}
Content: {knowledge_candidate['text']}
URL: {knowledge_candidate.get('url', '')}

Video:
Title: {video_candidate['title']}
Content: {video_candidate['text']}{timestamp_info}
Video URL: {video_candidate.get('video_url', '')}
"""
    
    def _format_with_llm(self, question: str, context: str, answer_type: str) -> str:
        """Single LLM pass for answer formatting"""
        try:
            if answer_type == 'knowledge':
                prompt = f"""Based on the following knowledge base content, provide a clear, helpful answer to the user's question. Focus on step-by-step instructions if applicable.

Question: {question}

Content:
{context}

Provide a direct, actionable answer:"""
            
            elif answer_type == 'video':
                prompt = f"""Based on the following video transcript, provide a clear, helpful answer to the user's question. Include the video timestamp for reference.

Question: {question}

Content:
{context}

Provide a direct, actionable answer with video reference:"""
            
            elif answer_type == 'combined':
                prompt = f"""Based on the following knowledge base and video content, provide a comprehensive answer to the user's question. Synthesize both sources and include video timestamp reference.

Question: {question}

Content:
{context}

Provide a comprehensive answer combining both sources:"""
            
            else:
                return context  # Fallback to raw content
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ LLM formatting error: {e}")
            return context  # Fallback to raw content
    
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
        
        if decision['decision'] == 'combine' and decision.get('video_candidate'):
            video_candidate = decision['video_candidate']
            sources.append({
                'type': 'video',
                'url': video_candidate.get('video_url', ''),
                'title': video_candidate['title']
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
    
    def _get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get embedding with caching"""
        # Normalize text for caching
        cache_key = f"{model}:{text.lower().strip()}"
        
        # Check cache
        if cache_key in self._embedding_cache:
            # Check TTL
            if time.time() - self._cache_timestamps[cache_key] < self.CACHE_TTL:
                self._embedding_cache.move_to_end(cache_key)
                return self._embedding_cache[cache_key]
            else:
                # Expired, remove from cache
                del self._embedding_cache[cache_key]
                del self._cache_timestamps[cache_key]
        
        # Generate new embedding
        try:
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
    
    def _log_observability(self, question: str, candidates: Dict, reranked: Dict, decision: Dict, latency: int):
        """Comprehensive observability logging"""
        try:
            video_candidates = reranked['video']
            knowledge_candidates = reranked['knowledge']
            
            best_video = video_candidates[0] if video_candidates else None
            best_knowledge = knowledge_candidates[0] if knowledge_candidates else None
            
            # Extract metrics
            kn_hits = len(candidates['knowledge'])
            vid_hits = len(candidates['video'])
            
            kn_score = best_knowledge['final_score'] if best_knowledge else 0
            vid_score = best_video['final_score'] if best_video else 0
            
            kn_text_len = len(best_knowledge['text']) if best_knowledge else 0
            kn_has_text = best_knowledge.get('content_has_text', False) if best_knowledge else False
            
            vid_seekable = best_video.get('seekable', False) if best_video else False
            vid_timestamps = best_video.get('has_timestamps', False) if best_video else False
            vid_start = best_video.get('start', 0) if best_video else 0
            vid_end = best_video.get('end', 0) if best_video else 0
            vid_duration = best_video.get('duration', 0) if best_video else 0
            
            compat = decision.get('compatibility', 0)
            
            # Log comprehensive observability
            print(f"📊 [QA] q_ms={latency}")
            print(f"    kn: hits={kn_hits} S={kn_score:.3f} text_len={kn_text_len} has_text={kn_has_text}")
            print(f"    vid: hits={vid_hits} S={vid_score:.3f} seekable={vid_seekable} ts={vid_timestamps} s={vid_start:.1f} e={vid_end:.1f} dur={vid_duration:.1f}")
            print(f"    compat={compat:.3f}")
            print(f"    decision={decision['decision']} reason={decision['reason']}")
            
        except Exception as e:
            print(f"❌ Observability logging error: {e}")
    
    def _safe_fallback(self) -> Dict:
        """Safe fallback response"""
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
            'estimated_time': 'unknown'
        }


# Global instance for singleton pattern
_context_first_qa_instance = None

def initialize_context_first_qa():
    """Initialize the context-first QA system"""
    global _context_first_qa_instance
    try:
        _context_first_qa_instance = ContextFirstQA()
        print("✅ Context-First QA system initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Context-First QA: {e}")
        return False

def get_context_first_qa():
    """Get the context-first QA system instance"""
    global _context_first_qa_instance
    if _context_first_qa_instance is None:
        print("⚠️ Context-First QA not initialized, initializing now...")
        initialize_context_first_qa()
    return _context_first_qa_instance
