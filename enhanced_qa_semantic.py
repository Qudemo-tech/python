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
    
    # Enhanced thresholds for better quality control
    KNOWLEDGE_MIN_RELEVANCE = 0.65  # Higher threshold for knowledge
    VIDEO_MIN_RELEVANCE = 0.75      # Much higher threshold for videos
    SEMANTIC_SIMILARITY_THRESHOLD = 0.8  # High semantic similarity required
    CONTEXT_UNDERSTANDING_THRESHOLD = 0.7  # Context understanding threshold
    
    # Retrieval parameters
    TOP_K_RECALL = 25
    TOP_K_RERANK = 10
    SCORE_THRESHOLD = 0.15  # Higher initial threshold
    
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
        
        # Embedding cache
        self._embedding_cache = OrderedDict()
        self._cache_timestamps = {}
        self.CACHE_SIZE = 100
        self.CACHE_TTL = 24 * 60 * 60  # 24 hours
    
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
        """Enhanced semantic retrieval with intent-aware searching"""
        try:
            # Get question embedding
            question_embedding = self._get_embedding(question, model="text-embedding-3-small")
            
            # Create enhanced query with intent context
            enhanced_query = self._create_enhanced_query(question, question_analysis)
            enhanced_embedding = self._get_embedding(enhanced_query, model="text-embedding-3-small")
            
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Search both indexes with enhanced parameters
            video_results = self._search_with_intent('video', namespace, question_embedding, enhanced_embedding, question_analysis)
            knowledge_results = self._search_with_intent('knowledge', namespace, question_embedding, enhanced_embedding, question_analysis)
            
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
            
            # Get question embedding for semantic comparison
            question_embedding = self._get_embedding(question, model="text-embedding-3-small")
            
            # Calculate enhanced scores for each candidate
            for candidate in all_candidates:
                # Semantic similarity score
                candidate_text = f"{candidate['title']} {candidate['summary']} {candidate['text_truncated']}"
                candidate_embedding = self._get_embedding(candidate_text, model="text-embedding-3-small")
                semantic_score = cosine_similarity([question_embedding], [candidate_embedding])[0][0]
                
                # Intent alignment score
                intent_score = self._calculate_intent_alignment(candidate, question_analysis)
                
                # Context relevance score
                context_score = self._calculate_context_relevance(candidate, question_analysis)
                
                # Quality score
                quality_score = self._calculate_quality_score(candidate)
                
                # Final enhanced score
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
            
            # Apply quality thresholds
            filtered_candidates = [c for c in all_candidates if c['enhanced_score'] >= 0.6]
            
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
    
    def _intelligent_answer_selection(self, question: str, question_analysis: Dict, scored_candidates: Dict) -> Dict:
        """Intelligent answer selection with strict quality control"""
        try:
            video_candidates = scored_candidates['video']
            knowledge_candidates = scored_candidates['knowledge']
            
            # Get best candidates
            best_video = video_candidates[0] if video_candidates else None
            best_knowledge = knowledge_candidates[0] if knowledge_candidates else None
            
            print(f"🔍 Best video score: {best_video['enhanced_score']:.3f if best_video else 0}")
            print(f"🔍 Best knowledge score: {best_knowledge['enhanced_score']:.3f if best_knowledge else 0}")
            
            # Apply strict quality gates
            video_passes_gate = self._video_passes_quality_gate(best_video, question_analysis)
            knowledge_passes_gate = self._knowledge_passes_quality_gate(best_knowledge, question_analysis)
            
            print(f"🔍 Video passes gate: {video_passes_gate}")
            print(f"🔍 Knowledge passes gate: {knowledge_passes_gate}")
            
            # Decision logic with strict quality control
            if knowledge_passes_gate and not video_passes_gate:
                return {
                    'decision': 'knowledge_only',
                    'candidate': best_knowledge,
                    'reason': 'video_failed_quality_gate'
                }
            
            if video_passes_gate and not knowledge_passes_gate:
                return {
                    'decision': 'video_only',
                    'candidate': best_video,
                    'reason': 'knowledge_failed_quality_gate'
                }
            
            if not knowledge_passes_gate and not video_passes_gate:
                return {
                    'decision': 'fallback',
                    'candidate': best_knowledge or best_video,
                    'reason': 'both_failed_quality_gates'
                }
            
            # Both pass gates - check dominance
            video_score = best_video['enhanced_score']
            knowledge_score = best_knowledge['enhanced_score']
            
            if video_score >= knowledge_score + 0.1:  # Video significantly better
                return {
                    'decision': 'video_only',
                    'candidate': best_video,
                    'reason': 'video_dominates'
                }
            
            if knowledge_score >= video_score + 0.1:  # Knowledge significantly better
                return {
                    'decision': 'knowledge_only',
                    'candidate': best_knowledge,
                    'reason': 'knowledge_dominates'
                }
            
            # Close scores - prefer knowledge for better answer quality
            return {
                'decision': 'knowledge_only',
                'candidate': best_knowledge,
                'reason': 'knowledge_preferred_for_quality'
            }
            
        except Exception as e:
            print(f"❌ Intelligent selection error: {e}")
            return {
                'decision': 'fallback',
                'candidate': None,
                'reason': 'selection_error'
            }
    
    def _video_passes_quality_gate(self, candidate: Optional[Dict], question_analysis: Dict) -> bool:
        """Strict quality gate for video candidates"""
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
        
        # Intent alignment threshold
        if candidate['intent_score'] < 0.6:
            print(f"❌ Video failed: intent score {candidate['intent_score']:.3f} < 0.6")
            return False
        
        # Video-specific requirements
        if not candidate.get('seekable', False) or not candidate.get('has_timestamps', False):
            print(f"❌ Video failed: not seekable or no timestamps")
            return False
        
        # Validate timestamps
        start = candidate.get('start', 0)
        end = candidate.get('end', 0)
        duration = candidate.get('duration', 0)
        
        if not (0 <= start < end <= duration):
            print(f"❌ Video failed: invalid timestamps")
            return False
        
        print(f"✅ Video passes all quality gates")
        return True
    
    def _knowledge_passes_quality_gate(self, candidate: Optional[Dict], question_analysis: Dict) -> bool:
        """Strict quality gate for knowledge candidates"""
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
        
        # Text length requirement
        if len(candidate['text']) < 100:
            print(f"❌ Knowledge failed: text too short")
            return False
        
        print(f"✅ Knowledge passes all quality gates")
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
            
            # Build response payload
            response = {
                'success': True,
                'answer': formatted_answer,
                'sources': self._build_sources(decision),
                'total_sources': len(self._build_sources(decision)),
                'search_score': candidate['enhanced_score'],
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
            print(f"❌ Quality assurance error: {e}")
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
    
    def _format_with_intent_awareness(self, question: str, context: str, answer_type: str, question_analysis: Dict) -> str:
        """Format answer with intent awareness"""
        try:
            intent_type = question_analysis.get('intent_type', '')
            main_action = question_analysis.get('main_action', '')
            
            if answer_type == 'knowledge':
                if intent_type == 'create':
                    prompt = f"""The user wants to CREATE something. Based on the following knowledge base content, provide clear, step-by-step instructions for creating what they asked for.

Question: {question}
Main Action: {main_action}

Content:
{context}

Provide a direct, actionable answer with numbered steps for creating what they need:"""
                elif intent_type == 'how_to':
                    prompt = f"""The user is asking HOW TO do something. Based on the following knowledge base content, provide clear, step-by-step instructions.

Question: {question}
Main Action: {main_action}

Content:
{context}

Provide a direct, actionable answer with numbered steps:"""
                else:
                    prompt = f"""Based on the following knowledge base content, provide a clear, helpful answer to the user's question.

Question: {question}

Content:
{context}

Provide a direct, actionable answer:"""
            
            elif answer_type == 'video':
                if intent_type == 'create':
                    prompt = f"""The user wants to CREATE something. Based on the following video transcript, provide clear, step-by-step instructions for creating what they asked for.

Question: {question}
Main Action: {main_action}

Content:
{context}

Provide a direct, actionable answer with numbered steps for creating what they need:"""
                elif intent_type == 'how_to':
                    prompt = f"""The user is asking HOW TO do something. Based on the following video transcript, provide clear, step-by-step instructions.

Question: {question}
Main Action: {main_action}

Content:
{context}

Provide a direct, actionable answer with numbered steps:"""
                else:
                    prompt = f"""Based on the following video transcript, provide a clear, helpful answer to the user's question.

Question: {question}

Content:
{context}

Provide a direct, actionable answer:"""
            
            else:
                return context  # Fallback to raw content
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Intent-aware formatting error: {e}")
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
            
            kn_score = best_knowledge['enhanced_score'] if best_knowledge else 0
            vid_score = best_video['enhanced_score'] if best_video else 0
            
            kn_semantic = best_knowledge['semantic_score'] if best_knowledge else 0
            vid_semantic = best_video['semantic_score'] if best_video else 0
            
            kn_intent = best_knowledge['intent_score'] if best_knowledge else 0
            vid_intent = best_video['intent_score'] if best_video else 0
            
            # Log comprehensive observability
            print(f"📊 [Enhanced QA] q_ms={latency}")
            print(f"    intent={question_analysis.get('intent_type', 'unknown')}")
            print(f"    kn: hits={kn_hits} S={kn_score:.3f} sem={kn_semantic:.3f} intent={kn_intent:.3f}")
            print(f"    vid: hits={vid_hits} S={vid_score:.3f} sem={vid_semantic:.3f} intent={vid_intent:.3f}")
            print(f"    decision={decision['decision']} reason={decision['reason']}")
            
        except Exception as e:
            print(f"❌ Enhanced observability logging error: {e}")
    
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
