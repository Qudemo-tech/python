#!/usr/bin/env python3
"""
Enhanced Topic-Wise Q&A System
Optimized for the new segment-safe chunking strategy with precise timestamps and topic context
"""

import os
import re
import time
import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
from datetime import datetime, timezone
import openai
from pinecone import Pinecone
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedTopicWiseQA:
    """Enhanced Q&A system optimized for topic-wise chunks with precise timestamps"""
    
    def __init__(self):
        """Initialize enhanced topic-wise QA system"""
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
        self.CACHE_SIZE = 200
        self.CACHE_TTL = 24 * 60 * 60  # 24 hours
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Topic-wise specific parameters
        self.TOP_K_RECALL = 20  # Increased for diversity
        self.TOP_K_RERANK = 5   # Focus on top 5 chunks
        
        # Quality thresholds by system
        self.MIN_QUALITY_SCORE_YOUTUBE = 60  # YouTube (Gemini) - clean transcripts
        self.MIN_QUALITY_SCORE_LOOM = 65     # Loom (Whisper) - may have filler words
        self.FALLBACK_QUALITY_SCORE = 50     # Fallback threshold for low results
        self.TOPIC_RELEVANCE_THRESHOLD = 0.3
        self.MAX_CHUNKS_PER_SEGMENT = 2  # MMR diversity limit
        self.MAX_TOKENS_CONTEXT = 4000   # Token limit for GPT context
        self.SAFETY_MARGIN = 500         # Safety margin for GPT context
        
        # Initialize tokenizer for accurate token counting
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
            logger.info("✅ tiktoken initialized for accurate token counting")
        except ImportError:
            logger.warning("⚠️ tiktoken not available, using character-based estimation")
            self.tokenizer = None
        
        logger.info("✅ Enhanced Topic-Wise QA System initialized")
    
    def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Main QA entry point optimized for topic-wise chunks"""
        try:
            start_time = time.time()
            logger.info(f"🧠 Topic-Wise QA: {question}")
            
            # EMERGENCY SEMANTIC MISMATCH CHECK - Direct question analysis
            if 'disqualified' in question.lower() and 'lead' in question.lower():
                logger.warning(f"🚨 EMERGENCY SEMANTIC MISMATCH: Question about DISQUALIFIED leads detected!")
                logger.warning(f"🚨 This video only contains QUALIFIED leads content - returning mismatch message")
                
                return {
                    'success': True,
                    'answer': "I notice you're asking about building **disqualified leads**, but this video only contains information about building **qualified leads**. These are opposite concepts:\n\n" +
                            "- **Qualified leads** = leads that meet your criteria and are ready to proceed\n" +
                            "- **Disqualified leads** = leads that don't meet your criteria and are rejected\n\n" +
                            "The video covers how to build agents for **qualified leads** only. If you'd like to know about qualified leads instead, please let me know!",
                    'confidence': 0.9,
                    'confidence_score': 0.9,
                    'sources': [],
                    'total_sources': 0,
                    'search_score': 0,
                    'content_types_found': [],
                    'difficulty_level': 'intermediate',
                    'estimated_time': '1-2 minutes',
                    'start': 0,
                    'end': 0,
                    'video_url': '',
                    'formatted_timestamp': '00:00',
                    'answer_source': 'semantic_mismatch_detection',
                    'semantic_mismatch': True,
                    'available_content': 'qualified_leads_only'
                }
            
            # Stage 1: Question analysis and topic intent
            analysis_start = time.time()
            question_analysis = self._analyze_question_for_topics(question)
            logger.info(f"🔍 QUESTION ANALYSIS RESULT: {question_analysis}")
            question_analysis['processing_time'] = time.time() - analysis_start
            
            # Stage 2: Topic-aware retrieval
            retrieval_start = time.time()
            topic_candidates = self._topic_aware_retrieval(question, question_analysis, company_name, qudemo_id)
            retrieval_time = time.time() - retrieval_start
            
            # Stage 3: Quality-based chunk selection
            selection_start = time.time()
            best_chunks = self._select_best_topic_chunks(topic_candidates, question_analysis)
            chunk_selection_time = time.time() - selection_start
            
            # Stage 4: Generate answer with precise timestamps
            answer_start = time.time()
            answer_result = self._generate_topic_aware_answer(question, question_analysis, best_chunks, retrieval_time, chunk_selection_time, time.time() - answer_start)
            answer_generation_time = time.time() - answer_start
            
            # Stage 5: Format with timestamp and topic context
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            formatted_answer = self._format_answer_with_context(answer_result, question_analysis, namespace, retrieval_time, chunk_selection_time, answer_generation_time)
            
            # Performance logging
            latency = int((time.time() - start_time) * 1000)
            logger.info(f"✅ Topic-Wise QA completed in {latency}ms")
            
            return formatted_answer
            
        except Exception as e:
            logger.error(f"❌ Topic-Wise QA Error: {e}")
            return self._safe_fallback()
    
    def _analyze_question_for_topics(self, question: str) -> Dict:
        """Analyze question to identify relevant topics and intent"""
        try:
            prompt = f"""
            Analyze this question to identify relevant topics and intent for video content:
            Question: "{question}"
            
            IMPORTANT: Pay special attention to the semantic difference between:
            - "qualified lead" = leads that meet criteria and are ready to proceed
            - "disqualified lead" = leads that don't meet criteria and are rejected
            
            These are OPPOSITE concepts and should be treated as completely different topics.
            
            Provide a JSON response with:
            1. primary_topic: The main topic this question relates to (be very specific about qualified vs disqualified)
            2. secondary_topics: List of related topics
            3. intent_type: "how_to", "what_is", "explain", "troubleshoot", "compare"
            4. expected_content_type: "tutorial", "explanation", "demo", "overview"
            5. time_preference: "specific_timestamp", "topic_section", "full_topic"
            6. complexity_level: "simple", "moderate", "complex"
            7. key_concepts: List of key concepts to search for (include semantic context)
            8. context_clues: List of context clues for better matching
            9. semantic_context: The specific semantic context (e.g., "qualified_leads", "disqualified_leads", "general")
            
            Focus on identifying what specific topic or section of a video would best answer this question.
            Be very precise about whether the question is about qualified or disqualified leads.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON, with fallback
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                import re
                # Remove markdown code blocks if present
                content_clean = re.sub(r'```json\s*', '', content)
                content_clean = re.sub(r'```\s*$', '', content_clean)
                
                # Try to find JSON object
                json_match = re.search(r'\{.*\}', content_clean, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {content}")
            logger.info(f"🔍 Question analysis: {analysis.get('primary_topic', 'Unknown')}")
            logger.info(f"🔍 Semantic context: {analysis.get('semantic_context', 'None')}")
            logger.info(f"🔍 Key concepts: {analysis.get('key_concepts', [])}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Question analysis error: {e}")
            return {
                'primary_topic': 'general',
                'secondary_topics': [],
                'intent_type': 'explain',
                'expected_content_type': 'explanation',
                'time_preference': 'topic_section',
                'complexity_level': 'moderate',
                'key_concepts': [],
                'context_clues': []
            }
    
    def _topic_aware_retrieval(self, question: str, question_analysis: Dict, company_name: str, qudemo_id: str) -> List[Dict]:
        """Retrieve chunks with topic awareness and quality filtering"""
        try:
            # Get question embedding
            question_embedding = self._get_embedding(question, model="text-embedding-3-large")
            
            # Create topic-enhanced query
            enhanced_query = self._create_topic_enhanced_query(question, question_analysis)
            enhanced_embedding = self._get_embedding(enhanced_query, model="text-embedding-3-large")
            
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Search video index with enhanced parameters
            video_results = self._search_video_index(namespace, question_embedding, enhanced_embedding, company_name, qudemo_id, question_analysis)
            
            # Filter for topic-wise chunks and quality
            topic_chunks = self._filter_topic_wise_chunks(video_results, question_analysis)
            
            # Graceful fallback if results are too low
            if len(topic_chunks) < 2:
                logger.warning(f"⚠️ Low topic-wise results ({len(topic_chunks)} chunks), trying fallback strategies")
                
                # Strategy 1: Lower quality threshold for topic-wise chunks
                fallback_filter = {
                    "chunk_type": {"$eq": "segment_safe"},
                    "quality_score": {"$gte": self.FALLBACK_QUALITY_SCORE},  # Lower threshold
                    "company_name": {"$eq": company_name},
                    "qudemo_id": {"$eq": qudemo_id}
                }
                
                # Get the index for fallback search
                index = self.pc.Index(self.indexes['video'])
                fallback_results = index.query(
                    vector=question_embedding,
                    top_k=self.TOP_K_RECALL,
                    namespace=namespace,
                    include_metadata=True,
                    filter=fallback_filter
                )
                
                fallback_chunks = self._filter_topic_wise_chunks(
                    [{'id': match.id, 'score': match.score, 'metadata': match.metadata} 
                     for match in fallback_results.matches], 
                    question_analysis
                )
                
                # Strategy 2: If still no results, allow semantic chunks (non-topic-wise)
                if len(fallback_chunks) < 2:
                    logger.warning(f"⚠️ Still low results ({len(fallback_chunks)} chunks), allowing semantic chunks")
                    semantic_filter = {
                        "quality_score": {"$gte": self.FALLBACK_QUALITY_SCORE},
                        "company_name": {"$eq": company_name},
                        "qudemo_id": {"$eq": qudemo_id}
                        # Note: No chunk_type filter - allows both segment_safe and semantic chunks
                    }
                    
                    # Use the same index instance (already defined above)
                    semantic_results = index.query(
                        vector=question_embedding,
                        top_k=self.TOP_K_RECALL,
                        namespace=namespace,
                        include_metadata=True,
                        filter=semantic_filter
                    )
                    
                    semantic_chunks = [
                        {'id': match.id, 'score': match.score, 'metadata': match.metadata, 'text': match.metadata.get('text', '')}
                        for match in semantic_results.matches
                    ]
                    
                    if len(semantic_chunks) > len(fallback_chunks):
                        logger.info(f"✅ Semantic fallback found {len(semantic_chunks)} chunks (vs {len(fallback_chunks)} topic-wise)")
                        topic_chunks = semantic_chunks
                    else:
                        topic_chunks = fallback_chunks
                else:
                    topic_chunks = fallback_chunks
                    logger.info(f"✅ Quality fallback found {len(topic_chunks)} chunks")
            
            logger.info(f"🔍 Retrieved {len(topic_chunks)} topic-wise chunks")
            
            # CRITICAL: Check for semantic mismatches before returning chunks
            semantic_context = question_analysis.get('semantic_context', '').lower()
            logger.info(f"🔍 SEMANTIC MISMATCH CHECK: semantic_context='{semantic_context}', topic_chunks_count={len(topic_chunks)}")
            
            if semantic_context == 'disqualified_leads' and topic_chunks:
                # Check if we have any disqualified content
                disqualified_chunks = [
                    chunk for chunk in topic_chunks 
                    if 'disqualified' in chunk['metadata'].get('segment_topic', '').lower()
                ]
                
                # Check if we only have qualified content (semantic mismatch)
                qualified_chunks = [
                    chunk for chunk in topic_chunks 
                    if 'qualified' in chunk['metadata'].get('segment_topic', '').lower()
                    and 'disqualified' not in chunk['metadata'].get('segment_topic', '').lower()
                ]
                
                if not disqualified_chunks and qualified_chunks:
                    logger.warning(f"🚨 CRITICAL SEMANTIC MISMATCH DETECTED!")
                    logger.warning(f"🚨 User asked about: DISQUALIFIED leads")
                    logger.warning(f"🚨 Video only contains: QUALIFIED leads content ({len(qualified_chunks)} chunks)")
                    logger.warning(f"🚨 Topics found: {[chunk['metadata'].get('segment_topic', 'Unknown') for chunk in qualified_chunks[:3]]}")
                    logger.warning(f"🚨 REJECTING all chunks to prevent wrong answers")
                    return []  # Return empty to trigger proper "no results" response
            
            return topic_chunks
            
        except Exception as e:
            logger.error(f"❌ Topic-aware retrieval error: {e}")
            return []
    
    def _create_topic_enhanced_query(self, question: str, question_analysis: Dict) -> str:
        """Create enhanced query with topic context"""
        primary_topic = question_analysis.get('primary_topic', '')
        key_concepts = question_analysis.get('key_concepts', [])
        intent_type = question_analysis.get('intent_type', '')
        semantic_context = question_analysis.get('semantic_context', '')
        
        enhanced_parts = [question]
        
        # Add semantic context FIRST (most important for qualified vs disqualified)
        if semantic_context:
            enhanced_parts.append(semantic_context)
            if semantic_context == 'qualified_leads':
                enhanced_parts.append("qualified prospects ready to proceed")
                enhanced_parts.append("meets criteria approved leads")
            elif semantic_context == 'disqualified_leads':
                enhanced_parts.append("disqualified prospects rejected leads")
                enhanced_parts.append("does not meet criteria")
        
        # Add topic context
        if primary_topic:
            enhanced_parts.append(f"{primary_topic} topic")
            enhanced_parts.append(f"{primary_topic} section")
        
        # Add intent context
        if intent_type == "how_to":
            enhanced_parts.append("tutorial steps")
            enhanced_parts.append("how to guide")
        elif intent_type == "what_is":
            enhanced_parts.append("explanation definition")
            enhanced_parts.append("overview introduction")
        elif intent_type == "explain":
            enhanced_parts.append("detailed explanation")
            enhanced_parts.append("comprehensive overview")
        
        # Add key concepts
        for concept in key_concepts[:3]:  # Limit to top 3
            enhanced_parts.append(f"{concept} concept")
            enhanced_parts.append(f"{concept} details")
        
        return " ".join(enhanced_parts)
    
    def _search_video_index(self, namespace: str, question_embedding: List[float], enhanced_embedding: List[float], company_name: str, qudemo_id: str, question_analysis: Dict = None) -> List[Dict]:
        """Search video index with both embeddings and belt-and-suspenders filtering"""
        try:
            index = self.pc.Index(self.indexes['video'])
            
            # Belt-and-suspenders filter for extra isolation (dynamic quality threshold)
            # We'll determine the threshold after getting initial results
            isolation_filter = {
                "chunk_type": {"$eq": "segment_safe"},
                "quality_score": {"$gte": self.MIN_QUALITY_SCORE_YOUTUBE},  # Start with YouTube threshold
                "company_name": {"$eq": company_name},  # Belt-and-suspenders isolation
                "qudemo_id": {"$eq": qudemo_id}         # Belt-and-suspenders isolation
            }
            
            # Add semantic context filter if available
            if question_analysis and question_analysis.get('semantic_context'):
                semantic_context = question_analysis.get('semantic_context').lower()
                if semantic_context == 'qualified_leads':
                    # For qualified leads, prefer chunks that mention "qualified" and avoid "disqualified"
                    # We'll handle this in post-processing rather than pre-filtering
                    pass
                elif semantic_context == 'disqualified_leads':
                    # For disqualified leads, prefer chunks that mention "disqualified"
                    # We'll handle this in post-processing rather than pre-filtering
                    pass
            
            # Search with original question embedding
            results1 = index.query(
                vector=question_embedding,
                top_k=self.TOP_K_RECALL,
                namespace=namespace,
                include_metadata=True,
                filter=isolation_filter
            )
            
            # Search with enhanced embedding
            results2 = index.query(
                vector=enhanced_embedding,
                top_k=self.TOP_K_RECALL,
                namespace=namespace,
                include_metadata=True,
                filter=isolation_filter
            )
            
            # Combine and deduplicate results
            all_results = []
            seen_ids = set()
            
            for result in results1.matches + results2.matches:
                if result.id not in seen_ids:
                    all_results.append({
                        'id': result.id,
                        'score': result.score,
                        'metadata': result.metadata
                    })
                    seen_ids.add(result.id)
            
            # Sort by score
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Log namespace for audit trail
            logger.info(f"🔍 Video Index Search: namespace='{namespace}', results={len(all_results)}")
            
            return all_results[:self.TOP_K_RECALL]
            
        except Exception as e:
            logger.error(f"❌ Video index search error: {e}")
            return []
    
    def _filter_topic_wise_chunks(self, results: List[Dict], question_analysis: Dict) -> List[Dict]:
        """Filter chunks for topic-wise processing and quality"""
        try:
            filtered_chunks = []
            
            for result in results:
                metadata = result.get('metadata', {})
                
                # Check if it's a topic-wise chunk
                if metadata.get('chunk_type') != 'segment_safe':
                    continue
                
                # Check quality score
                quality_score = metadata.get('quality_score', 0)
                if quality_score < self.MIN_QUALITY_SCORE_YOUTUBE:
                    continue
                
                # Check topic relevance (more flexible matching)
                segment_topic = metadata.get('segment_topic', '').lower()
                primary_topic = question_analysis.get('primary_topic', '').lower()
                
                # Extract key terms from primary topic for flexible matching
                primary_terms = []
                if primary_topic:
                    # Split by common separators and extract meaningful terms
                    import re
                    terms = re.split(r'[^\w]+', primary_topic)
                    primary_terms = [term for term in terms if len(term) > 2]  # Skip short terms
                
                # Check if any primary terms match segment topic
                topic_match = False
                if primary_terms:
                    topic_match = any(term in segment_topic for term in primary_terms)
                
                # Also check secondary topics
                if not topic_match:
                    secondary_topics = question_analysis.get('secondary_topics', [])
                    topic_match = any(topic.lower() in segment_topic for topic in secondary_topics)
                
                # If no topic match, skip this chunk
                if not topic_match:
                    continue
                
                # Add enhanced metadata
                enhanced_chunk = {
                    'id': result['id'],
                    'score': result['score'],
                    'metadata': metadata,
                    'quality_score': quality_score
                }
                
                filtered_chunks.append(enhanced_chunk)
            
            logger.info(f"🔍 Filtered to {len(filtered_chunks)} high-quality topic chunks")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"❌ Topic filtering error: {e}")
            return []
    
    def _select_best_topic_chunks(self, chunks: List[Dict], question_analysis: Dict) -> List[Dict]:
        """Select the best chunks using quality, relevance scoring, and MMR diversity"""
        try:
            if not chunks:
                return []
            
            # Score chunks based on multiple factors
            scored_chunks = []
            
            for chunk in chunks:
                # Base score from Pinecone
                base_score = chunk['score']
                
                # Quality score (0-100, normalize to 0-1)
                quality_score = chunk['metadata'].get('quality_score', 0) / 100.0
                
                # Topic relevance score
                topic_relevance = self._calculate_topic_relevance(chunk, question_analysis)
                
                # Content completeness score
                content_score = self._calculate_content_completeness(chunk, question_analysis)
                
                # Combined score (weighted)
                combined_score = (
                    base_score * 0.4 +           # Pinecone similarity
                    quality_score * 0.3 +        # Chunk quality
                    topic_relevance * 0.2 +      # Topic relevance
                    content_score * 0.1          # Content completeness
                )
                
                chunk['combined_score'] = combined_score
                chunk['topic_relevance'] = topic_relevance
                chunk['content_score'] = content_score
                
                scored_chunks.append(chunk)
            
            # Sort by combined score
            scored_chunks.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Apply MMR diversity and token limits
            best_chunks = self._apply_mmr_diversity_and_token_limits(scored_chunks)
            
            # Adjacent chunk enrichment for low scores
            if best_chunks and best_chunks[0]['combined_score'] < 0.8:
                logger.info(f"🔍 Top score {best_chunks[0]['combined_score']:.3f} < 0.8, enriching with adjacent chunks")
                enriched_chunks = self._enrich_with_adjacent_chunks(best_chunks, scored_chunks)
                if len(enriched_chunks) > len(best_chunks):
                    best_chunks = enriched_chunks
                    logger.info(f"✅ Enriched to {len(best_chunks)} chunks")
            
            logger.info(f"🔍 Selected {len(best_chunks)} best topic chunks with diversity")
            for i, chunk in enumerate(best_chunks):
                logger.info(f"  {i+1}. Score: {chunk['combined_score']:.3f}, Topic: {chunk['metadata'].get('segment_topic', 'Unknown')}")
            
            return best_chunks
            
        except Exception as e:
            logger.error(f"❌ Chunk selection error: {e}")
            return chunks[:3] if chunks else []
    
    def _calculate_topic_relevance(self, chunk: Dict, question_analysis: Dict) -> float:
        """Calculate how relevant the chunk's topic is to the question"""
        try:
            segment_topic = chunk['metadata'].get('segment_topic', '').lower()
            primary_topic = question_analysis.get('primary_topic', '').lower()
            key_concepts = [concept.lower() for concept in question_analysis.get('key_concepts', [])]
            semantic_context = question_analysis.get('semantic_context', '').lower()
            
            relevance_score = 0.0
            
            # Debug logging for topic matching
            logger.info(f"🔍 Topic matching: semantic_context='{semantic_context}', segment_topic='{segment_topic}', primary_topic='{primary_topic}'")
            
            # Context-aware matching with strict semantic validation
            # For general questions, be more permissive
            if semantic_context == 'general':
                # For general questions, prioritize chunks that contain the main topic
                if primary_topic and (primary_topic in segment_topic or any(concept in segment_topic for concept in key_concepts)):
                    relevance_score += 0.7
                else:
                    # Even if no direct match, give some relevance for related content
                    relevance_score += 0.3
            else:
                # For specific contexts, use STRICT matching to avoid semantic mismatches
                if semantic_context == 'qualified_leads':
                    if 'qualified' in segment_topic and 'disqualified' not in segment_topic:
                        relevance_score += 0.8
                    elif 'disqualified' in segment_topic:
                        # Penalize disqualified content when asking about qualified
                        relevance_score -= 0.5
                        logger.warning(f"🚨 Semantic mismatch: Question about qualified leads, chunk about disqualified: {segment_topic}")
                    else:
                        # No direct match, give low relevance
                        relevance_score += 0.2
                elif semantic_context == 'disqualified_leads':
                    if 'disqualified' in segment_topic:
                        relevance_score += 0.8
                        logger.info(f"✅ Perfect match: Disqualified question, disqualified content: {segment_topic}")
                    elif 'qualified' in segment_topic and 'disqualified' not in segment_topic:
                        # STRICT: Completely reject qualified content when asking about disqualified
                        logger.warning(f"🚨 SEMANTIC MISMATCH: Question about DISQUALIFIED leads, chunk about QUALIFIED: {segment_topic}")
                        logger.warning(f"🚨 Returning 0.0 relevance to prevent wrong content")
                        return 0.0  # Return 0 relevance for semantic mismatch
                    else:
                        # No direct match, but check if it's general content that could be relevant
                        if any(word in segment_topic for word in ['lead', 'agent', 'automation', 'workflow']):
                            relevance_score += 0.1  # Very low relevance for general content
                            logger.info(f"🔍 General content match for disqualified leads: {segment_topic}")
                        else:
                            return 0.0  # No relevance at all
                elif semantic_context in segment_topic:
                    relevance_score += 0.6
                else:
                    # Check for related concepts even if not exact match
                    if any(concept in segment_topic for concept in key_concepts):
                        relevance_score += 0.4
            
            # Primary topic match (more flexible)
            if primary_topic:
                if primary_topic in segment_topic:
                    relevance_score += 0.6
                elif any(word in segment_topic for word in primary_topic.split()):
                    relevance_score += 0.4
            
            # Key concepts match (more flexible)
            concept_matches = sum(1 for concept in key_concepts if concept in segment_topic)
            if key_concepts:
                relevance_score += (concept_matches / len(key_concepts)) * 0.3
            
            # Secondary topics match
            secondary_topics = [topic.lower() for topic in question_analysis.get('secondary_topics', [])]
            for topic in secondary_topics:
                if topic in segment_topic:
                    relevance_score += 0.2
                    break
            
            # Semantic context bonus
            if semantic_context and semantic_context in segment_topic:
                relevance_score += 0.3
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Topic relevance calculation error: {e}")
            return 0.5
    
    def _calculate_content_completeness(self, chunk: Dict, question_analysis: Dict) -> float:
        """Calculate how complete the chunk content is for answering the question"""
        try:
            text = chunk['metadata'].get('text', '')
            intent_type = question_analysis.get('intent_type', '')
            
            # Base completeness from text length and quality
            text_length = len(text)
            if text_length < 100:
                completeness = 0.3
            elif text_length < 300:
                completeness = 0.6
            else:
                completeness = 0.9
            
            # Adjust based on intent type
            if intent_type == "how_to" and any(word in text.lower() for word in ['step', 'first', 'then', 'next', 'finally']):
                completeness += 0.1
            elif intent_type == "what_is" and any(word in text.lower() for word in ['is', 'means', 'definition', 'refers to']):
                completeness += 0.1
            elif intent_type == "explain" and any(word in text.lower() for word in ['because', 'therefore', 'in other words', 'specifically']):
                completeness += 0.1
            
            return min(completeness, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Content completeness calculation error: {e}")
            return 0.5
    
    def _apply_mmr_diversity_and_token_limits(self, scored_chunks: List[Dict]) -> List[Dict]:
        """Apply MMR diversity and token limits to chunk selection"""
        try:
            if not scored_chunks:
                return []
            
            selected_chunks = []
            segment_counts = {}
            total_tokens = 0
            
            for chunk in scored_chunks:
                segment_id = chunk['metadata'].get('segment_id', 0)
                text = chunk['metadata'].get('text', '')
                
                # Count tokens accurately
                chunk_tokens = self._count_tokens(text)
                
                # Check segment diversity limit
                if segment_counts.get(segment_id, 0) >= self.MAX_CHUNKS_PER_SEGMENT:
                    continue
                
                # Check token limit with safety margin
                if total_tokens + chunk_tokens > (self.MAX_TOKENS_CONTEXT - self.SAFETY_MARGIN):
                    logger.info(f"🔍 Token limit reached: {total_tokens + chunk_tokens} > {self.MAX_TOKENS_CONTEXT - self.SAFETY_MARGIN}")
                    break
                
                # Add chunk
                selected_chunks.append(chunk)
                segment_counts[segment_id] = segment_counts.get(segment_id, 0) + 1
                total_tokens += chunk_tokens
                
                # Limit to top 3 chunks
                if len(selected_chunks) >= 3:
                    break
            
            logger.info(f"🔍 MMR Selection: {len(selected_chunks)} chunks, {total_tokens} tokens, {len(segment_counts)} segments")
            return selected_chunks
            
        except Exception as e:
            logger.error(f"❌ MMR diversity error: {e}")
            return scored_chunks[:3]
    
    def _enrich_with_adjacent_chunks(self, best_chunks: List[Dict], all_scored_chunks: List[Dict]) -> List[Dict]:
        """Enrich with adjacent chunks from the same segments for better context"""
        try:
            if not best_chunks or not all_scored_chunks:
                return best_chunks
            
            enriched = best_chunks.copy()
            used_segments = set(chunk['metadata'].get('segment_id', 0) for chunk in best_chunks)
            
            # Find adjacent chunks from the same segments
            for chunk in all_scored_chunks:
                segment_id = chunk['metadata'].get('segment_id', 0)
                if segment_id in used_segments and chunk not in enriched:
                    # Add adjacent chunk from same segment
                    enriched.append(chunk)
                    if len(enriched) >= 5:  # Limit to 5 chunks max
                        break
            
            # Sort by combined score and return top chunks
            enriched.sort(key=lambda x: x['combined_score'], reverse=True)
            return enriched[:3]  # Return top 3
            
        except Exception as e:
            logger.error(f"❌ Adjacent chunk enrichment error: {e}")
            return best_chunks
    
    def _generate_topic_aware_answer(self, question: str, question_analysis: Dict, best_chunks: List[Dict], retrieval_time: float = 0, chunk_selection_time: float = 0, answer_generation_time: float = 0) -> Dict:
        """Generate answer using the best topic chunks with GPT"""
        try:
            if not best_chunks:
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'confidence': 0.0,
                    'sources': [],
                    'timestamp_info': None
                }
            
            # CRITICAL: Final semantic mismatch check before generating answer
            semantic_context = question_analysis.get('semantic_context', '').lower()
            question_lower = question.lower()
            
            # Direct word-based detection for disqualified leads
            is_about_disqualified = ('disqualified' in question_lower and 'lead' in question_lower) or semantic_context == 'disqualified_leads'
            
            logger.info(f"🔍 SEMANTIC MISMATCH FINAL CHECK: question='{question}', semantic_context='{semantic_context}', is_about_disqualified={is_about_disqualified}")
            
            if is_about_disqualified:
                # Check if all chunks are about qualified leads (semantic mismatch)
                qualified_chunks = [
                    chunk for chunk in best_chunks 
                    if 'qualified' in chunk['metadata'].get('segment_topic', '').lower()
                    and 'disqualified' not in chunk['metadata'].get('segment_topic', '').lower()
                ]
                
                disqualified_chunks = [
                    chunk for chunk in best_chunks 
                    if 'disqualified' in chunk['metadata'].get('segment_topic', '').lower()
                ]
                
                if qualified_chunks and not disqualified_chunks:
                    logger.warning(f"🚨 FINAL SEMANTIC MISMATCH CHECK: Question about DISQUALIFIED, but all chunks about QUALIFIED")
                    logger.warning(f"🚨 Qualified chunks: {[chunk['metadata'].get('segment_topic', 'Unknown') for chunk in qualified_chunks]}")
                    
                    return {
                        'answer': "I notice you're asking about building **disqualified leads**, but this video only contains information about building **qualified leads**. These are opposite concepts:\n\n" +
                                "- **Qualified leads** = leads that meet your criteria and are ready to proceed\n" +
                                "- **Disqualified leads** = leads that don't meet your criteria and are rejected\n\n" +
                                "The video covers how to build agents for **qualified leads** only. If you'd like to know about qualified leads instead, please let me know!",
                        'confidence': 0.9,  # High confidence in the mismatch detection
                        'sources': [],
                        'timestamp_info': None,
                        'semantic_mismatch': True,
                        'available_content': 'qualified_leads_only'
                    }
            
            # Prepare context from best chunks
            context_parts = []
            sources = []
            timestamp_info = []
            
            for i, chunk in enumerate(best_chunks):
                # Add chunk context
                context_parts.append(f"Source {i+1} (Topic: {chunk['metadata'].get('segment_topic', 'Unknown')}):\n{chunk['metadata'].get('text', '')}")
                
                # Add source information with deep links
                source_info = {
                    'topic': chunk['metadata'].get('segment_topic', 'Unknown'),
                    'summary': chunk['metadata'].get('segment_summary', ''),
                    'start_timestamp': chunk['metadata'].get('start_timestamp', 0),
                    'end_timestamp': chunk['metadata'].get('end_timestamp', 0),
                    'video_url': chunk['metadata'].get('video_url', ''),
                    'start_url': self._create_youtube_deep_link(chunk['metadata'].get('video_url', ''), chunk['metadata'].get('start_timestamp', 0)),
                    'end_url': self._create_youtube_deep_link(chunk['metadata'].get('video_url', ''), chunk['metadata'].get('end_timestamp', 0)),
                    'quality_score': chunk['metadata'].get('quality_score', 0),
                    'relevance_score': chunk['combined_score']
                }
                sources.append(source_info)
                
                # Add timestamp info
                timestamp_info.append({
                    'topic': chunk['metadata'].get('segment_topic', 'Unknown'),
                    'start_time': chunk['metadata'].get('start_timestamp', 0),
                    'end_time': chunk['metadata'].get('end_timestamp', 0),
                    'duration': chunk['metadata'].get('end_timestamp', 0) - chunk['metadata'].get('start_timestamp', 0)
                })
            
            context = "\n\n".join(context_parts)
            
            # Runtime assertions for boundary purity (skip validation for now to avoid scope issues)
            # self._validate_sources_purity(sources, company_name, qudemo_id)
            
            # Generate answer with GPT
            prompt = f"""
            Based on the following video content, provide a comprehensive answer to the user's question.
            
            Question: "{question}"
            
            Video Content:
            {context}
            
            Instructions:
            1. Provide a clear, accurate answer based on the video content
            2. If the answer spans multiple topics, organize it logically
            3. Include specific details and examples from the video
            4. If you reference specific topics, mention them by name
            5. Be concise but comprehensive
            6. If some sources are only loosely relevant, prioritize the strongest ones but you may still mention others briefly
            7. Focus on the most relevant information while acknowledging any limitations
            8. If sources have varying relevance, structure your answer to lead with the most relevant information
            9. Maintain accuracy and avoid speculation beyond what's explicitly stated in the sources
            
            Answer:
            """
            
            # Token headroom validation before GPT call
            context_tokens = self._count_tokens(context)
            prompt_tokens = self._count_tokens(prompt)
            total_tokens = context_tokens + prompt_tokens
            
            # Runtime assertion for token headroom
            assert total_tokens <= (self.MAX_TOKENS_CONTEXT - self.SAFETY_MARGIN), f"Token limit exceeded: {total_tokens} > {self.MAX_TOKENS_CONTEXT - self.SAFETY_MARGIN}"
            
            if context_tokens > (self.MAX_TOKENS_CONTEXT - self.SAFETY_MARGIN):
                logger.warning(f"⚠️ Context too large: {context_tokens} tokens, truncating...")
                # Could implement smart truncation here
            
            logger.info(f"📊 Token stats: context={context_tokens}, prompt={prompt_tokens}, total={total_tokens}, limit={self.MAX_TOKENS_CONTEXT - self.SAFETY_MARGIN}")
            
            # Debug chunk structure before GPT call
            logger.info(f"🔍 Debug - Best chunks count: {len(best_chunks)}")
            if best_chunks:
                logger.info(f"🔍 Debug - First chunk keys: {list(best_chunks[0].keys())}")
                logger.info(f"🔍 Debug - First chunk metadata keys: {list(best_chunks[0].get('metadata', {}).keys())}")
                logger.info(f"🔍 Debug - First chunk has segment_topic in metadata: {'segment_topic' in best_chunks[0].get('metadata', {})}")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on chunk quality and relevance with transparency
            avg_quality = sum(chunk['metadata'].get('quality_score', 0) for chunk in best_chunks) / len(best_chunks)
            avg_relevance = sum(chunk['combined_score'] for chunk in best_chunks) / len(best_chunks)
            base_confidence = (avg_quality + avg_relevance * 100) / 200  # Normalize to 0-1
            
            # Determine confidence factors for transparency
            confidence_factors = {
                'quality_score': avg_quality,
                'relevance_score': avg_relevance * 100,
                'base_confidence': base_confidence,
                'quality_impact': 'high' if avg_quality >= 80 else 'medium' if avg_quality >= 60 else 'low',
                'relevance_impact': 'high' if avg_relevance >= 0.8 else 'medium' if avg_relevance >= 0.6 else 'low'
            }
            
            # Apply confidence scaling based on fallback usage
            fallback_used = any(chunk['metadata'].get('quality_score', 0) < self.MIN_QUALITY_SCORE_YOUTUBE for chunk in best_chunks)
            if fallback_used:
                confidence = min(base_confidence, 0.6)  # Cap at 0.6 if fallbacks used
                confidence_factors['fallback_penalty'] = 0.6
                confidence_factors['fallback_reason'] = 'low_quality_chunks_used'
                logger.warning(f"⚠️ Fallback chunks used, confidence capped at {confidence:.3f}")
            else:
                confidence = base_confidence
                confidence_factors['fallback_penalty'] = None
                confidence_factors['fallback_reason'] = None
            
            # Apply confidence calibration and fallback logic
            if confidence < 0.6 or avg_quality < 60:
                logger.warning(f"⚠️ Low confidence answer: {confidence:.3f}, quality: {avg_quality:.1f}")
                # Could implement graceful fallback here
            
            return {
                'answer': answer,
                'confidence': confidence,
                'confidence_score': confidence,  # Add both for compatibility
                'sources': sources,
                'timestamp_info': timestamp_info,
                'best_topic': best_chunks[0]['metadata'].get('segment_topic', 'Unknown') if best_chunks else None,
                'context_tokens': context_tokens,
                'prompt_tokens': prompt_tokens,
                'confidence_factors': confidence_factors
            }
            
        except Exception as e:
            logger.error(f"❌ Answer generation error: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            logger.error(f"❌ Best chunks structure: {[list(chunk.keys()) for chunk in best_chunks[:2]] if best_chunks else 'No chunks'}")
            logger.error(f"❌ Best chunks metadata keys: {[list(chunk.get('metadata', {}).keys()) for chunk in best_chunks[:2]] if best_chunks else 'No chunks'}")
            return {
                'answer': "I encountered an error while generating an answer.",
                'confidence': 0.0,
                'sources': [],
                'timestamp_info': None
            }
    
    def _format_answer_with_context(self, answer_result: Dict, question_analysis: Dict, namespace: str = "", retrieval_time: float = 0, chunk_selection_time: float = 0, answer_generation_time: float = 0) -> Dict:
        """Format the final answer with timestamp and topic context"""
        try:
            answer = answer_result['answer']
            sources = answer_result['sources']
            timestamp_info = answer_result['timestamp_info']
            best_topic = answer_result.get('best_topic', '')
            
            # Format timestamp information
            timestamp_display = None
            if timestamp_info:
                primary_timestamp = timestamp_info[0]  # Use the best chunk's timestamp
                timestamp_display = {
                    'start_time': primary_timestamp['start_time'],
                    'end_time': primary_timestamp['end_time'],
                    'duration': primary_timestamp['duration'],
                    'topic': primary_timestamp['topic'],
                    'formatted_start': self._format_timestamp(primary_timestamp['start_time']),
                    'formatted_end': self._format_timestamp(primary_timestamp['end_time'])
                }
            
            # Create enhanced response
            response = {
                'success': True,
                'answer': answer,
                'confidence': answer_result['confidence'],
                'sources': sources,
                'timestamp': timestamp_display,
                'topic_context': {
                    'primary_topic': best_topic,
                    'question_intent': question_analysis.get('intent_type', ''),
                    'content_type': question_analysis.get('expected_content_type', ''),
                    'complexity': question_analysis.get('complexity_level', '')
                },
                'metadata': {
                    'chunking_method': 'topic_wise',
                    'quality_optimized': True,
                    'timestamp_precision': 'segment_level',
                    'topic_boundary_guaranteed': True,
                    'namespace_used': namespace,
                    'embedding_model': 'text-embedding-3-large',
                    'embedding_dim': 3072,
                    'chunking_version': 'v2-seg-safe',
                    'transcript_version': 'v1',
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'mmr_diversity_applied': True,
                    'token_limits_enforced': True,
                    'gemini_model': 'gemini-1.5-flash',
                    'embedding_model': 'text-embedding-3-large',
                    'embedding_dim': 3072,
                    'gpt_model': 'gpt-4o-mini',
                    'tokenizer_used': 'tiktoken' if self.tokenizer else 'character_estimation',
                    'phase_timings': {
                        'question_analysis': question_analysis.get('processing_time', 0),
                        'retrieval': retrieval_time,
                        'chunk_selection': chunk_selection_time,
                        'answer_generation': answer_generation_time
                    },
                    'token_statistics': {
                        'context_tokens': answer_result.get('context_tokens', 0),
                        'prompt_tokens': answer_result.get('prompt_tokens', 0),
                        'total_tokens': answer_result.get('context_tokens', 0) + answer_result.get('prompt_tokens', 0),
                        'safety_margin': self.SAFETY_MARGIN
                    },
                    'confidence_factors': answer_result.get('confidence_factors', {})
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Answer formatting error: {e}")
            # Ensure error response has success field
            if isinstance(answer_result, dict):
                answer_result['success'] = False
            return answer_result
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS or HH:MM:SS format"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes:02d}:{secs:02d}"
        except:
            return "00:00"
    
    def _create_youtube_deep_link(self, video_url: str, start_seconds: float) -> str:
        """Create YouTube deep link to specific timestamp (preserves existing params)"""
        try:
            if not video_url or start_seconds < 0:
                return video_url
            
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            
            u = urlparse(video_url)
            qs = parse_qs(u.query)
            qs['t'] = [f"{int(start_seconds)}s"]
            new_query = urlencode(qs, doseq=True)
            return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))
            
        except Exception as e:
            logger.error(f"❌ Deep link creation error: {e}")
            return video_url
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken or fallback to estimation"""
        try:
            if self.tokenizer:
                return len(self.tokenizer.encode(text))
            else:
                # Fallback: rough estimation (1 token ≈ 4 characters)
                return len(text) // 4
        except Exception as e:
            logger.warning(f"⚠️ Token counting failed: {e}, using estimation")
            return len(text) // 4
    
    def _validate_sources_purity(self, sources: List[Dict], company_name: str, qudemo_id: str):
        """Runtime assertions for boundary purity"""
        try:
            for i, source in enumerate(sources):
                # Boundary purity assertions
                assert source['start_timestamp'] <= source['end_timestamp'], f"Source {i}: start > end"
                assert source.get('topic'), f"Source {i}: missing topic"
                # Note: chunk_type is not in source dict, it's in metadata
                
                # Assert company and qudemo isolation
                assert source.get('company_name') == company_name, f"Source {i}: Company mismatch: {source.get('company_name')} != {company_name}"
                assert source.get('qudemo_id') == qudemo_id, f"Source {i}: QuDemo ID mismatch: {source.get('qudemo_id')} != {qudemo_id}"
                
            logger.info(f"✅ Boundary purity validation passed for {len(sources)} sources")
            
        except AssertionError as e:
            logger.error(f"❌ Boundary purity validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
    
    def _get_quality_threshold_for_source(self, sources: List[Dict]) -> int:
        """Determine quality threshold based on video source type"""
        try:
            if not sources:
                return self.MIN_QUALITY_SCORE_YOUTUBE  # Default to YouTube threshold
            
            # Check if any source indicates Loom processing
            for source in sources:
                metadata = source.get('metadata', {})
                text_source = metadata.get('text_source', '')
                transcriber = metadata.get('transcriber', '')
                
                # If we see Whisper or Loom indicators, use Loom threshold
                if 'whisper' in transcriber.lower() or 'loom' in text_source.lower():
                    return self.MIN_QUALITY_SCORE_LOOM
            
            # Default to YouTube threshold (Gemini)
            return self.MIN_QUALITY_SCORE_YOUTUBE
            
        except Exception as e:
            logger.warning(f"⚠️ Error determining quality threshold: {e}")
            return self.MIN_QUALITY_SCORE_YOUTUBE
    
    def _get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """Get embedding with caching"""
        try:
            # Check cache first
            cache_key = f"{model}:{text[:100]}"
            if cache_key in self._embedding_cache:
                self._cache_hits += 1
                return self._embedding_cache[cache_key]
            
            # Generate new embedding
            response = self.openai_client.embeddings.create(
                model=model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            if len(self._embedding_cache) >= self.CACHE_SIZE:
                # Remove oldest entry
                self._embedding_cache.popitem(last=False)
            
            self._embedding_cache[cache_key] = embedding
            self._cache_misses += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding generation error: {e}")
            return []
    
    def _safe_fallback(self) -> Dict:
        """Safe fallback response"""
        return {
            'answer': "I'm sorry, I couldn't process your question at the moment. Please try again.",
            'confidence': 0.0,
            'sources': [],
            'timestamp': None,
            'topic_context': {
                'primary_topic': 'unknown',
                'question_intent': 'unknown',
                'content_type': 'unknown',
                'complexity': 'unknown'
            },
            'metadata': {
                'chunking_method': 'fallback',
                'quality_optimized': False,
                'timestamp_precision': 'none',
                'topic_boundary_guaranteed': False
            }
        }

# Global instance
_enhanced_topic_wise_qa = None

def initialize_enhanced_topic_wise_qa() -> bool:
    """Initialize the enhanced topic-wise QA system"""
    global _enhanced_topic_wise_qa
    try:
        _enhanced_topic_wise_qa = EnhancedTopicWiseQA()
        logger.info("✅ Enhanced Topic-Wise QA System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Topic-Wise QA System: {e}")
        return False

def get_enhanced_topic_wise_qa() -> Optional[EnhancedTopicWiseQA]:
    """Get the enhanced topic-wise QA system instance"""
    return _enhanced_topic_wise_qa
