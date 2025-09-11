#!/usr/bin/env python3
"""
Enhanced Q&A System with Hybrid Search
Combines semantic search with label-based hybrid search for better results
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedHybridQA:
    """Enhanced Q&A system with hybrid search capabilities"""
    
    def __init__(self):
        """Initialize enhanced hybrid Q&A system"""
        # Pinecone Standard Plan - Multiple indexes
        self.indexes = {
            'video': 'qudemo-video-index',
            'knowledge': 'qudemo-knowledge-index',
            'legacy': 'qudemo-index'  # Fallback for existing content
        }
        
        # Hybrid search configuration
        self.hybrid_config = {
            'semantic_weight': 0.7,  # Weight for semantic similarity
            'label_weight': 0.3,     # Weight for label matching
            'min_confidence': 0.1,   # Minimum confidence threshold
            'max_results': 20,       # Maximum results to consider
            'label_boost': 1.2       # Boost factor for label matches
        }
        
        logger.info("✅ Enhanced Hybrid Q&A System initialized")
    
    def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Ask a question using hybrid search approach"""
        try:
            logger.info(f"❓ Hybrid Q&A question for {company_name} qudemo {qudemo_id}: {question}")
            
            # Get question embedding
            question_embedding = self._get_embedding(question)
            
            # Extract keywords and labels from question
            question_keywords = self._extract_question_keywords(question)
            question_labels = self._extract_question_labels(question)
            
            # Search video transcripts with hybrid approach
            video_result = self._search_video_transcripts_hybrid(
                question, question_embedding, question_keywords, question_labels,
                company_name, qudemo_id
            )
            
            # Search knowledge sources with hybrid approach
            knowledge_result = self._search_knowledge_sources_hybrid(
                question, question_embedding, question_keywords, question_labels,
                company_name, qudemo_id
            )
            
            # Select the best answer intelligently
            final_answer = self._select_best_answer_hybrid(
                video_result, knowledge_result, question, question_keywords, question_labels
            )
            
            return final_answer
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid ask_question: {e}")
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
                'estimated_time': '1 minute',
                'search_method': 'hybrid_fallback'
            }
    
    def _search_video_transcripts_hybrid(
        self, 
        question: str, 
        question_embedding: List[float],
        question_keywords: List[str],
        question_labels: List[str],
        company_name: str, 
        qudemo_id: str
    ) -> Dict:
        """Search video transcripts using hybrid approach"""
        try:
            logger.info(f"🎬 Hybrid search in video transcripts for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            logger.info(f"🔍 Searching in namespace: {namespace}")
            
            # Try video index first
            try:
                index = pc.Index(self.indexes['video'])
                query_results = index.query(
                    vector=question_embedding,
                    top_k=self.hybrid_config['max_results'],
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=self.hybrid_config['min_confidence']
                )
                
                if query_results.matches:
                    logger.info(f"✅ Found {len(query_results.matches)} matches in video index")
                    return self._process_video_matches_hybrid(
                        query_results, question, question_keywords, question_labels,
                        company_name, qudemo_id
                    )
                
            except Exception as video_error:
                logger.warning(f"⚠️ Video index search failed: {video_error}")
            
            # Fallback to legacy index
            try:
                index = pc.Index(self.indexes['legacy'])
                query_results = index.query(
                    vector=question_embedding,
                    top_k=self.hybrid_config['max_results'],
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=self.hybrid_config['min_confidence']
                )
                
                if query_results.matches:
                    logger.info(f"✅ Found {len(query_results.matches)} matches in legacy index")
                    return self._process_video_matches_hybrid(
                        query_results, question, question_keywords, question_labels,
                        company_name, qudemo_id
                    )
                
            except Exception as legacy_error:
                logger.warning(f"⚠️ Legacy index search failed: {legacy_error}")
            
            # No results found
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'video',
                'sources': [],
                'total_sources': 0,
                'search_method': 'hybrid_video'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid video search: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'video',
                'sources': [],
                'total_sources': 0,
                'search_method': 'hybrid_video_error'
            }
    
    def _process_video_matches_hybrid(
        self, 
        query_results, 
        question: str, 
        question_keywords: List[str],
        question_labels: List[str],
        company_name: str, 
        qudemo_id: str
    ) -> Dict:
        """Process video matches using hybrid scoring"""
        try:
            if not query_results.matches:
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'video',
                    'sources': [],
                    'total_sources': 0,
                    'search_method': 'hybrid_video'
                }
            
            # Calculate hybrid scores for each match
            enhanced_matches = []
            for match in query_results.matches:
                hybrid_score = self._calculate_hybrid_score(
                    match, question_keywords, question_labels
                )
                enhanced_matches.append({
                    'match': match,
                    'hybrid_score': hybrid_score,
                    'original_score': match.score
                })
            
            # Sort by hybrid score
            enhanced_matches.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # Get top matches
            top_matches = enhanced_matches[:5]  # Top 5 matches
            
            if not top_matches or top_matches[0]['hybrid_score'] < 0.3:
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'video',
                    'sources': [],
                    'total_sources': 0,
                    'search_method': 'hybrid_video'
                }
            
            # Process top matches
            relevant_chunks = []
            chunk_texts = []
            earliest_start_time = float('inf')
            latest_end_time = 0
            video_url = None
            
            # Process top matches - use topic-wise chunking metadata for filtering
            logger.info(f"🔍 Processing {len(top_matches)} semantic matches")
            
            # Filter for specific topic content using topic metadata
            is_disqualified_question = 'disqualified' in question.lower() or 'unqualified' in question.lower()
            is_qualified_question = 'qualified' in question.lower() and not is_disqualified_question
            
            for enhanced_match in top_matches:
                match = enhanced_match['match']
                metadata = match.metadata
                chunk_text = metadata.get('text', '')
                
                if chunk_text and len(chunk_text) > 20:
                    # Check topic metadata for specific topic content
                    segment_topic = metadata.get('segment_topic', '').lower()
                    parent_segment_topic = metadata.get('parent_segment_topic', '').lower()
                    
                    # For disqualified questions, only include chunks with disqualified topic
                    if is_disqualified_question:
                        if not ('disqualified' in segment_topic or 'disqualified' in parent_segment_topic):
                            logger.info(f"🚫 Filtered out chunk (not disqualified topic): {segment_topic or parent_segment_topic}")
                            continue
                    
                    # For qualified questions, only include chunks with qualified topic (exclude disqualified)
                    elif is_qualified_question:
                        if ('disqualified' in segment_topic or 'disqualified' in parent_segment_topic):
                            logger.info(f"🚫 Filtered out chunk (disqualified topic for qualified question): {segment_topic or parent_segment_topic}")
                            continue
                        elif not ('qualified' in segment_topic or 'qualified' in parent_segment_topic):
                            logger.info(f"🚫 Filtered out chunk (not qualified topic): {segment_topic or parent_segment_topic}")
                            continue
                    
                    # Remove timestamps from individual chunks
                    chunk_text = re.sub(r'\[\d{1,2}:\d{2}\]', '', chunk_text).strip()
                    chunk_text = re.sub(r'\s+', ' ', chunk_text)
                    chunk_text = chunk_text.strip()
                    
                    if len(chunk_text) > 20 and chunk_text not in chunk_texts:
                        chunk_texts.append(chunk_text)
                        relevant_chunks.append({
                            'text': chunk_text,
                            'score': enhanced_match['hybrid_score'],
                            'original_score': enhanced_match['original_score'],
                            'metadata': metadata
                        })
                        
                        # Track timestamps from all selected chunks
                        start_time = metadata.get('start_timestamp', 0)
                        end_time = metadata.get('end_timestamp', 0)
                        
                        # Log chunk selection and timestamps
                        logger.info(f"✅ Selected chunk: {start_time}s-{end_time}s (score: {enhanced_match['hybrid_score']:.3f})")
                        logger.info(f"   Topic: {segment_topic or parent_segment_topic}")
                        logger.info(f"   Text preview: {chunk_text[:100]}...")
                        
                        if start_time > 0 and start_time < earliest_start_time:
                            earliest_start_time = start_time
                        if end_time > 0 and end_time > latest_end_time:
                            latest_end_time = end_time
                        
                        # Get video URL from first chunk
                        if not video_url:
                            video_url = metadata.get('video_url', '') or metadata.get('url', '')
            
            # Generate proper answer using GPT instead of just concatenating
            if not relevant_chunks:
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'video',
                    'sources': [],
                    'total_sources': 0,
                    'search_method': 'hybrid_video'
                }
            
            # Generate concise, sales-focused answer
            answer = self._generate_sales_focused_answer(question, relevant_chunks)
            
            # Calculate final score
            final_score = sum(c['score'] for c in relevant_chunks) / len(relevant_chunks)
            
            # Log final timestamp calculation
            final_start = earliest_start_time if earliest_start_time != float('inf') else 0
            final_end = latest_end_time
            
            logger.info(f"🎯 Final timestamps: {final_start}s - {final_end}s")
            logger.info(f"📊 Selected {len(relevant_chunks)} chunks out of {len(top_matches)} total matches")
            
            return {
                'success': True,
                'answer': answer,
                'score': final_score,
                'source': 'video',
                'start': final_start,
                'end': final_end,
                'video_url': video_url,
                'sources': relevant_chunks,
                'total_sources': len(relevant_chunks),
                'search_method': 'hybrid_video',
                'hybrid_scores': [c['score'] for c in relevant_chunks]
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing video matches hybrid: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'video',
                'sources': [],
                'total_sources': 0,
                'search_method': 'hybrid_video_error'
            }
    
    def _generate_sales_focused_answer(self, question: str, relevant_chunks: List[Dict]) -> str:
        """Generate a concise, sales-focused answer from relevant chunks"""
        try:
            import openai
            
            # Use all relevant chunks (semantic search already found the right ones)
            context_parts = []
            filtered_chunks = relevant_chunks[:5]  # Use top 5 chunks
            
            for i, chunk in enumerate(filtered_chunks):
                context_parts.append(f"Chunk {i+1}: {chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Create sales-focused prompt
            prompt = f"""You are a sales expert explaining to a client. Answer the question concisely and professionally.

Question: {question}

Context from video:
{context}

Instructions:
1. Answer the question based on the provided context
2. Be concise - maximum 3-4 sentences
3. Use bullet points or numbered steps when appropriate
4. Write like a sales professional explaining to a client
5. Focus on actionable steps and key benefits
6. If the context doesn't contain the specific information asked, say "I don't see specific information about [topic] in the available content"
7. IMPORTANT: If asked about "disqualified leads" but the context only shows "qualified leads" workflow, explain that the video demonstrates the qualified leads process and mention that the same approach can be applied to disqualified leads

Answer:"""

            # Generate answer using GPT (OpenAI v1.0+ API)
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sales expert providing concise, professional answers to clients."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Clean up the answer
            answer = answer.replace("Answer:", "").strip()
            answer = answer.replace("Based on the context:", "").strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error generating sales-focused answer: {e}")
            # Fallback to simple concatenation
            return " ".join([chunk['text'] for chunk in relevant_chunks[:2]])
    
    def _calculate_hybrid_score(
        self, 
        match, 
        question_keywords: List[str], 
        question_labels: List[str]
    ) -> float:
        """Calculate hybrid score combining semantic similarity and label matching with proper normalization"""
        try:
            metadata = match.metadata
            original_score = match.score
            
            # Start with semantic score (already normalized [0,1])
            hybrid_score = original_score * self.hybrid_config['semantic_weight']
            
            # Add topic metadata matching bonus (for questions like "disqualified leads")
            chunk_text = metadata.get('text', '').lower()
            segment_topic = metadata.get('segment_topic', '').lower()
            parent_segment_topic = metadata.get('parent_segment_topic', '').lower()
            topic_bonus = 0
            
            # Check topic metadata first (more reliable than text content)
            if 'disqualified' in segment_topic or 'disqualified' in parent_segment_topic:
                if 'lead' in segment_topic or 'lead' in parent_segment_topic:
                    topic_bonus += 0.8  # Very strong bonus for exact topic match
                else:
                    topic_bonus += 0.6  # Strong bonus for disqualified topic
            elif 'qualified' in segment_topic or 'qualified' in parent_segment_topic:
                if 'lead' in segment_topic or 'lead' in parent_segment_topic:
                    topic_bonus += 0.8  # Very strong bonus for exact topic match
                else:
                    topic_bonus += 0.6  # Strong bonus for qualified topic
            elif 'unqualified' in segment_topic or 'unqualified' in parent_segment_topic:
                topic_bonus += 0.5  # Strong bonus for unqualified topic
            
            # Fallback to text content if no topic metadata
            elif 'disqualified' in chunk_text and 'lead' in chunk_text:
                topic_bonus += 0.5  # Very strong bonus for exact topic match
            elif 'disqualified' in chunk_text or 'unqualified' in chunk_text:
                topic_bonus += 0.3  # Strong bonus for related terms
            elif 'disqualify' in chunk_text or 'disqualifying' in chunk_text:
                topic_bonus += 0.2  # Medium bonus for action terms
            elif 'qualified' in chunk_text and 'lead' in chunk_text:
                topic_bonus -= 0.3  # Strong penalty for opposite topic
            elif 'qualify' in chunk_text and 'lead' in chunk_text:
                topic_bonus -= 0.2  # Medium penalty for opposite action
            
            hybrid_score += topic_bonus
            
            # Add label matching bonus (normalized to [0,1])
            chunk_labels = metadata.get('labels', [])
            if chunk_labels and question_labels:
                label_matches = 0
                for chunk_label in chunk_labels:
                    for question_label in question_labels:
                        if self._labels_match(chunk_label, question_label):
                            label_matches += 1
                
                if label_matches > 0:
                    # Normalize label match ratio to [0,1]
                    label_match_ratio = label_matches / len(question_labels)
                    label_bonus = label_match_ratio * self.hybrid_config['label_weight']
                    hybrid_score += label_bonus * self.hybrid_config['label_boost']
            
            # Add keyword matching bonus (normalized to [0,1])
            keyword_matches = sum(1 for keyword in question_keywords if keyword.lower() in chunk_text)
            if keyword_matches > 0 and question_keywords:
                # Normalize keyword match ratio to [0,1]
                keyword_match_ratio = keyword_matches / len(question_keywords)
                keyword_bonus = keyword_match_ratio * 0.1  # Small bonus
                hybrid_score += keyword_bonus
            
            # Add quality score bonus (normalized to [0,1])
            quality_score = metadata.get('quality_score', 50)
            if quality_score > 70:
                # Normalize quality score to [0,1] and apply small bonus
                quality_normalized = (quality_score - 70) / 30  # [0,1] for scores 70-100
                quality_bonus = quality_normalized * 0.03  # Small bonus
                hybrid_score += quality_bonus
            
            return min(1.0, hybrid_score)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating hybrid score: {e}")
            return match.score  # Fallback to original score
    
    def _labels_match(self, chunk_label: str, question_label: str) -> bool:
        """Check if labels match (with fuzzy matching and normalization)"""
        try:
            # Normalize labels: lowercase, strip punctuation, lemmatize
            chunk_label_normalized = self._normalize_label(chunk_label)
            question_label_normalized = self._normalize_label(question_label)
            
            # Exact match
            if chunk_label_normalized == question_label_normalized:
                return True
            
            # Partial match
            if chunk_label_normalized in question_label_normalized or question_label_normalized in chunk_label_normalized:
                return True
            
            # Word overlap with fuzzy matching
            chunk_words = set(chunk_label_normalized.split())
            question_words = set(question_label_normalized.split())
            overlap = len(chunk_words.intersection(question_words))
            
            # Require at least 50% word overlap
            min_words = min(len(chunk_words), len(question_words))
            return overlap > 0 and overlap >= min_words * 0.5
            
        except Exception as e:
            logger.error(f"❌ Error in label matching: {e}")
            return False
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label for better matching"""
        try:
            import re
            
            # Convert to lowercase
            normalized = label.lower()
            
            # Remove punctuation
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            # Remove extra whitespace
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # Simple lemmatization for common cases
            lemmatization_map = {
                'tutorials': 'tutorial',
                'demos': 'demo',
                'demonstrations': 'demonstration',
                'explanations': 'explanation',
                'setups': 'setup',
                'installations': 'installation',
                'configurations': 'configuration',
                'troubleshootings': 'troubleshooting',
                'problems': 'problem',
                'errors': 'error',
                'issues': 'issue',
                'fixes': 'fix'
            }
            
            for plural, singular in lemmatization_map.items():
                normalized = normalized.replace(plural, singular)
            
            return normalized
            
        except Exception as e:
            logger.error(f"❌ Error normalizing label: {e}")
            return label.lower()
    
    def _search_knowledge_sources_hybrid(
        self, 
        question: str, 
        question_embedding: List[float],
        question_keywords: List[str],
        question_labels: List[str],
        company_name: str, 
        qudemo_id: str
    ) -> Dict:
        """Search knowledge sources using hybrid approach"""
        try:
            logger.info(f"📚 Hybrid search in knowledge sources for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Try knowledge index first
            try:
                index = pc.Index(self.indexes['knowledge'])
                query_results = index.query(
                    vector=question_embedding,
                    top_k=self.hybrid_config['max_results'],
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=self.hybrid_config['min_confidence']
                )
                
                if query_results.matches:
                    logger.info(f"✅ Found {len(query_results.matches)} matches in knowledge index")
                    return self._process_knowledge_matches_hybrid(
                        query_results, question, question_keywords, question_labels
                    )
                
            except Exception as knowledge_error:
                logger.warning(f"⚠️ Knowledge index search failed: {knowledge_error}")
            
            # Fallback to legacy index
            try:
                index = pc.Index(self.indexes['legacy'])
                query_results = index.query(
                    vector=question_embedding,
                    top_k=self.hybrid_config['max_results'],
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=self.hybrid_config['min_confidence']
                )
                
                if query_results.matches:
                    logger.info(f"✅ Found {len(query_results.matches)} matches in legacy index")
                    return self._process_knowledge_matches_hybrid(
                        query_results, question, question_keywords, question_labels
                    )
                
            except Exception as legacy_error:
                logger.warning(f"⚠️ Legacy index search failed: {legacy_error}")
            
            # No results found
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'knowledge',
                'sources': [],
                'total_sources': 0,
                'search_method': 'hybrid_knowledge'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid knowledge search: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'knowledge',
                'sources': [],
                'total_sources': 0,
                'search_method': 'hybrid_knowledge_error'
            }
    
    def _process_knowledge_matches_hybrid(
        self, 
        query_results, 
        question: str, 
        question_keywords: List[str],
        question_labels: List[str]
    ) -> Dict:
        """Process knowledge matches using hybrid scoring"""
        try:
            if not query_results.matches:
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'knowledge',
                    'sources': [],
                    'total_sources': 0,
                    'search_method': 'hybrid_knowledge'
                }
            
            # Calculate hybrid scores for each match
            enhanced_matches = []
            for match in query_results.matches:
                hybrid_score = self._calculate_hybrid_score(
                    match, question_keywords, question_labels
                )
                enhanced_matches.append({
                    'match': match,
                    'hybrid_score': hybrid_score,
                    'original_score': match.score
                })
            
            # Sort by hybrid score
            enhanced_matches.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # Get top matches
            top_matches = enhanced_matches[:3]  # Top 3 matches for knowledge
            
            if not top_matches or top_matches[0]['hybrid_score'] < 0.3:
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'knowledge',
                    'sources': [],
                    'total_sources': 0,
                    'search_method': 'hybrid_knowledge'
                }
            
            # Process top matches
            relevant_chunks = []
            chunk_texts = []
            
            for enhanced_match in top_matches:
                match = enhanced_match['match']
                metadata = match.metadata
                chunk_text = metadata.get('text', '')
                
                if chunk_text and len(chunk_text) > 20:
                    chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
                    
                    if len(chunk_text) > 20 and chunk_text not in chunk_texts:
                        chunk_texts.append(chunk_text)
                        relevant_chunks.append({
                            'text': chunk_text,
                            'score': enhanced_match['hybrid_score'],
                            'original_score': enhanced_match['original_score'],
                            'metadata': metadata
                        })
            
            # Combine unique chunk texts
            combined_text = ' '.join(chunk_texts)
            
            if not relevant_chunks:
                return {
                    'success': False,
                    'answer': None,
                    'score': 0,
                    'source': 'knowledge',
                    'sources': [],
                    'total_sources': 0,
                    'search_method': 'hybrid_knowledge'
                }
            
            # Calculate final score
            final_score = sum(c['score'] for c in relevant_chunks) / len(relevant_chunks)
            
            return {
                'success': True,
                'answer': combined_text,
                'score': final_score,
                'source': 'knowledge',
                'start': 0,
                'end': 0,
                'video_url': None,
                'sources': relevant_chunks,
                'total_sources': len(relevant_chunks),
                'search_method': 'hybrid_knowledge',
                'hybrid_scores': [c['score'] for c in relevant_chunks]
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing knowledge matches hybrid: {e}")
            return {
                'success': False,
                'answer': None,
                'score': 0,
                'source': 'knowledge',
                'sources': [],
                'total_sources': 0,
                'search_method': 'hybrid_knowledge_error'
            }
    
    def _select_best_answer_hybrid(
        self, 
        video_result: Dict, 
        knowledge_result: Dict, 
        question: str,
        question_keywords: List[str],
        question_labels: List[str]
    ) -> Dict:
        """Select the best answer using hybrid approach"""
        try:
            # Determine which result is better
            video_score = video_result.get('score', 0)
            knowledge_score = knowledge_result.get('score', 0)
            
            # Prefer video if it has a good score and timestamps
            if video_result.get('success') and video_score > 0.3:
                # Check if video has timestamps (including 00:00 valid hits)
                has_timestamps = ('start' in video_result) and ('end' in video_result)
                if has_timestamps:
                    return {
                        'success': True,
                        'answer': video_result['answer'],
                        'start': video_result.get('start', 0),
                        'end': video_result.get('end', 0),
                        'video_url': video_result.get('video_url'),
                        'sources': video_result.get('sources', []),
                        'total_sources': video_result.get('total_sources', 0),
                        'search_score': video_score,
                        'content_types_found': ['video'],
                        'difficulty_level': 'beginner',
                        'estimated_time': '1 minute',
                        'answer_source': 'video',
                        'search_method': video_result.get('search_method', 'hybrid_video'),
                        'hybrid_scores': video_result.get('hybrid_scores', [])
                    }
            
            # Use knowledge if it has a better score
            if knowledge_result.get('success') and knowledge_score > video_score:
                return {
                    'success': True,
                    'answer': knowledge_result['answer'],
                    'start': 0,
                    'end': 0,
                    'video_url': None,
                    'sources': knowledge_result.get('sources', []),
                    'total_sources': knowledge_result.get('total_sources', 0),
                    'search_score': knowledge_score,
                    'content_types_found': ['knowledge'],
                    'difficulty_level': 'beginner',
                    'estimated_time': '1 minute',
                    'answer_source': 'knowledge',
                    'search_method': knowledge_result.get('search_method', 'hybrid_knowledge'),
                    'hybrid_scores': knowledge_result.get('hybrid_scores', [])
                }
            
            # Use video as fallback if available
            if video_result.get('success'):
                return {
                    'success': True,
                    'answer': video_result['answer'],
                    'start': video_result.get('start', 0),
                    'end': video_result.get('end', 0),
                    'video_url': video_result.get('video_url'),
                    'sources': video_result.get('sources', []),
                    'total_sources': video_result.get('total_sources', 0),
                    'search_score': video_score,
                    'content_types_found': ['video'],
                    'difficulty_level': 'beginner',
                    'estimated_time': '1 minute',
                    'answer_source': 'video',
                    'search_method': video_result.get('search_method', 'hybrid_video'),
                    'hybrid_scores': video_result.get('hybrid_scores', [])
                }
            
            # No good results found
            return {
                'success': False,
                'answer': "I couldn't find a relevant answer to your question. Please try rephrasing or asking about a different topic.",
                'start': 0,
                'end': 0,
                'video_url': None,
                'sources': [],
                'total_sources': 0,
                'search_score': 0,
                'content_types_found': [],
                'difficulty_level': 'beginner',
                'estimated_time': '1 minute',
                'answer_source': 'none',
                'search_method': 'hybrid_no_results'
            }
            
        except Exception as e:
            logger.error(f"❌ Error selecting best answer hybrid: {e}")
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
                'estimated_time': '1 minute',
                'answer_source': 'error',
                'search_method': 'hybrid_error'
            }
    
    def _extract_question_keywords(self, question: str) -> List[str]:
        """Extract keywords from question for hybrid search"""
        try:
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
            
            # Filter out common words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an', 'what', 'how', 'when', 'where', 'why', 'who'}
            
            keywords = [word for word in words if word not in stop_words]
            return keywords[:5]  # Top 5 keywords
            
        except Exception as e:
            logger.error(f"❌ Error extracting question keywords: {e}")
            return []
    
    def _extract_question_labels(self, question: str) -> List[str]:
        """Extract potential labels from question for hybrid search"""
        try:
            # Simple label extraction based on common patterns
            labels = []
            question_lower = question.lower()
            
            # Common label patterns
            label_patterns = {
                'tutorial': ['tutorial', 'how to', 'guide', 'step by step'],
                'demo': ['demo', 'demonstration', 'example', 'show'],
                'explanation': ['explain', 'what is', 'definition', 'meaning'],
                'setup': ['setup', 'install', 'configure', 'setup'],
                'troubleshooting': ['problem', 'error', 'issue', 'fix', 'troubleshoot']
            }
            
            for label, patterns in label_patterns.items():
                if any(pattern in question_lower for pattern in patterns):
                    labels.append(label)
            
            return labels
            
        except Exception as e:
            logger.error(f"❌ Error extracting question labels: {e}")
            return []
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API"""
        try:
            import openai
            
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"❌ Error getting embedding: {e}")
            # Return a dummy embedding
            return [0.0] * 3072

# Global instance
_hybrid_qa = None

def initialize_hybrid_qa() -> bool:
    """Initialize global hybrid Q&A system"""
    global _hybrid_qa
    try:
        _hybrid_qa = EnhancedHybridQA()
        logger.info("✅ Enhanced Hybrid Q&A System initialized globally")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Enhanced Hybrid Q&A System: {e}")
        return False

def get_hybrid_qa() -> Optional[EnhancedHybridQA]:
    """Get global hybrid Q&A system instance"""
    return _hybrid_qa
