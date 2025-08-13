#!/usr/bin/env python3
"""
Enhanced Q&A Module
Combines video transcripts, scraped data, and documents for comprehensive answers
"""

import logging
import re
from typing import List, Dict, Optional
import openai
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedQA:
    def __init__(self, openai_api_key: str):
        """Initialize enhanced Q&A system"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        
    def query_all_sources(self, company_name: str, question: str, top_k: int = 10) -> List[Dict]:
        """Query all knowledge sources (video, website, document) with Q&A prioritization"""
        try:
            from qa_system import query_pinecone
            
            logger.info(f"🔍 Querying all knowledge sources for {company_name}")
            
            # Create question embedding
            q_embedding = self.openai_client.embeddings.create(
                input=[question],
                model="text-embedding-3-small",
                timeout=15
            ).data[0].embedding
            
            # Query Pinecone for all sources
            matches = query_pinecone(company_name, q_embedding, top_k=top_k * 2)  # Get more matches to filter
            
            if not matches:
                logger.warning(f"⚠️ No matches found for {company_name}")
                return []
            
            # Separate Q&A pairs from regular chunks
            qa_matches = []
            regular_matches = []
            
            for match in matches:
                metadata = match["metadata"]
                chunk_type = metadata.get('chunk_type', 'regular')
                
                if chunk_type == 'qa_pair':
                    qa_matches.append(match)
                else:
                    regular_matches.append(match)
            
            logger.info(f"📊 Found {len(qa_matches)} Q&A pairs and {len(regular_matches)} regular chunks")
            
            # Prioritize Q&A pairs if available (isolated from video transcripts)
            if qa_matches:
                logger.info(f"🎯 Using pre-generated Q&A pairs for answer")
                # Convert Q&A matches to the expected format
                qa_chunks = []
                for match in qa_matches[:top_k]:
                    metadata = match["metadata"]
                    content_type = metadata.get('content_type', 'regular')
                    
                    # Only use Q&A pairs (isolated from video transcripts)
                    if content_type == 'qa_pair':
                        qa_chunk = {
                            'text': metadata.get('answer', ''),
                            'question': metadata.get('question', ''),
                            'question_type': metadata.get('question_type', 'general'),
                            'difficulty': metadata.get('difficulty', 'basic'),
                            'source_type': metadata.get('source_type', 'unknown'),
                            'source_info': metadata.get('source_info', 'unknown'),
                            'qa_id': metadata.get('qa_id', ''),
                            'chunk_type': 'qa_pair',
                            'content_type': 'qa_pair',
                            'similarity_score': match.get('score', 0.0)
                        }
                        qa_chunks.append(qa_chunk)
                
                if qa_chunks:
                    # Log source distribution
                    source_counts = {}
                    for chunk in qa_chunks:
                        source_type = chunk.get('source_type', 'unknown')
                        source_counts[source_type] = source_counts.get(source_type, 0) + 1
                    
                    logger.info(f"📊 Q&A Source distribution: {source_counts}")
                    return qa_chunks
                else:
                    logger.info(f"📄 No Q&A pairs found, using regular chunks")
            else:
                logger.info(f"📄 No Q&A pairs found, using regular chunks")
                # Fall back to regular chunks
                chunks = [m["metadata"] for m in regular_matches[:top_k]]
                
                # Log source distribution
                source_counts = {}
                for chunk in chunks:
                    source_type = chunk.get('source_type', 'unknown')
                    source_counts[source_type] = source_counts.get(source_type, 0) + 1
                
                logger.info(f"📊 Regular chunk source distribution: {source_counts}")
                return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to query all sources: {e}")
            return []
    
    def rank_sources_by_relevance(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Rank chunks by relevance using GPT-3.5"""
        try:
            if not chunks:
                return []
            
            # Prepare chunks for ranking
            rerank_prompt = f"Question: {question}\n\nHere are the knowledge chunks:\n"
            for i, chunk in enumerate(chunks):
                source_type = chunk.get('source_type', 'unknown')
                source_info = chunk.get('source_info', 'unknown')
                text = chunk.get('text', '')[:500].strip().replace('\n', ' ')
                rerank_prompt += f"{i+1}. [{source_type}] {source_info}\n{text}\n\n"
            
            rerank_prompt += "Which chunk is most relevant to the question above? Just give the number."
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": rerank_prompt}],
                timeout=20
            )
            
            response_text = response.choices[0].message.content
            numbers = re.findall(r"\d+", response_text)
            
            if numbers:
                best_index = int(numbers[0]) - 1
                if 0 <= best_index < len(chunks):
                    best_chunk = chunks[best_index]
                    logger.info(f"🏅 GPT-3.5 ranked chunk #{best_index+1} as most relevant (source: {best_chunk.get('source_type', 'unknown')})")
                    return [best_chunk] + [c for i, c in enumerate(chunks) if i != best_index]
            
            logger.warning(f"⚠️ Reranking failed, using original order")
            return chunks
            
        except Exception as e:
            logger.warning(f"⚠️ Source ranking failed: {e}")
            return chunks
    
    def determine_answer_source(self, chunks: List[Dict]) -> Dict:
        """Determine the best answer source based on relevance and type"""
        try:
            if not chunks:
                return {'source_type': 'none', 'confidence': 0.0}
            
            best_chunk = chunks[0]
            source_type = best_chunk.get('source_type', 'unknown')
            
            # Calculate confidence based on source type and position
            confidence = 1.0 - (0.1 * len(chunks))  # Higher confidence for fewer, more relevant results
            
            # Adjust confidence based on source type
            if source_type == 'video':
                confidence *= 1.0  # Video has highest priority
            elif source_type == 'website':
                confidence *= 0.9  # Website content is good
            elif source_type == 'document':
                confidence *= 0.8  # Document content is good but less dynamic
            
            return {
                'source_type': source_type,
                'confidence': min(confidence, 1.0),
                'best_chunk': best_chunk
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to determine answer source: {e}")
            return {'source_type': 'unknown', 'confidence': 0.0}
    
    def generate_combined_answer(self, question: str, chunks: List[Dict], company_name: str) -> Dict:
        """Generate answer combining multiple knowledge sources with Q&A optimization"""
        try:
            if not chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ensure knowledge sources have been processed.",
                    "sources": [],
                    "source_type": "none"
                }
            
            # Check if we have Q&A pairs
            has_qa_pairs = any(chunk.get('chunk_type') == 'qa_pair' for chunk in chunks)
            
            if has_qa_pairs:
                # Use Q&A pairs for direct answer
                logger.info(f"🎯 Using pre-generated Q&A pairs for direct answer")
                return self._generate_qa_based_answer(question, chunks, company_name)
            else:
                # Use traditional chunk-based answer generation
                logger.info(f"📄 Using traditional chunk-based answer generation")
                return self._generate_chunk_based_answer(question, chunks, company_name)
                
        except Exception as e:
            logger.error(f"❌ Failed to generate combined answer: {e}")
            return {
                "answer": "Sorry, I encountered an error while generating the answer. Please try again.",
                "sources": [],
                "source_type": "error"
            }
    
    def _generate_qa_based_answer(self, question: str, chunks: List[Dict], company_name: str) -> Dict:
        """Generate answer using pre-generated Q&A pairs"""
        try:
            # Get the best Q&A pair
            best_qa = chunks[0]  # Already sorted by relevance
            
            # Extract answer and metadata
            answer = best_qa.get('text', '')
            question_type = best_qa.get('question_type', 'general')
            difficulty = best_qa.get('difficulty', 'basic')
            source_type = best_qa.get('source_type', 'unknown')
            source_info = best_qa.get('source_info', 'unknown')
            similarity_score = best_qa.get('similarity_score', 0.0)
            
            # If similarity is low, enhance the answer
            if similarity_score < 0.7:
                logger.info(f"⚠️ Low similarity score ({similarity_score:.2f}), enhancing answer")
                enhanced_answer = self._enhance_qa_answer(question, answer, company_name)
            else:
                enhanced_answer = answer
            
            # Prepare sources
            sources = [f"{source_type.title()} ({source_info})"]
            
            # Add confidence based on similarity score
            confidence = min(similarity_score * 1.2, 1.0)  # Boost confidence slightly
            
            logger.info(f"✅ Generated Q&A-based answer (confidence: {confidence:.2f})")
            
            return {
                "answer": enhanced_answer,
                "sources": sources,
                "source_type": source_type,
                "confidence": confidence,
                "answer_method": "qa_pair",
                "question_type": question_type,
                "difficulty": difficulty
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Q&A-based answer: {e}")
            return self._generate_chunk_based_answer(question, chunks, company_name)
    
    def _enhance_qa_answer(self, question: str, answer: str, company_name: str) -> str:
        """Enhance a Q&A answer with additional context"""
        try:
            system_prompt = f"""You are a helpful assistant for {company_name}. 
Enhance the provided answer to better address the user's question while maintaining accuracy.

Guidelines:
1. Keep the core information from the original answer
2. Add relevant context if needed
3. Make the answer more comprehensive and helpful
4. Maintain a professional tone
5. Keep the answer concise (100-200 words)"""

            user_prompt = f"User Question: {question}\n\nOriginal Answer: {answer}\n\nEnhanced Answer:"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            enhanced_answer = response.choices[0].message.content.strip()
            return enhanced_answer
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to enhance answer: {e}")
            return answer
    
    def _generate_chunk_based_answer(self, question: str, chunks: List[Dict], company_name: str) -> Dict:
        """Generate answer using traditional chunk-based approach"""
        # Determine answer source
        source_info = self.determine_answer_source(chunks)
        source_type = source_info['source_type']
        confidence = source_info['confidence']
        
        # Prepare context from all chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:6]):  # Use top 6 chunks
            source_type_chunk = chunk.get('source_type', 'unknown')
            source_info_chunk = chunk.get('source_info', 'unknown')
            text = chunk.get('text', '')[:800].strip()
            
            context_parts.append(f"[{source_type_chunk.upper()}] {source_info_chunk}:\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer based on source type
        if source_type == 'video' and confidence > 0.5:
            # Video answer with timestamp
            system_prompt = (
                f"You are a product demo assistant for {company_name}. "
                "Answer using the provided video content. If the content contains relevant info, "
                "provide a comprehensive answer. Return ONLY the following markdown sections:\n"
                "- TL;DR: <one concise line>\n"
                "- Steps:\n  1. <step>\n  2. <step>\n  3. <step>\n"
                "- Notes:\n  - <note>\n  - <note>\n"
                "Keep it crisp (roughly 120-180 words total)."
            )
        else:
            # Text-based answer (website/document)
            system_prompt = (
                f"You are a product knowledge assistant for {company_name}. "
                "Answer using the provided knowledge content. Provide a comprehensive answer. "
                "Return ONLY the following markdown sections:\n"
                "- Answer: <comprehensive answer>\n"
                "- Key Points:\n  - <point>\n  - <point>\n"
                "- Sources: <list sources used>\n"
                "Keep it informative (roughly 150-250 words total)."
            )
        
        user_prompt = (
            "Knowledge Content:\n" + context + "\n\n" +
            "Question: " + question + "\n\n" +
            "Respond using the exact structure above."
        )
        
        completion = self.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=30,
            max_tokens=400,
            temperature=0.2
        )
        
        raw_answer = completion.choices[0].message.content
        
        # Extract sources
        sources = []
        for chunk in chunks[:3]:  # Top 3 sources
            source_type_chunk = chunk.get('source_type', 'unknown')
            source_info_chunk = chunk.get('source_info', 'unknown')
            
            if source_type_chunk == 'video':
                video_url = chunk.get('video_url', '')
                if video_url:
                    if "youtu.be" in video_url or "youtube.com" in video_url:
                        video_id = video_url.split("?")[0].split("/")[-1]
                        sources.append(f"Video ({video_id})")
                    else:
                        sources.append(f"Video ({video_url.split('/')[-1]})")
            elif source_type_chunk == 'website':
                url = chunk.get('url', source_info_chunk)
                sources.append(f"Website ({url})")
            elif source_type_chunk == 'document':
                filename = chunk.get('filename', source_info_chunk)
                sources.append(f"Document ({filename})")
        
        # Log source attribution
        logger.info(f"📝 Answer generated from {source_type} source (confidence: {confidence:.2f})")
        logger.info(f"📊 Sources used: {sources}")
        
        return {
            "answer": raw_answer,
            "sources": sources,
            "source_type": source_type,
            "confidence": confidence,
            "answer_method": "chunk_based"
        }
    
    def answer_question_enhanced(self, company_name: str, question: str) -> Dict:
        """Enhanced Q&A that combines all knowledge sources"""
        try:
            logger.info(f"🤖 Enhanced Q&A for {company_name}: {question}")
            
            # Query all knowledge sources
            chunks = self.query_all_sources(company_name, question)
            
            if not chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ensure knowledge sources have been processed.",
                    "sources": [],
                    "source_type": "none"
                }
            
            # Rank sources by relevance
            ranked_chunks = self.rank_sources_by_relevance(question, chunks)
            
            # Generate combined answer
            result = self.generate_combined_answer(question, ranked_chunks, company_name)
            
            # Add timestamp info for video sources
            if result.get('source_type') == 'video' and ranked_chunks:
                best_chunk = ranked_chunks[0]
                start = float(best_chunk.get('start', 0.0))
                end = float(best_chunk.get('end', 0.0))
                video_url = best_chunk.get('video_url', '')
                
                if start > 0.0 or end > 0.0:
                    result['video_timestamp'] = {
                        'start': start,
                        'end': end,
                        'video_url': video_url
                    }
                    logger.info(f"⏰ Video timestamp: {start:.2f}s - {end:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced Q&A failed: {e}")
            return {
                "answer": "Sorry, I encountered an error while processing your question. Please try again.",
                "sources": [],
                "source_type": "error"
            }
    
    def get_source_content(self, source_id: str, company_name: str = None) -> Optional[Dict]:
        """Get knowledge source content from Pinecone for preview"""
        try:
            # Ensure Q&A system is initialized
            from qa_system import initialize_qa_system, default_index_name
            import os
            
            # Initialize if not already done
            initialize_qa_system()
            
            # Get the initialized pc after initialization
            from qa_system import pc
            
            logger.info(f"📄 Getting content for source: {source_id} (company: {company_name})")
            logger.info(f"🔍 Searching for knowledge_source_id: {source_id}")
            
            # Get all indexes
            existing_indexes = [index.name for index in pc.list_indexes()]
            index_name = os.getenv("PINECONE_INDEX", default_index_name)
            
            if index_name not in existing_indexes and existing_indexes:
                index_name = existing_indexes[0]
                logger.info(f"⚠️ Using fallback index: {index_name}")
            
            index = pc.Index(index_name)
            
            # Determine namespace - if company_name is provided, use it
            namespace = None
            if company_name:
                namespace = company_name.lower().replace(' ', '-')
                logger.info(f"🔍 Searching in namespace: {namespace}")
            
            # Search for vectors with this source ID
            dummy_embedding = [0.0] * 1536  # OpenAI embedding dimension
            
            try:
                # Get index stats to check available namespaces
                stats = index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                
                if total_vectors == 0:
                    logger.warning(f"⚠️ No vectors found in index {index_name}")
                    return None
                
                # If namespace is specified, search only in that namespace
                if namespace:
                    try:
                        matches = index.query(
                            vector=dummy_embedding,
                            top_k=min(1000, total_vectors),
                            include_metadata=True,
                            namespace=namespace
                        ).matches
                        logger.info(f"🔍 Found {len(matches)} vectors in namespace '{namespace}'")
                    except Exception as e:
                        logger.warning(f"⚠️ Namespace '{namespace}' not found, searching all namespaces")
                        matches = index.query(
                            vector=dummy_embedding,
                            top_k=min(1000, total_vectors),
                            include_metadata=True
                        ).matches
                else:
                    # Search all namespaces if no specific namespace provided
                    matches = index.query(
                        vector=dummy_embedding,
                        top_k=min(1000, total_vectors),
                        include_metadata=True
                    ).matches
                    logger.info(f"🔍 Found {len(matches)} vectors across all namespaces")
                
                # Filter matches by source_id
                source_matches = []
                logger.info(f"🔍 Filtering {len(matches)} vectors for source_id: {source_id}")
                
                for match in matches:
                    metadata = match.metadata
                    
                    # First, check if we have a direct knowledge_source_id match
                    knowledge_source_id = metadata.get('knowledge_source_id', '')
                    if knowledge_source_id == source_id:
                        source_matches.append(match)
                        logger.info(f"✅ Found direct match with knowledge_source_id: {knowledge_source_id}")
                        continue
                    
                    # Fallback: Extract clean source IDs from URLs
                    source_info = metadata.get('source_info', '')
                    url = metadata.get('url', '')
                    video_url = metadata.get('video_url', '')
                    
                    # Helper function to extract clean source ID
                    def extract_source_id(url_string):
                        if not url_string:
                            return None
                        # Remove query parameters and fragments
                        clean_url = url_string.split('?')[0].split('#')[0]
                        # Extract the last part (filename or ID)
                        parts = clean_url.rstrip('/').split('/')
                        return parts[-1] if parts else None
                    
                    # Extract source IDs
                    source_info_id = extract_source_id(source_info)
                    url_id = extract_source_id(url)
                    video_url_id = extract_source_id(video_url)
                    
                    # Check if any of the extracted IDs match our search
                    if (source_info_id == source_id or 
                        url_id == source_id or 
                        video_url_id == source_id or
                        # Also check if the source_id is contained in any of the URLs
                        source_id in source_info or
                        source_id in url or
                        source_id in video_url):
                        source_matches.append(match)
                
                # If no matches found with knowledge_source_id, try a broader search
                # This is for existing vectors that don't have knowledge_source_id
                if not source_matches and len(matches) > 0:
                    logger.info(f"🔍 No direct matches found, using fallback to return all content...")
                    # Return all Q&A pairs and chunks as a fallback
                    source_matches = matches  # Return all matches
                    logger.info(f"📄 Using fallback: returning {len(source_matches)} vectors")
                elif len(source_matches) == 0 and len(matches) > 0:
                    logger.info(f"🔍 No exact matches found, but {len(matches)} vectors available in namespace")
                    logger.info(f"📄 Using fallback: returning all {len(matches)} vectors")
                    source_matches = matches  # Return all matches as fallback
                
                if not source_matches:
                    logger.warning(f"⚠️ No content found for source: {source_id}")
                    return None
                
                # Extract content from matches
                chunks = []
                qa_pairs = []
                total_words = 0
                total_characters = 0
                
                for match in source_matches:
                    metadata = match.metadata
                    text = metadata.get('text', '')
                    chunk_type = metadata.get('chunk_type', 'regular')
                    
                    # Debug: Log chunk types
                    if chunk_type == 'qa_pair':
                        logger.info(f"🔍 Found Q&A pair: {metadata.get('question', 'N/A')[:50]}...")
                    
                    if chunk_type == 'qa_pair':
                        # This is a Q&A pair - get answer from 'answer' field if text is empty
                        answer = text if text else metadata.get('answer', '')
                        if answer:
                            qa_pair = {
                                'question': metadata.get('question', ''),
                                'answer': answer,
                                'question_type': metadata.get('question_type', 'general'),
                                'difficulty': metadata.get('difficulty', 'basic'),
                                'qa_id': metadata.get('qa_id', ''),
                                'metadata': metadata,
                                'score': match.score
                            }
                            qa_pairs.append(qa_pair)
                            logger.info(f"✅ Added Q&A pair: {qa_pair['question'][:50]}...")
                        else:
                            logger.warning(f"⚠️ Q&A pair has no answer: {metadata.get('question', 'N/A')[:50]}...")
                    elif text:
                        # This is a regular chunk
                        chunks.append({
                            'text': text,
                            'metadata': metadata,
                            'score': match.score
                        })
                        
                        total_words += len(text.split())
                        total_characters += len(text)
                
                # Sort chunks and Q&A pairs by score (highest first)
                chunks.sort(key=lambda x: x['score'], reverse=True)
                qa_pairs.sort(key=lambda x: x['score'], reverse=True)
                
                # Prepare response
                content_data = {
                    'chunks': chunks,
                    'qa_pairs': qa_pairs,
                    'stats': {
                        'total_chunks': len(chunks),
                        'total_qa_pairs': len(qa_pairs),
                        'total_words': total_words,
                        'total_characters': total_characters,
                        'processing_time': 'N/A'  # We don't store this in Pinecone
                    },
                    'source_id': source_id,
                    'namespace': namespace
                }
                
                logger.info(f"✅ Found {len(chunks)} chunks for source {source_id}")
                return content_data
                
            except Exception as e:
                logger.error(f"❌ Failed to query Pinecone: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get source content: {e}")
            return None

    def process_website_knowledge(self, company_name: str, website_url: str, knowledge_source_id: str = None) -> Dict:
        """Process website knowledge and store ONLY Q&A pairs in Pinecone"""
        try:
            from web_scraper import WebScraper
            from knowledge_integration import KnowledgeIntegrator
            
            logger.info(f"🌐 Processing website knowledge for {company_name}: {website_url}")
            
            # Initialize components
            web_scraper = WebScraper(self.openai_client.api_key)
            knowledge_integrator = KnowledgeIntegrator(self.openai_client.api_key)
            
            # Process website and store ONLY Q&A pairs
            success = knowledge_integrator.process_website_knowledge(
                company_name=company_name,
                website_url=website_url,
                scraper=web_scraper,
                knowledge_source_id=knowledge_source_id
            )
            
            if success:
                return {
                    'success': True,
                    'data': {
                        'website_url': website_url,
                        'processing_type': 'qa_pairs_only',
                        'message': 'Q&A pairs generated and stored successfully'
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to process website and store Q&A pairs'
                }
                
        except Exception as e:
            logger.error(f"❌ Website knowledge processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
