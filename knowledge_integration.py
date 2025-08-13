#!/usr/bin/env python3
"""
Knowledge Integration Module
Combines video transcripts, scraped data, and documents for enhanced Q&A
"""

import logging
import os
import time
from typing import List, Dict, Optional
import openai
from openai import OpenAI
from pinecone import ServerlessSpec

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeIntegrator:
    def __init__(self, openai_api_key: str):
        """Initialize knowledge integrator"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks"""
        try:
            logger.info(f"🧠 Creating embeddings for {len(texts)} chunks...")
            
            embeddings = []
            batch_size = 100  # OpenAI batch size limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    response = self.openai_client.embeddings.create(
                        input=batch,
                        model="text-embedding-3-small"
                    )
                    batch_embeddings = [e.embedding for e in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f"✅ Created embeddings for batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"❌ Batch embedding failed: {e}")
                    # Create zero embeddings for failed batch
                    zero_embedding = [0.0] * 1536  # OpenAI embedding dimension
                    embeddings.extend([zero_embedding] * len(batch))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return []
    
    def store_qa_pairs_only(self, company_name: str, chunks: List[Dict],
                           source_type: str, source_info: str, knowledge_source_id: str = None) -> bool:
        """Store ONLY Q&A pairs in Pinecone (not raw text chunks)"""
        try:
            from qa_system import pc, default_index_name
            import os
            
            # Initialize Q&A generator
            from qa_generator import QAGenerator
            qa_generator = QAGenerator(self.openai_client.api_key)
            
            logger.info(f"📦 Processing {len(chunks)} chunks for Q&A generation for {company_name}")
            
            # Check if any chunks are FAQ content
            faq_chunks = []
            regular_chunks = []
            
            for chunk in chunks:
                chunk_source_type = chunk.get('metadata', {}).get('source_type', source_type)
                # Only process website and document content, not video transcripts
                if chunk_source_type in ['website', 'document']:
                    # Check if this is FAQ content
                    if chunk.get('is_faq', False) and chunk.get('faq_pair'):
                        faq_chunks.append(chunk)
                    else:
                        regular_chunks.append(chunk)
            
            qa_pairs = []
            
            # Handle FAQ content first - extract Q&A pairs directly
            if faq_chunks:
                logger.info(f"🔍 Found {len(faq_chunks)} FAQ chunks - extracting Q&A pairs directly")
                for chunk in faq_chunks:
                    faq_pair = chunk.get('faq_pair', {})
                    if faq_pair.get('question') and faq_pair.get('answer'):
                        qa_pairs.append({
                            'id': f"faq_{hash(faq_pair['question']) % 10000}",
                            'question': faq_pair['question'],
                            'answer': faq_pair['answer'],
                            'question_type': 'faq',
                            'difficulty': 'basic',
                            'source': 'extracted_faq'
                        })
                logger.info(f"✅ Extracted {len(qa_pairs)} Q&A pairs from FAQ content")
            
            # Handle regular content - generate Q&A pairs using AI
            if regular_chunks:
                # Select top chunks based on quality criteria
                top_chunks = self._select_best_chunks_for_qa(regular_chunks, max_chunks=10)
                logger.info(f"🤖 Generating Q&A pairs from {len(top_chunks)} best regular chunks (selected by quality)...")
                generated_qa_pairs = qa_generator.generate_qa_from_chunks(
                    chunks=top_chunks,
                    source_info=source_info,
                    source_type=source_type
                )
                qa_pairs.extend(generated_qa_pairs)
                logger.info(f"✅ Generated {len(generated_qa_pairs)} additional Q&A pairs from regular content")
            
            if not qa_pairs:
                logger.info(f"📄 No Q&A pairs found or generated")
                return True  # Success, just no Q&A to store
            
            if not qa_pairs:
                logger.info(f"📄 No Q&A pairs generated")
                return True  # Success, just no Q&A generated
            
            logger.info(f"✅ Generated {len(qa_pairs)} Q&A pairs")
            
            # Prepare Q&A chunks for storage
            qa_chunks = []
            for qa in qa_pairs:
                # Create Q&A chunk with answer as main text
                qa_chunk = {
                    'text': qa['answer'],  # Store answer as main text for retrieval
                    'metadata': {
                        'source_type': source_type,
                        'source_info': source_info,
                        'company': company_name,
                        'chunk_type': 'qa_pair',
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'question_type': qa.get('question_type', 'general'),
                        'difficulty': qa.get('difficulty', 'basic'),
                        'qa_id': qa['id'],
                        'knowledge_source_id': str(knowledge_source_id) if knowledge_source_id else 'unknown',
                        'content_type': 'qa_pair'  # Special identifier to isolate from video transcripts
                    }
                }
                qa_chunks.append(qa_chunk)
            
            # Create embeddings for Q&A pairs
            qa_texts = [chunk['text'] for chunk in qa_chunks]
            qa_embeddings = self.create_embeddings(qa_texts)
            
            if not qa_embeddings:
                logger.error(f"❌ Failed to create embeddings for Q&A pairs")
                return False
            
            # Store ONLY Q&A pairs in Pinecone
            existing_indexes = [index.name for index in pc.list_indexes()]
            index_name = os.getenv("PINECONE_INDEX", default_index_name)
            
            if index_name not in existing_indexes and existing_indexes:
                logger.warning(f"⚠️ Index quota reached; falling back to existing index: {existing_indexes[0]}")
                index_name = existing_indexes[0]
            
            index = pc.Index(index_name)
            namespace = company_name.lower().replace(' ', '-')
            
            # Prepare vectors for upsert (ONLY Q&A pairs)
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(qa_chunks, qa_embeddings)):
                metadata = chunk['metadata'].copy()
                
                vectors.append({
                    'id': f"{company_name}_qa_{source_type}_{i}_{hash(source_info) % 10000}",
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert vectors (ONLY Q&A pairs)
            index.upsert(vectors=vectors, namespace=namespace)
            
            logger.info(f"✅ Successfully stored {len(vectors)} Q&A vectors in namespace '{namespace}'")
            
            # Log Q&A statistics
            qa_types = {}
            qa_difficulties = {}
            for qa in qa_pairs:
                qa_types[qa.get('question_type', 'general')] = qa_types.get(qa.get('question_type', 'general'), 0) + 1
                qa_difficulties[qa.get('difficulty', 'basic')] = qa_difficulties.get(qa.get('difficulty', 'basic'), 0) + 1
            
            logger.info(f"📊 Q&A Statistics:")
            logger.info(f"   Question Types: {qa_types}")
            logger.info(f"   Difficulties: {qa_difficulties}")
            logger.info(f"   Total Q&A pairs stored: {len(qa_pairs)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store Q&A pairs in Pinecone: {e}")
            return False
    
    def _select_best_chunks_for_qa(self, chunks: List[Dict], max_chunks: int = 10) -> List[Dict]:
        """Select the best chunks for Q&A generation based on quality criteria"""
        try:
            if len(chunks) <= max_chunks:
                return chunks
            
            # Score each chunk based on quality criteria
            scored_chunks = []
            for chunk in chunks:
                text = chunk.get('text', '')
                if not text:
                    continue
                
                # Calculate quality score based on multiple factors
                score = self._calculate_chunk_quality_score(chunk)
                scored_chunks.append((score, chunk))
            
            # Sort by score (highest first) and take top chunks
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [chunk for score, chunk in scored_chunks[:max_chunks]]
            
            # Log selection criteria
            logger.info(f"📊 Chunk selection criteria:")
            logger.info(f"   Total chunks available: {len(chunks)}")
            logger.info(f"   Selected chunks: {len(top_chunks)}")
            logger.info(f"   Selection method: Quality-based scoring")
            
            return top_chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to select best chunks: {e}")
            # Fallback to first N chunks
            return chunks[:max_chunks]
    
    def _calculate_chunk_quality_score(self, chunk: Dict) -> float:
        """Calculate quality score for a chunk based on multiple criteria"""
        try:
            text = chunk.get('text', '')
            if not text:
                return 0.0
            
            score = 0.0
            
            # 1. Content length score (prefer medium-length content)
            char_count = len(text)
            word_count = len(text.split())
            
            if 500 <= char_count <= 2000:
                score += 3.0  # Optimal length
            elif 200 <= char_count < 500:
                score += 2.0  # Good length
            elif 2000 < char_count <= 4000:
                score += 2.5  # Long but acceptable
            else:
                score += 1.0  # Too short or too long
            
            # 2. Content density score (prefer content-rich text)
            # Calculate ratio of meaningful words to total length
            meaningful_words = len([word for word in text.split() if len(word) > 3])
            if word_count > 0:
                density_ratio = meaningful_words / word_count
                score += density_ratio * 2.0
            
            # 3. Question potential score (prefer content that can generate good Q&A)
            question_indicators = ['how', 'what', 'why', 'when', 'where', 'which', 'can', 'should', 'will', 'does']
            question_count = sum(1 for indicator in question_indicators if indicator in text.lower())
            score += min(question_count * 0.5, 3.0)  # Cap at 3 points
            
            # 4. Technical content score (prefer technical/educational content)
            technical_terms = ['feature', 'function', 'process', 'system', 'account', 'transaction', 'report', 'data', 'integration']
            technical_count = sum(1 for term in technical_terms if term in text.lower())
            score += min(technical_count * 0.3, 2.0)  # Cap at 2 points
            
            # 5. Structure score (prefer well-structured content)
            if any(char in text for char in [':', '-', '•', '*', '1.', '2.', '3.']):
                score += 1.0  # Has structure indicators
            
            # 6. URL quality score (prefer main content pages over navigation/utility pages)
            url = chunk.get('url', '')
            if url:
                low_quality_indicators = ['/tag/', '/category/', '/author/', '/page/', '/search', '/login', '/register']
                if not any(indicator in url.lower() for indicator in low_quality_indicators):
                    score += 1.0  # Good URL structure
            
            return score
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate chunk quality score: {e}")
            return 1.0  # Default score
    
    def process_website_knowledge(self, company_name: str, website_url: str, scraper, knowledge_source_id: str = None) -> bool:
        """Process website knowledge and store ONLY Q&A pairs in Pinecone"""
        try:
            logger.info(f"🌐 Processing website knowledge for {company_name}: {website_url}")
            
            # Scrape website
            scraped_pages = scraper.scrape_website(website_url)
            
            if not scraped_pages:
                logger.warning(f"⚠️ No content scraped from {website_url}")
                return False
            
            # Chunk scraped content
            chunks = scraper.chunk_scraped_content(scraped_pages)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created from scraped content")
                return False
            
            logger.info(f"📄 Created {len(chunks)} chunks from scraped content")
            logger.info(f"🤖 Generating Q&A pairs from website content...")
            
            # Store ONLY Q&A pairs in Pinecone (not raw chunks)
            success = self.store_qa_pairs_only(
                company_name=company_name,
                chunks=chunks,
                source_type='website',
                source_info=website_url,
                knowledge_source_id=knowledge_source_id
            )
            
            if success:
                logger.info(f"✅ Website Q&A pairs processed and stored successfully")
            else:
                logger.error(f"❌ Failed to store website Q&A pairs in Pinecone")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Website knowledge processing failed: {e}")
            return False
    
    def process_document_knowledge(self, company_name: str, document_data: Dict, doc_processor) -> bool:
        """Process document knowledge and store in Pinecone"""
        try:
            logger.info(f"📄 Processing document knowledge for {company_name}: {document_data['filename']}")
            
            # Chunk document content
            chunks = doc_processor.chunk_document_content(document_data)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created from document content")
                return False
            
            # Create embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.create_embeddings(texts)
            
            if not embeddings:
                logger.error(f"❌ Failed to create embeddings for document content")
                return False
            
            # Store in Pinecone
            success = self.store_in_pinecone(
                company_name=company_name,
                chunks=chunks,
                embeddings=embeddings,
                source_type='document',
                source_info=document_data['filename']
            )
            
            if success:
                logger.info(f"✅ Document knowledge processed successfully: {len(chunks)} chunks")
            else:
                logger.error(f"❌ Failed to store document knowledge in Pinecone")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Document knowledge processing failed: {e}")
            return False
    
    def log_knowledge_sources(self, company_name: str, sources: List[Dict]) -> None:
        """Log knowledge sources for debugging"""
        try:
            logger.info(f"📊 Knowledge sources for {company_name}:")
            
            source_counts = {}
            for source in sources:
                source_type = source.get('source_type', 'unknown')
                source_counts[source_type] = source_counts.get(source_type, 0) + 1
            
            for source_type, count in source_counts.items():
                logger.info(f"  - {source_type}: {count} chunks")
            
            # Log sample sources
            for source in sources[:3]:  # Show first 3 sources
                source_type = source.get('source_type', 'unknown')
                source_info = source.get('source_info', 'unknown')
                logger.info(f"  Sample {source_type}: {source_info}")
                
        except Exception as e:
            logger.error(f"❌ Failed to log knowledge sources: {e}")
    
    def get_knowledge_summary(self, company_name: str) -> Dict:
        """Get summary of knowledge sources for a company"""
        try:
            from qa_system import pc, default_index_name
            
            index = pc.Index(default_index_name)
            namespace = company_name.lower()
            
            # Get index stats
            stats = index.describe_index_stats()
            
            if 'namespaces' in stats and namespace in stats['namespaces']:
                namespace_stats = stats['namespaces'][namespace]
                total_vectors = namespace_stats.get('vector_count', 0)
                
                logger.info(f"📊 Knowledge summary for {company_name}: {total_vectors} total vectors")
                
                return {
                    'company': company_name,
                    'total_vectors': total_vectors,
                    'namespace': namespace
                }
            else:
                logger.warning(f"⚠️ No knowledge found for {company_name}")
                return {
                    'company': company_name,
                    'total_vectors': 0,
                    'namespace': namespace
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get knowledge summary: {e}")
            return {
                'company': company_name,
                'total_vectors': 0,
                'error': str(e)
            }
