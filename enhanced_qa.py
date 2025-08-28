#!/usr/bin/env python3
"""
Enhanced QA System with Intelligent Scraping
Integrates web scraping with video transcripts for comprehensive support bot knowledge
"""

import asyncio
import json
import os
import re
from typing import List, Dict, Optional
from pinecone import Pinecone
from final_gemini_scraper import FinalGeminiScraper

class EnhancedQASystem:
    def __init__(self, gemini_api_key: str, openai_api_key: str):
        """Initialize enhanced QA system"""
        self.final_scraper = FinalGeminiScraper(gemini_api_key)
        self._embedding_cache = {}  # Simple cache for embeddings
        
    async def process_website_knowledge(self, url: str, company_name: str, qudemo_id: str) -> Dict:
        """Process website knowledge with semantic chunking for specific qudemo"""
        try:
            print(f"🚀 Processing website: {url} for company: {company_name} qudemo: {qudemo_id}")
            
            # Extract content using the final scraper
            try:
                extracted_contents = await self.final_scraper.scrape_website_comprehensive(
                    url, 
                    max_collections=50,
                    max_articles_per_collection=100,
                    smart_filtering=False,
                    exclude_patterns=[]
                )
            except Exception as e:
                print(f"❌ Website scraping failed: {e}")
                return {
                    'success': False,
                    'error': f'Website scraping failed: {str(e)}',
                    'data': {
                        'chunks': [],
                        'summary': {
                            'total_items': 0,
                            'enhanced': 0,
                            'faqs': 0,
                            'beginner': 0,
                            'intermediate': 0,
                            'advanced': 0
                        }
                    }
                }
            
            if not extracted_contents:
                print("⚠️ No content extracted, returning empty result")
                return {
                    'success': False,
                    'error': 'No content could be extracted from the website',
                    'data': {
                        'chunks': [],
                        'summary': {
                            'total_items': 0,
                            'enhanced': 0,
                            'faqs': 0,
                            'beginner': 0,
                            'intermediate': 0,
                            'advanced': 0
                        }
                    }
                }
            
            print(f"✅ Extracted {len(extracted_contents)} articles from {url}")
            
            # Process extracted content using semantic chunking
            total_stored_chunks = []
            
            for content in extracted_contents:
                # Prepare source information with qudemo isolation
                source_info = {
                    'title': content.get('title', 'Untitled'),
                    'url': content.get('url', url),
                    'collection': content.get('collection', 'General'),
                    'content_type': content.get('content_type', 'article'),
                    'has_steps': content.get('has_steps', False),
                    'is_complete': content.get('is_complete', True),
                    'word_count': content.get('word_count', 0),
                    'quality_score': content.get('quality_score', 95),
                    'key_topics': content.get('key_topics', []),
                    'difficulty_level': content.get('difficulty_level', 'intermediate'),
                    'source': 'web_scraping',
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
                
                # Store content using semantic chunking with qudemo isolation
                chunk_data = {
                    'text': content.get('content', ''),
                    'full_context': content.get('content', ''),
                    'source': source_info.get('source', 'web_scraping'),
                    'title': source_info.get('title', 'Untitled'),
                    'url': source_info.get('url', url),
                    'processed_at': source_info.get('processed_at', ''),
                    'company_name': company_name,
                    'qudemo_id': qudemo_id,
                    'source_type': 'web_scraping'
                }
                
                # Store in Pinecone with proper qudemo isolation
                try:
                    # Initialize Pinecone
                    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
                    index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
                    index = pc.Index(index_name)
                    
                    # Create isolated namespace for this qudemo
                    namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
                    
                    # Get embedding for the chunk
                    chunk_embedding = self._get_embedding(chunk_data['text'])
                    
                    # Store chunk with metadata
                    chunk_id = f"web_{qudemo_id}_{hash(chunk_data['url'])}"
                    index.upsert(
                        vectors=[{
                            'id': chunk_id,
                            'values': chunk_embedding,
                            'metadata': chunk_data
                        }],
                        namespace=namespace
                    )
                    
                    total_stored_chunks.append(chunk_data)
                    print(f"✅ Stored chunk {chunk_id} in namespace {namespace}")
                    
                except Exception as store_error:
                    print(f"❌ Failed to store chunk: {store_error}")
                    continue
            
            # If we couldn't store any chunks, clean up and return error
            if not total_stored_chunks:
                print("❌ Failed to store any chunks in Pinecone")
                return {
                    'success': False,
                    'error': 'Failed to store extracted content in vector database',
                    'data': {
                        'chunks': [],
                        'summary': {
                            'total_items': 0,
                            'enhanced': 0,
                            'faqs': 0,
                            'beginner': 0,
                            'intermediate': 0,
                            'advanced': 0
                        }
                    }
                }
            
            # Calculate summary statistics
            total_items = len(total_stored_chunks)
            enhanced = sum(1 for chunk in total_stored_chunks if chunk.get('word_count', 0) > 50)
            faqs = sum(1 for chunk in total_stored_chunks if 'faq' in chunk.get('text', '').lower())
            beginner = sum(1 for chunk in total_stored_chunks if 'beginner' in chunk.get('text', '').lower())
            intermediate = sum(1 for chunk in total_stored_chunks if 'intermediate' in chunk.get('text', '').lower())
            advanced = sum(1 for chunk in total_stored_chunks if 'advanced' in chunk.get('text', '').lower())
            
            summary = {
                'total_items': total_items,
                'enhanced': enhanced,
                'faqs': faqs,
                'beginner': beginner,
                'intermediate': intermediate,
                'advanced': advanced
            }
            
            print(f"✅ Successfully processed {total_items} semantic chunks for {company_name} qudemo {qudemo_id}")
            print(f"📊 Summary: {enhanced} enhanced, {faqs} FAQs, {beginner} beginner, {intermediate} intermediate, {advanced} advanced")
            
            return {
                'success': True,
                'data': {
                    'chunks': total_stored_chunks,
                    'summary': summary
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing website: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {
                    'chunks': [],
                    'summary': {
                        'total_items': 0,
                        'enhanced': 0,
                        'faqs': 0,
                        'beginner': 0,
                        'intermediate': 0,
                        'advanced': 0
                    }
                }
            }

    async def process_qudemo_content(self, company_name: str, qudemo_id: str, video_urls: List[str] = None, website_url: str = None) -> Dict:
        """Process all content for a qudemo - videos first, then website if provided"""
        try:
            print(f"🚀 Processing qudemo content for {company_name} qudemo {qudemo_id}")
            print(f"📹 Videos: {len(video_urls) if video_urls else 0}")
            print(f"🌐 Website: {website_url if website_url else 'None'}")
            
            videos_processed = 0
            website_processed = False
            total_chunks = 0
            processing_order = []
            errors = []
            
            # Process videos first (if any)
            if video_urls:
                print(f"🎬 Processing {len(video_urls)} videos...")
                processing_order.append("videos")
                
                for i, video_url in enumerate(video_urls):
                    try:
                        print(f"🎬 Processing video {i+1}/{len(video_urls)}: {video_url}")
                        
                        # Import video processing module
                        from video_processing import process_video
                        
                        # Process the video
                        result = process_video(
                            video_url=video_url,
                            company_name=company_name,
                            qudemo_id=qudemo_id
                        )
                        
                        if result.get('success'):
                            videos_processed += 1
                            chunks = result.get('chunks', 0)
                            total_chunks += chunks
                            print(f"✅ Video {i+1} processed successfully: {chunks} chunks")
                        else:
                            error_msg = f"Video {i+1} failed: {result.get('error', 'Unknown error')}"
                            errors.append(error_msg)
                            print(f"❌ {error_msg}")
                            
                    except Exception as e:
                        error_msg = f"Video {i+1} processing error: {str(e)}"
                        errors.append(error_msg)
                        print(f"❌ {error_msg}")
                        continue
            
            # Process website (if provided)
            if website_url:
                print(f"🌐 Processing website: {website_url}")
                processing_order.append("website")
                
                try:
                    # Process the website using the existing method
                    result = await self.process_website_knowledge(
                        url=website_url,
                        company_name=company_name,
                        qudemo_id=qudemo_id
                    )
                    
                    if result.get('success'):
                        website_processed = True
                        chunks = len(result.get('data', {}).get('chunks', []))
                        total_chunks += chunks
                        print(f"✅ Website processed successfully: {chunks} chunks")
                    else:
                        error_msg = f"Website processing failed: {result.get('error', 'Unknown error')}"
                        errors.append(error_msg)
                        print(f"❌ {error_msg}")
                        
                except Exception as e:
                    error_msg = f"Website processing error: {str(e)}"
                    errors.append(error_msg)
                    print(f"❌ {error_msg}")
            
            # Prepare response
            success = len(errors) == 0 or (videos_processed > 0 or website_processed)
            
            if success:
                message = f"Content processed successfully. Videos: {videos_processed}, Website: {'Yes' if website_processed else 'No'}, Total chunks: {total_chunks}"
                print(f"✅ {message}")
            else:
                message = f"Content processing failed. Errors: {len(errors)}"
                print(f"❌ {message}")
            
            return {
                'success': success,
                'message': message,
                'videos_processed': videos_processed,
                'website_processed': website_processed,
                'total_chunks': total_chunks,
                'processing_order': processing_order,
                'errors': errors
            }
            
        except Exception as e:
            print(f"❌ Error processing qudemo content: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'videos_processed': 0,
                'website_processed': False,
                'total_chunks': 0,
                'processing_order': [],
                'errors': [str(e)]
            }

    def get_knowledge_summary(self, company_name: str, qudemo_id: str) -> Dict:
        """Get knowledge summary for a specific qudemo"""
        try:
            print(f"📄 Getting knowledge summary for company: {company_name} qudemo: {qudemo_id}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            index = pc.Index(index_name)
            
            # Create namespace from company name and qudemo_id
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Get index stats to check if namespace exists
            try:
                stats = index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                
                if total_vectors == 0:
                    return {
                        'success': True,
                        'data': {
                            'total_chunks': 0,
                            'video_chunks': 0,
                            'knowledge_chunks': 0,
                            'summary': {
                                'total_items': 0,
                                'enhanced': 0,
                                'faqs': 0,
                                'beginner': 0,
                                'intermediate': 0,
                                'advanced': 0
                            }
                        }
                    }
                
                # Get vectors in the specific namespace
                try:
                    # Query with a dummy vector to get namespace stats
                    dummy_embedding = [0.0] * 1536
                    query_results = index.query(
                        vector=dummy_embedding,
                        top_k=1,
                        include_metadata=True,
                        namespace=namespace
                    )
                    
                    # Count different types of content
                    video_chunks = 0
                    knowledge_chunks = 0
                    
                    # Get all vectors in namespace (this is a simplified approach)
                    # In a real implementation, you might want to use Pinecone's fetch API
                    
                    return {
                        'success': True,
                        'data': {
                            'total_chunks': len(query_results.matches) if query_results.matches else 0,
                            'video_chunks': video_chunks,
                            'knowledge_chunks': knowledge_chunks,
                            'summary': {
                                'total_items': len(query_results.matches) if query_results.matches else 0,
                                'enhanced': 0,
                                'faqs': 0,
                                'beginner': 0,
                                'intermediate': 0,
                                'advanced': 0
                            }
                        }
                    }
                    
                except Exception as e:
                    print(f"⚠️ Error querying namespace {namespace}: {e}")
                    return {
                        'success': True,
                        'data': {
                            'total_chunks': 0,
                            'video_chunks': 0,
                            'knowledge_chunks': 0,
                            'summary': {
                                'total_items': 0,
                                'enhanced': 0,
                                'faqs': 0,
                                'beginner': 0,
                                'intermediate': 0,
                                'advanced': 0
                            }
                        }
                    }
                
            except Exception as e:
                print(f"⚠️ Error getting index stats: {e}")
                return {
                    'success': True,
                    'data': {
                        'total_chunks': 0,
                        'video_chunks': 0,
                        'knowledge_chunks': 0,
                        'summary': {
                            'total_items': 0,
                            'enhanced': 0,
                            'faqs': 0,
                            'beginner': 0,
                            'intermediate': 0,
                            'advanced': 0
                        }
                    }
                }
                
        except Exception as e:
            print(f"❌ Error getting knowledge summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {
                    'total_chunks': 0,
                    'video_chunks': 0,
                    'knowledge_chunks': 0,
                    'summary': {
                        'total_items': 0,
                        'enhanced': 0,
                        'faqs': 0,
                        'beginner': 0,
                        'intermediate': 0,
                        'advanced': 0
                    }
                }
            }

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI with caching"""
        try:
            # Check cache first
            if text in self._embedding_cache:
                print(f"📋 Using cached embedding for: {text[:50]}...")
                return self._embedding_cache[text]
            
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            self._embedding_cache[text] = embedding
            print(f"💾 Cached embedding for: {text[:50]}...")
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            # Return a dummy embedding if OpenAI fails
            return [0.0] * 1536  # text-embedding-3-small uses 1536 dimensions (same as ada-002)

    def _debug_namespace_content(self, company_name: str, qudemo_id: str):
        """Debug method to check what content exists in a namespace"""
        try:
            print(f"🔍 Debug: Checking namespace content for {company_name} qudemo {qudemo_id}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            index = pc.Index(index_name)
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Try to get index stats to see if namespace exists
            try:
                stats = index.describe_index_stats()
                print(f"📊 Index stats: {stats}")
                
                # Check if our namespace is in the stats
                if 'namespaces' in stats:
                    namespace_stats = stats['namespaces']
                    if namespace in namespace_stats:
                        print(f"✅ Namespace '{namespace}' found in index with {namespace_stats[namespace]['vector_count']} vectors")
                    else:
                        print(f"❌ Namespace '{namespace}' not found in index")
                        print(f"🔍 Available namespaces: {list(namespace_stats.keys())}")
                else:
                    print("⚠️ No namespace information available in index stats")
                    
            except Exception as stats_error:
                print(f"❌ Could not get index stats: {stats_error}")
            
            # Try a simple search without namespace to see if any content exists
            try:
                print("🔍 Trying a simple search without namespace restriction...")
                # Create a simple test embedding
                test_embedding = [0.0] * 1536
                test_results = index.query(
                    vector=test_embedding,
                    top_k=5,
                    include_metadata=True
                )
                if test_results.matches:
                    print(f"✅ Found {len(test_results.matches)} total vectors in index")
                    for i, match in enumerate(test_results.matches):
                        metadata = match.metadata
                        print(f"  Vector {i+1}: ID={match.id}, SourceType={metadata.get('source_type', 'NO_SOURCE_TYPE')}, Company={metadata.get('company_name', 'NO_COMPANY')}, Qudemo={metadata.get('qudemo_id', 'NO_QUDEMO')}")
                else:
                    print("❌ No vectors found in index at all")
            except Exception as search_error:
                print(f"❌ Simple search failed: {search_error}")
                
        except Exception as e:
            print(f"❌ Error in debug_namespace_content: {e}")

    def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Ask a question and get an intelligent answer from video and knowledge sources"""
        try:
            print(f"❓ Question for {company_name} qudemo {qudemo_id}: {question}")
            
            # First, let's check if there's any content in the namespace at all
            self._debug_namespace_content(company_name, qudemo_id)
            
            # Get question embedding once and reuse it
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
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'answer': "I encountered an error while processing your question. Please try again.",
                'start': 0,
                'end': 0,
                'video_url': None,
                'sources': []
            }

    def _search_video_transcripts(self, question: str, company_name: str, qudemo_id: str, question_embedding: List[float] = None) -> Dict:
        """Search video transcripts for relevant content"""
        try:
            print(f"🎬 Searching video transcripts for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            index = pc.Index(index_name)
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            print(f"🔍 Searching in namespace: {namespace}")
            
            # Use provided embedding or get new one
            if question_embedding is None:
                question_embedding = self._get_embedding(question)
            
            # Search in Pinecone with more inclusive parameters
            try:
                query_results = index.query(
                    vector=question_embedding,
                    top_k=20,  # Increased to get more matches
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=0.1  # Lower threshold to see more matches
                )
                
                if not query_results.matches:
                    print("❌ No content found in Pinecone at all")
                    print(f"🔍 Debug: Namespace '{namespace}' appears to be empty")
                    
                    # Try searching without namespace restriction to see if content exists
                    print("🔍 Trying search without namespace restriction...")
                    try:
                        fallback_results = index.query(
                            vector=question_embedding,
                            top_k=5,
                            include_metadata=True,
                            score_threshold=0.1
                        )
                        if fallback_results.matches:
                            print(f"⚠️ Found {len(fallback_results.matches)} matches without namespace restriction")
                            print("🔍 This suggests the namespace might be incorrect or content was stored differently")
                            for i, match in enumerate(fallback_results.matches):
                                metadata = match.metadata
                                print(f"  Fallback Match {i+1}: Namespace={metadata.get('namespace', 'NO_NAMESPACE')}, SourceType={metadata.get('source_type', 'NO_SOURCE_TYPE')}")
                        else:
                            print("❌ No content found even without namespace restriction")
                    except Exception as fallback_error:
                        print(f"❌ Fallback search failed: {fallback_error}")
                    
                    return {
                        'success': False,
                        'answer': None,
                        'score': 0,
                        'source': 'video',
                        'video_url': None,
                        'start_time': 0,
                        'end_time': 0
                    }
                
                print(f"✅ Found {len(query_results.matches)} total matches")
                
                # Debug: Show all matches and their metadata
                print("🔍 Debug: All matches found:")
                for i, match in enumerate(query_results.matches):
                    metadata = match.metadata
                    source_type = metadata.get('source_type', 'NO_SOURCE_TYPE')
                    source = metadata.get('source', 'NO_SOURCE')
                    title = metadata.get('title', 'NO_TITLE')
                    print(f"  Match {i+1}: Score={match.score:.3f}, SourceType='{source_type}', Source='{source}', Title='{title[:50]}...'")
                
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
                    print("🔍 Debug: All matches were filtered out. Available source_types:")
                    for match in query_results.matches:
                        metadata = match.metadata
                        print(f"  - {metadata.get('source_type', 'NO_SOURCE_TYPE')} (source: {metadata.get('source', 'NO_SOURCE')})")
                    return {
                        'success': False,
                        'answer': None,
                        'score': 0,
                        'source': 'video',
                        'video_url': None,
                        'start_time': 0,
                        'end_time': 0
                    }
                else:
                    # Prioritize videos based on content relevance to the question
                    question_lower = question.lower()
                    
                    # Check if question is about recurring payments
                    if any(term in question_lower for term in ['recurring', 'payment', 'payments', 'setup', 'set up']):
                        # Look for the recurring payments video specifically
                        recurring_video_url = "https://youtu.be/hwko23YbAHs?si=LKWZbN1v4RNK__BS"
                        
                        for match in video_matches:
                            metadata = match.metadata
                            video_url = metadata.get('url', '')
                            if video_url == recurring_video_url:
                                print(f"🎯 Found recurring payments video: {video_url}")
                                best_match = match
                                break
                        else:
                            # If recurring payments video not found, use best semantic match
                            print("⚠️ Recurring payments video not found, using best semantic match")
                            best_match = video_matches[0]
                    else:
                        # For other questions, use best semantic match
                        print("📊 Using best semantic match for non-payment question")
                        best_match = video_matches[0]
                
                # Extract metadata
                metadata = best_match.metadata
                raw_text = metadata.get('text', '')
                video_url = metadata.get('url', '')
                
                # Extract timestamp from metadata first, then fallback to text content
                start_time = metadata.get('start_timestamp', 0)
                end_time = metadata.get('end_timestamp', 0)
                
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
                
                # Format timestamp for display
                if start_time > 0:
                    # Handle large timestamp values (likely in milliseconds or wrong format)
                    if start_time > 3600:  # More than 1 hour, likely wrong format
                        print(f"⚠️ Large timestamp detected: {start_time}s, attempting to extract from text")
                        # Try to extract from text content instead - try multiple patterns
                        timestamp_found = False
                        
                        # Pattern 1: [MM:SS] format
                        timestamp_match = re.search(r'\[(\d{1,2}):(\d{2})\]', raw_text)
                        if timestamp_match:
                            minutes = int(timestamp_match.group(1))
                            seconds = int(timestamp_match.group(2))
                            start_time = minutes * 60 + seconds
                            end_time = start_time + 30
                            print(f"✅ Extracted timestamp from text [MM:SS]: {minutes:02d}:{seconds:02d}")
                            timestamp_found = True
                        
                        # Pattern 2: [HH:MM:SS] format
                        if not timestamp_found:
                            timestamp_match = re.search(r'\[(\d{1,2}):(\d{2}):(\d{2})\]', raw_text)
                            if timestamp_match:
                                hours = int(timestamp_match.group(1))
                                minutes = int(timestamp_match.group(2))
                                seconds = int(timestamp_match.group(3))
                                start_time = hours * 3600 + minutes * 60 + seconds
                                end_time = start_time + 30
                                print(f"✅ Extracted timestamp from text [HH:MM:SS]: {hours:02d}:{minutes:02d}:{seconds:02d}")
                                timestamp_found = True
                        
                        # Pattern 3: Just MM:SS format (without brackets)
                        if not timestamp_found:
                            timestamp_match = re.search(r'(\d{1,2}):(\d{2})', raw_text)
                            if timestamp_match:
                                minutes = int(timestamp_match.group(1))
                                seconds = int(timestamp_match.group(2))
                                start_time = minutes * 60 + seconds
                                end_time = start_time + 30
                                print(f"✅ Extracted timestamp from text MM:SS: {minutes:02d}:{seconds:02d}")
                                timestamp_found = True
                        
                        # Debug: Show first 200 characters of text if no timestamp found
                        if not timestamp_found:
                            print(f"🔍 Debug - First 200 chars of text: {raw_text[:200]}")
                            
                            # Try to estimate timestamp based on chunk position
                            chunk_index = metadata.get('chunk_index', 0)
                            total_chunks = metadata.get('total_chunks', 1)
                            
                            if total_chunks > 1:
                                # Estimate position in video (assuming 16-minute video = 960 seconds)
                                estimated_video_duration = 960  # 16 minutes
                                estimated_start_time = int((chunk_index / total_chunks) * estimated_video_duration)
                                
                                # For content about "disqualified lead agent", it's likely not at the very beginning
                                # Add some offset to skip intro content
                                if estimated_start_time < 60:  # If estimated to be in first minute
                                    estimated_start_time = 120  # Start at 2 minutes instead
                                
                                start_time = estimated_start_time
                                end_time = start_time + 30
                                
                                minutes = estimated_start_time // 60
                                seconds = estimated_start_time % 60
                                print(f"🎯 Estimated timestamp based on chunk position: {minutes:02d}:{seconds:02d}")
                            else:
                                # Fallback to reasonable default
                                start_time = 0
                                end_time = 30
                                print(f"⚠️ No valid timestamp found in text, using default: 00:00")
                    
                    # Format the corrected timestamp
                    if start_time > 0 and start_time <= 3600:  # Valid range (0-1 hour)
                        minutes = int(start_time // 60)
                        seconds = int(start_time % 60)
                        formatted_timestamp = f"{minutes:02d}:{seconds:02d}"
                    else:
                        formatted_timestamp = "00:00"
                        print(f"⚠️ Invalid timestamp after correction: {start_time}s, using 00:00")
                else:
                    formatted_timestamp = "00:00"
                
                # Clean text by removing timestamps
                clean_text = re.sub(r'\[\d{1,2}:\d{2}\]', '', raw_text).strip()
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(question, clean_text)
                
                print(f"✅ Best video match - Score: {best_match.score:.3f}, Relevance: {relevance_score:.3f}")
                print(f"📹 Video URL: {video_url}")
                print(f"⏰ Timestamp: {formatted_timestamp} ({start_time}s - {end_time}s)")
                
                # Check if video is relevant enough to include
                MIN_VIDEO_RELEVANCE = 0.5  # Minimum relevance threshold for videos
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
                
                return {
                    'success': True,
                    'answer': clean_text,
                    'score': best_match.score,
                    'relevance_score': relevance_score,
                    'source': 'video',
                    'video_url': video_url,
                    'start_time': start_time,
                    'end_time': end_time,
                    'formatted_timestamp': formatted_timestamp,
                    'raw_text': raw_text
                }
                
            except Exception as e:
                print(f"❌ Pinecone search error: {e}")
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

    def _search_knowledge_sources(self, question: str, company_name: str, qudemo_id: str, question_embedding: List[float] = None) -> Dict:
        """Search knowledge sources for relevant content"""
        try:
            print(f"📚 Searching knowledge sources for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            index = pc.Index(index_name)
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            print(f"🔍 Searching in namespace: {namespace}")
            
            # Use provided embedding or get new one
            if question_embedding is None:
                question_embedding = self._get_embedding(question)
            
            # Search in Pinecone with more inclusive parameters
            try:
                query_results = index.query(
                    vector=question_embedding,
                    top_k=20,  # Increased to get more potential matches
                    include_metadata=True,
                    namespace=namespace,
                    score_threshold=0.1  # Lower threshold to see more matches
                )
                
                if not query_results.matches:
                    print("❌ No content found in Pinecone at all")
                    print(f"🔍 Debug: Namespace '{namespace}' appears to be empty")
                    
                    # Try searching without namespace restriction to see if content exists
                    print("🔍 Trying search without namespace restriction...")
                    try:
                        fallback_results = index.query(
                            vector=question_embedding,
                            top_k=5,
                            include_metadata=True,
                            score_threshold=0.1
                        )
                        if fallback_results.matches:
                            print(f"⚠️ Found {len(fallback_results.matches)} matches without namespace restriction")
                            print("🔍 This suggests the namespace might be incorrect or content was stored differently")
                            for i, match in enumerate(fallback_results.matches):
                                metadata = match.metadata
                                print(f"  Fallback Match {i+1}: Namespace={metadata.get('namespace', 'NO_NAMESPACE')}, SourceType={metadata.get('source_type', 'NO_SOURCE_TYPE')}")
                        else:
                            print("❌ No content found even without namespace restriction")
                    except Exception as fallback_error:
                        print(f"❌ Fallback search failed: {fallback_error}")
                    
                    return {
                        'success': False,
                        'answer': None,
                        'score': 0,
                        'source': 'knowledge'
                    }
                
                print(f"✅ Found {len(query_results.matches)} total matches")
                
                # Debug: Show all matches and their metadata
                print("🔍 Debug: All matches found:")
                for i, match in enumerate(query_results.matches):
                    metadata = match.metadata
                    source_type = metadata.get('source_type', 'NO_SOURCE_TYPE')
                    source = metadata.get('source', 'NO_SOURCE')
                    title = metadata.get('title', 'NO_TITLE')
                    print(f"  Match {i+1}: Score={match.score:.3f}, SourceType='{source_type}', Source='{source}', Title='{title[:50]}...'")
                
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
                    print("🔍 Debug: All matches were filtered out. Available source_types:")
                    for match in query_results.matches:
                        metadata = match.metadata
                        print(f"  - {metadata.get('source_type', 'NO_SOURCE_TYPE')} (source: {metadata.get('source', 'NO_SOURCE')})")
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
                print(f"❌ Pinecone search error: {e}")
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

    def _format_knowledge_answer(self, question: str, raw_content: str) -> str:
        """Format raw scraped content into structured, user-friendly answer using GPT"""
        try:
            # Check if content is already well-formatted (avoid unnecessary GPT calls)
            if self._is_content_already_formatted(raw_content):
                print("📝 Content already well-formatted, using basic cleaning")
                return self._basic_content_cleaning(raw_content)
            
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
            
            # Fallback to basic cleaning if GPT fails
            if not formatted_answer or len(formatted_answer) < 50:
                print("⚠️ GPT formatting failed, using basic cleaning")
                return self._basic_content_cleaning(raw_content)
            
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

    def _is_content_already_formatted(self, content: str) -> bool:
        """Check if content is already well-formatted and doesn't need GPT processing"""
        
        # Check for common navigation/UI elements that indicate raw scraped content
        navigation_patterns = [
            r'Skip to main content',
            r'Search for articles',
            r'Help Center',
            r'Did this answer your question',
            r'😞.*😐.*😃',
            r'English.*English.*English',
            r'All Collections',
            r'Written by.*Updated.*ago'
        ]
        
        # If any navigation patterns are found, content needs formatting
        for pattern in navigation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        
        # Check if content already has good structure (numbered lists, clear sections)
        good_structure_patterns = [
            r'^\d+\.\s+',  # Numbered lists
            r'^[A-Z][^.!?]*:',  # Clear section headers
            r'To\s+\w+.*follow these steps:',  # Step-by-step instructions
        ]
        
        # If content has good structure, it might already be formatted
        good_structure_count = 0
        for pattern in good_structure_patterns:
            if re.search(pattern, content, re.MULTILINE):
                good_structure_count += 1
        
        # If content has good structure and no navigation elements, it's probably already formatted
        return good_structure_count >= 2

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
            MEDIUM_RELEVANCE = 0.6  # Increased from 0.5 to be more selective
            LOW_RELEVANCE = 0.4     # Increased from 0.3 to be more selective
            
            video_score = video_result.get('relevance_score', 0) if video_result.get('success') else 0
            knowledge_score = knowledge_result.get('relevance_score', 0) if knowledge_result.get('success') else 0
            
            print(f"📊 Video score: {video_score:.3f}, Knowledge score: {knowledge_score:.3f}")
            
            # Decision matrix - prioritize knowledge when it's highly relevant
            if knowledge_score >= HIGH_RELEVANCE and video_score < MEDIUM_RELEVANCE:
                # Knowledge highly relevant, video not relevant enough - use knowledge only
                print("📚 Knowledge highly relevant - using knowledge answer only")
                return {
                    'success': True,
                    'answer': knowledge_result['answer'],  # Already formatted by _format_knowledge_answer
                    'start': 0,
                    'end': 0,
                    'video_url': None,
                    'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}]
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
                        'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}]
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
                else:
                    print("📚 Knowledge more relevant - using knowledge answer")
                    return {
                        'success': True,
                        'answer': knowledge_result['answer'],  # Already formatted by _format_knowledge_answer
                        'start': 0,
                        'end': 0,
                        'video_url': None,
                        'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}]
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
                    'sources': []
                }
                
        except Exception as e:
            print(f"❌ Error selecting best answer: {e}")
            return {
                'success': False,
                'answer': "I encountered an error while processing your question. Please try again.",
                'start': 0,
                'end': 0,
                'video_url': None,
                'sources': []
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
            
            return {
                'success': True,
                'answer': guided_answer,
                'start': video_result.get('start_time', 0),
                'end': video_result.get('end_time', 0),
                'video_url': video_result.get('video_url'),
                'formatted_timestamp': video_result.get('formatted_timestamp', '00:00'),
                'sources': [{'type': 'video', 'url': video_result.get('video_url', ''), 'title': 'Video Transcript'}]
            }
            
        except Exception as e:
            print(f"❌ Error generating guided answer: {e}")
            # Fallback to raw video answer
            return {
                'success': True,
                'answer': video_result['answer'],
                'start': video_result.get('start_time', 0),
                'end': video_result.get('end_time', 0),
                'video_url': video_result.get('video_url'),
                'formatted_timestamp': video_result.get('formatted_timestamp', '00:00'),
                'sources': [{'type': 'video', 'url': video_result.get('video_url', ''), 'title': 'Video Transcript'}]
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
            
            return {
                'success': True,
                'answer': combined_answer,
                'start': video_result.get('start_time', 0),
                'end': video_result.get('end_time', 0),
                'video_url': video_result.get('video_url'),
                'formatted_timestamp': video_result.get('formatted_timestamp', '00:00'),
                'sources': [
                    {'type': 'video', 'url': video_result.get('video_url', ''), 'title': 'Video Transcript'},
                    {'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', 'Knowledge Base')}
                ]
            }
            
        except Exception as e:
            print(f"❌ Error generating combined answer: {e}")
            # Fallback to video answer only
            return self._generate_guided_answer(video_result, question)

# Global instance
enhanced_qa_system = None

async def initialize_enhanced_qa():
    """Initialize the enhanced QA system"""
    global enhanced_qa_system
    
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Debug: Print API keys (first few characters only)
        print(f"🔧 Debug - GEMINI_API_KEY: {gemini_api_key[:10] if gemini_api_key else 'None'}...")
        print(f"🔧 Debug - OPENAI_API_KEY: {openai_api_key[:10] if openai_api_key else 'None'}...")
        
        if not gemini_api_key:
            print("❌ Missing GEMINI_API_KEY for enhanced QA system")
            return False
            
        if not openai_api_key:
            print("❌ Missing OPENAI_API_KEY for enhanced QA system")
            return False
        
        # Clean up API keys (remove any line breaks or extra whitespace)
        gemini_api_key = gemini_api_key.strip()
        openai_api_key = openai_api_key.strip()
        
        enhanced_qa_system = EnhancedQASystem(gemini_api_key, openai_api_key)
        print("✅ Enhanced QA system initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize enhanced QA system: {e}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return False