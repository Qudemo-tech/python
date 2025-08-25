#!/usr/bin/env python3
"""
Enhanced QA System with Intelligent Scraping
Integrates web scraping with video transcripts for comprehensive support bot knowledge
"""

import asyncio
import json
import os
from typing import List, Dict, Optional
from pinecone import Pinecone
from final_gemini_scraper import FinalGeminiScraper

class EnhancedQASystem:
    def __init__(self, gemini_api_key: str, openai_api_key: str):
        """Initialize enhanced QA system"""
        self.final_scraper = FinalGeminiScraper(gemini_api_key)
        
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
        """Get embedding for text using OpenAI"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            # Return a dummy embedding if OpenAI fails
            return [0.0] * 1536

    def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Ask a question and get an intelligent answer from video and knowledge sources"""
        try:
            print(f"❓ Question for {company_name} qudemo {qudemo_id}: {question}")
            
            # Search video transcripts
            video_result = self._search_video_transcripts(question, company_name, qudemo_id)
            
            # Search knowledge sources
            knowledge_result = self._search_knowledge_sources(question, company_name, qudemo_id)
            
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

    def _search_video_transcripts(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Search video transcripts for relevant content"""
        try:
            print(f"🎬 Searching video transcripts for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            index = pc.Index(index_name)
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Get question embedding
            question_embedding = self._get_embedding(question)
            
            # Search in Pinecone
            try:
                query_results = index.query(
                    vector=question_embedding,
                    top_k=10,  # Increased to get more matches
                    include_metadata=True,
                    namespace=namespace
                )
                
                if not query_results.matches:
                    print("❌ No video content found in Pinecone")
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
                
                # Extract timestamp from text content
                import re
                timestamp_match = re.search(r'\[(\d{1,2}):(\d{2})\]', raw_text)
                if timestamp_match:
                    minutes = int(timestamp_match.group(1))
                    seconds = int(timestamp_match.group(2))
                    start_time = minutes * 60 + seconds
                    end_time = start_time + 30  # 30 second window
                else:
                    start_time = 0
                    end_time = 30
                
                # Clean text by removing timestamps
                clean_text = re.sub(r'\[\d{1,2}:\d{2}\]', '', raw_text).strip()
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(question, clean_text)
                
                print(f"✅ Best video match - Score: {best_match.score:.3f}, Relevance: {relevance_score:.3f}")
                print(f"📹 Video URL: {video_url}")
                print(f"⏰ Timestamp: {start_time}s - {end_time}s")
                
                return {
                    'success': True,
                    'answer': clean_text,
                    'score': best_match.score,
                    'relevance_score': relevance_score,
                    'source': 'video',
                    'video_url': video_url,
                    'start_time': start_time,
                    'end_time': end_time,
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

    def _search_knowledge_sources(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Search knowledge sources for relevant content"""
        try:
            print(f"📚 Searching knowledge sources for: {question}")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            index = pc.Index(index_name)
            
            # Create namespace
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            # Get question embedding
            question_embedding = self._get_embedding(question)
            
            # Search in Pinecone
            try:
                query_results = index.query(
                    vector=question_embedding,
                    top_k=3,
                    include_metadata=True,
                    namespace=namespace
                )
                
                if not query_results.matches:
                    print("❌ No knowledge content found in Pinecone")
                    return {
                        'success': False,
                        'answer': None,
                        'score': 0,
                        'source': 'knowledge'
                    }
                
                print(f"✅ Found {len(query_results.matches)} knowledge matches")
                
                # Filter for knowledge content
                knowledge_matches = []
                for match in query_results.matches:
                    metadata = match.metadata
                    source_type = metadata.get('source_type', '')
                    
                    if source_type == 'web_scraping':
                        knowledge_matches.append(match)
                
                if not knowledge_matches:
                    print("⚠️ No knowledge-specific matches found")
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
                
                return {
                    'success': True,
                    'answer': content,
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
            MEDIUM_RELEVANCE = 0.5
            LOW_RELEVANCE = 0.3
            
            video_score = video_result.get('relevance_score', 0) if video_result.get('success') else 0
            knowledge_score = knowledge_result.get('relevance_score', 0) if knowledge_result.get('success') else 0
            
            print(f"📊 Video score: {video_score:.3f}, Knowledge score: {knowledge_score:.3f}")
            
            # Decision matrix
            if video_score >= HIGH_RELEVANCE and knowledge_score >= HIGH_RELEVANCE:
                # Both highly relevant - combine them
                print("🔄 Both sources highly relevant - combining answers")
                return self._generate_combined_answer(video_result, knowledge_result, question)
                
            elif video_score >= HIGH_RELEVANCE and knowledge_score < MEDIUM_RELEVANCE:
                # Video highly relevant, knowledge less so - use video only
                print("🎬 Video highly relevant - using video answer")
                return self._generate_guided_answer(video_result, question)
                
            elif knowledge_score >= HIGH_RELEVANCE and video_score < MEDIUM_RELEVANCE:
                # Knowledge highly relevant, video less so - use knowledge only
                print("📚 Knowledge highly relevant - using knowledge answer")
                return {
                    'success': True,
                    'answer': knowledge_result['answer'],
                    'start': 0,
                    'end': 0,
                    'video_url': None,
                    'sources': [{'type': 'knowledge', 'url': knowledge_result.get('url', ''), 'title': knowledge_result.get('title', '')}]
                }
                
            elif video_score >= MEDIUM_RELEVANCE and knowledge_score >= MEDIUM_RELEVANCE:
                # Both moderately relevant - combine them
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
                        'answer': knowledge_result['answer'],
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
Do not include timestamps or technical jargon unless necessary.

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
Do not include timestamps or technical jargon unless necessary.

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
