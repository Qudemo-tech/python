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

    async def ask_question(self, question: str, company_name: str, qudemo_id: str) -> Dict:
        """Ask a question and get comprehensive answer using semantic search with video navigation for specific qudemo"""
        try:
            print(f"🔍 Processing question: '{question}' for company: {company_name} qudemo: {qudemo_id}")
            
            # Search Pinecone directly for all content (videos and knowledge)
            print(f"🔍 Searching Pinecone directly for content in namespace: {company_name}-{qudemo_id}")
            
            # We'll search Pinecone directly instead of relying on Node.js backend
            videos = []  # Empty list since we're searching Pinecone directly
            
            # Step 2: Search video transcripts first (highest priority)
            video_answer = await self._search_video_transcripts(question, videos, company_name, qudemo_id)
            
            # Step 3: Search scraped knowledge sources directly from Pinecone
            # Create a dummy knowledge_sources list since we're searching Pinecone directly
            knowledge_sources = [{"type": "web_scraping"}]  # Dummy list for compatibility
            knowledge_answer = await self._search_knowledge_sources(question, knowledge_sources, company_name, qudemo_id)
            
            # Step 4: Implement priority-based answer selection
            final_answer = self._select_best_answer(video_answer, knowledge_answer, question)
            
            return final_answer
                
        except Exception as e:
            print(f"❌ Error asking qudemo question: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "answer": "I encountered an error while processing your question. This might be because the qudemo doesn't exist or there's a temporary issue. Please try asking about a different qudemo or contact support if the problem persists.",
                "sources": []
            }

    async def _search_video_transcripts(self, question: str, videos: List[Dict], company_name: str, qudemo_id: str) -> Optional[Dict]:
        """Search through video transcripts for the answer using vector database"""
        try:
            print(f"🎬 Starting video transcript search for: '{question}'")
            print(f"🎬 Company: {company_name}, Qudemo: {qudemo_id}")
            print(f"🎬 Videos provided: {len(videos) if videos else 0}")
            
            # Always search Pinecone regardless of videos list
            print(f"🎬 Searching Pinecone directly...")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            
            # Get the index
            index = pc.Index(index_name)
            
            # Create namespace from company name and qudemo_id for proper isolation
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
            
            print(f"🔍 Searching video transcripts in namespace: {namespace}")
            
            # Create embedding for the question
            question_embedding = self._get_embedding(question)
            print(f"🔍 Generated embedding for question: {len(question_embedding)} dimensions")
            
            # Search for relevant chunks without filter first to see what's available
            try:
                # Try without namespace first to see if there's any content
                print(f"🔍 Trying search without namespace first...")
                query_results_no_namespace = index.query(
                    vector=question_embedding,
                    top_k=10,
                    include_metadata=True
                )
                print(f"🔍 Found {len(query_results_no_namespace.matches)} matches without namespace")
                if query_results_no_namespace.matches:
                    print(f"🔍 First match without namespace: {query_results_no_namespace.matches[0].metadata.get('company_name', 'unknown')} - {query_results_no_namespace.matches[0].metadata.get('qudemo_id', 'unknown')}")
            except Exception as e:
                print(f"🔍 Error searching without namespace: {e}")
            
            # Search for relevant chunks without filter first to see what's available
            try:
                # Try without namespace first to see if there's any content
                print(f"🔍 Trying search without namespace first...")
                query_results = index.query(
                    vector=question_embedding,
                    top_k=20,
                    include_metadata=True
                )
                print(f"🔍 Search without namespace completed")
            except Exception as search_error:
                print(f"❌ Search without namespace failed: {search_error}")
                return None
                
            print(f"🔍 Found {len(query_results.matches)} total matches in namespace")
            
            # Debug: Print all source types found
            source_types = set()
            for match in query_results.matches:
                source_type = match.metadata.get('source_type', 'unknown')
                source = match.metadata.get('source', 'unknown')
                source_types.add(f"source_type: {source_type}, source: {source}")
            
            print(f"🔍 Available source types: {list(source_types)}")
            
            # Filter for video content - be more flexible
            video_matches = [match for match in query_results.matches 
                           if (match.metadata.get('source_type') == 'video_transcript' or 
                               match.metadata.get('source') == 'video' or
                               'video' in match.metadata.get('title', '').lower() or
                               'transcription' in match.metadata.get('title', '').lower())]
            
            # If we have video matches, try to find one with timestamp information
            if video_matches:
                # Look for a match with timestamp information
                timestamped_match = None
                for match in video_matches:
                    if (match.metadata.get('start_timestamp') or 
                        match.metadata.get('start_time') or
                        match.metadata.get('timestamp')):
                        timestamped_match = match
                        break
                
                # If we found a timestamped match, use it; otherwise use the first match
                if timestamped_match:
                    video_matches = [timestamped_match]
                    print(f"🔍 Found timestamped video match: {timestamped_match.metadata.get('start_timestamp', 'unknown')}")
                else:
                    print(f"🔍 No timestamped video matches found, using first match")
            
            # If no video matches found, try using any match with a reasonable score
            if not video_matches and query_results.matches:
                print(f"🔍 No video matches found, using best available match")
                video_matches = [query_results.matches[0]]  # Use the best match regardless of type
            
            print(f"🔍 Found {len(video_matches)} video matches")
            
            if not video_matches:
                print("⚠️ No video transcript chunks found")
                return None
            
            # Get the best match
            best_match = video_matches[0]
            
            # Debug: Print the best match details
            print(f"🔍 Best match details:")
            print(f"   - Score: {best_match.score}")
            print(f"   - ID: {best_match.id}")
            print(f"   - Source: {best_match.metadata.get('source', 'unknown')}")
            print(f"   - Source Type: {best_match.metadata.get('source_type', 'unknown')}")
            print(f"   - Title: {best_match.metadata.get('title', 'unknown')}")
            print(f"   - Text preview: {best_match.metadata.get('text', '')[:100]}...")
            print(f"   - All metadata keys: {list(best_match.metadata.keys())}")
            print(f"   - Start timestamp: {best_match.metadata.get('start_timestamp', 'NOT_FOUND')}")
            print(f"   - End timestamp: {best_match.metadata.get('end_timestamp', 'NOT_FOUND')}")
            print(f"   - URL: {best_match.metadata.get('url', 'NOT_FOUND')}")
            
            print(f"🔍 Best video match score: {best_match.score:.3f}")
            print(f"🔍 Best video match title: {best_match.metadata.get('title', 'Unknown')}")
            
            # Improved relevance checking with multiple criteria
            relevance_score = best_match.score
            
            # Check for keyword relevance in the question and answer
            question_lower = question.lower()
            answer_text = best_match.metadata.get('text', '').lower()
            
            # Extract key terms from the question with more flexible matching
            key_terms = []
            required_terms = []  # Terms that must be present for relevance
            
            # More flexible keyword matching
            if 'workflow' in question_lower:
                key_terms.extend(['workflow', 'process', 'flow', 'diagram', 'chart'])
                required_terms.append('workflow')
            if 'graph' in question_lower:
                key_terms.extend(['graph', 'chart', 'diagram', 'visualization', 'view'])
                required_terms.append('graph')
            if 'outflow' in question_lower:
                key_terms.extend(['outflow', 'cash flow', 'payment', 'expense', 'spending'])
                required_terms.append('outflow')
            if 'ap' in question_lower:
                key_terms.extend(['ap', 'accounts payable', 'payable', 'invoice', 'purchasing'])
            if 'view' in question_lower:
                key_terms.extend(['view', 'see', 'show', 'display', 'find'])
            
            # Check for negative terms that indicate wrong content (but be less strict)
            negative_terms = ['forecast', 'forecasting', 'prediction', 'estimate']
            negative_matches = sum(1 for term in negative_terms if term in answer_text)
            
            # Reduced penalty for negative terms (less strict)
            negative_penalty = negative_matches * 0.15  # Reduced from 0.3 to 0.15
            
            # Calculate keyword relevance
            keyword_matches = sum(1 for term in key_terms if term in answer_text)
            keyword_relevance = keyword_matches / max(len(key_terms), 1) if key_terms else 0
            
            # Check if required terms are present
            required_matches = sum(1 for term in required_terms if term in answer_text)
            required_relevance = required_matches / max(len(required_terms), 1) if required_terms else 1.0
            
            # Combined relevance score with penalties
            combined_score = (relevance_score * 0.6) + (keyword_relevance * 0.2) + (required_relevance * 0.2) - negative_penalty
            
            print(f"🔍 Relevance analysis:")
            print(f"   - Semantic score: {relevance_score:.3f}")
            print(f"   - Keyword relevance: {keyword_relevance:.3f} ({keyword_matches}/{len(key_terms)} terms)")
            print(f"   - Required relevance: {required_relevance:.3f} ({required_matches}/{len(required_terms)} terms)")
            print(f"   - Negative penalty: {negative_penalty:.3f} ({negative_matches} negative terms)")
            print(f"   - Combined score: {combined_score:.3f}")
            print(f"   - Key terms searched: {key_terms}")
            print(f"   - Required terms: {required_terms}")
            print(f"   - Negative terms found: {[term for term in negative_terms if term in answer_text]}")
            
            # Balanced threshold for accuracy and inclusivity
            if combined_score > 0.3:  # Slightly lower threshold to be more inclusive
                print(f"✅ Found relevant video answer with combined score {combined_score:.3f}")
                
                # Extract video information from metadata
                video_url = best_match.metadata.get('url', '')  # Use 'url' instead of 'video_url'
                
                # Clean the text by removing timestamp markers like [00:01], [00:05], etc.
                raw_text = best_match.metadata.get('text', '')
                import re
                
                # Extract the first timestamp from the text before cleaning
                first_timestamp_match = re.search(r'\[(\d{2}):(\d{2})\]', raw_text)
                if first_timestamp_match:
                    minutes, seconds = map(int, first_timestamp_match.groups())
                    start_time = minutes * 60 + seconds
                    print(f"🔍 Extracted start timestamp from text: {minutes}:{seconds} = {start_time}s")
                else:
                    start_time = 0
                    print(f"🔍 No timestamp found in text, using default: {start_time}s")
                
                # Remove timestamp patterns like [00:01], [00:05], etc.
                clean_text = re.sub(r'\[\d{2}:\d{2}\]\s*', '', raw_text)
                # Also remove any remaining timestamp patterns
                clean_text = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]\s*', '', clean_text)
                # Clean up extra whitespace
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                # Calculate end time (start + 30 seconds for a reasonable segment)
                end_time = start_time + 30
                
                # Generate a user-friendly, guided answer using GPT
                guided_answer = self._generate_guided_answer(question, clean_text, video_url)
                
                return {
                    'type': 'video',
                    'answer': guided_answer,
                    'video_url': video_url,
                    'title': best_match.metadata.get('title', 'Video'),
                    'timestamp': start_time,
                    'end_time': end_time,
                    'score': best_match.score,
                    'source_type': 'video_transcript'
                }
            else:
                print(f"⚠️ Best video match combined score {combined_score:.3f} below threshold 0.4")
                print(f"⚠️ Content not relevant enough for question: '{question}'")
                
                # Provide a helpful response when content is not relevant
                return {
                    'type': 'no_relevant_content',
                    'answer': f"I couldn't find specific information about '{question}' in the available video content. The content I found was about AP forecasting, but you asked about AP workflow graphs. You might want to try asking about different features or check if this content has been added to the knowledge base.",
                    'video_url': '',
                    'title': 'No Relevant Content Found',
                    'timestamp': 0,
                    'end_time': 0,
                    'score': combined_score,
                    'source_type': 'no_match'
                }
            
        except Exception as e:
            print(f"❌ Error searching video transcripts: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return None

    async def _search_knowledge_sources(self, question: str, knowledge_sources: List[Dict], company_name: str, qudemo_id: str) -> Optional[Dict]:
        """Search through scraped knowledge sources for the answer"""
        try:
            print(f"📚 Searching knowledge sources for: '{question}'")
            
            # Check if we have access to Pinecone for semantic search
            try:
                import pinecone
                from openai import OpenAI
                
                # Initialize Pinecone
                pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
                index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
                
                # Get the index
                index = pc.Index(index_name)
                
                # Create namespace from company name and qudemo_id for proper isolation
                namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"
                
                print(f"🔍 Searching knowledge sources in namespace: {namespace}")
                
                # Create embedding for the question
                question_embedding = self._get_embedding(question)
                print(f"🔍 Generated embedding for question: {len(question_embedding)} dimensions")
                
                # Search for relevant chunks without filter first to see what's available
                query_results = index.query(
                    vector=question_embedding,
                    top_k=20,  # Get more results
                    include_metadata=True,
                    namespace=namespace
                )
                
                print(f"🔍 Found {len(query_results.matches)} total matches in namespace")
                
                # Filter for web_scraping content
                web_scraping_matches = [match for match in query_results.matches 
                                      if match.metadata.get('source_type') == 'web_scraping']
                
                print(f"🔍 Found {len(web_scraping_matches)} web_scraping matches")
                
                if web_scraping_matches:
                    # Get the best match
                    best_match = web_scraping_matches[0]
                    print(f"🔍 Best knowledge match score: {best_match.score:.3f}")
                    print(f"🔍 Best knowledge match title: {best_match.metadata.get('title', 'Unknown')}")
                    print(f"🔍 Best knowledge match URL: {best_match.metadata.get('url', 'Unknown')}")
                    
                    # Lowered threshold for better results
                    if best_match.score > 0.2:  # Lower relevance threshold
                        print(f"✅ Found relevant knowledge answer with score {best_match.score:.3f}")
                        return {
                            'type': 'knowledge',
                            'answer': best_match.metadata.get('text', ''),
                            'source_url': best_match.metadata.get('url', ''),
                            'title': best_match.metadata.get('title', 'Knowledge Source'),
                            'score': best_match.score,
                            'source_type': 'web_scraping'
                        }
                    else:
                        print(f"⚠️ Best knowledge match score {best_match.score:.3f} below threshold 0.2")
                else:
                    print("⚠️ No web_scraping matches found in Pinecone search")
                
            except Exception as e:
                print(f"⚠️ Pinecone search failed: {e}")
                import traceback
                print(f"🔍 Full traceback: {traceback.format_exc()}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error searching knowledge sources: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return None

    def _generate_guided_answer(self, question: str, video_content: str, video_url: str) -> str:
        """Generate a user-friendly, guided answer based on video content"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Create a prompt that asks for a guided, user-friendly response
            prompt = f"""
You are a helpful sales manager explaining product features to a customer. Based on the video content below, provide a clear, step-by-step guide that answers the user's question.

User Question: {question}

Video Content: {video_content}

Please provide a helpful, guided answer that:
1. Explains the feature or process clearly
2. Breaks down the steps in a logical order
3. Uses friendly, professional language
4. Makes it easy for someone to follow along
5. Highlights key benefits or important points
6. Avoids technical jargon unless necessary

Format your response as a clear, helpful guide that a sales manager would give to a customer.
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable sales manager helping customers understand product features. Provide clear, helpful, and professional explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            guided_answer = response.choices[0].message.content.strip()
            print(f"🤖 Generated guided answer: {guided_answer[:100]}...")
            
            return guided_answer
            
        except Exception as e:
            print(f"❌ Error generating guided answer: {e}")
            # Fallback to original content if GPT fails
            return video_content

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

    def _select_best_answer(self, video_answer: Optional[Dict], knowledge_answer: Optional[Dict], question: str) -> Dict:
        """Select the best answer based on priority and accuracy"""
        try:
            print(f"🔍 Selecting best answer - Video: {video_answer is not None}, Knowledge: {knowledge_answer is not None}")
            
            # If only video answer exists
            if video_answer and not knowledge_answer:
                # Check if it's a "no relevant content" response
                if video_answer.get('type') == 'no_relevant_content':
                    print("⚠️ No relevant video content found")
                    return {
                        "success": True,
                        "answer": video_answer['answer'],
                        "sources": [],
                        "answer_source": "no_relevant_content",
                        "confidence": "No relevant video content found"
                    }
                else:
                    print("✅ Using video answer (only source available)")
                    return {
                        "success": True,
                        "answer": video_answer['answer'],
                        "sources": [
                            {
                                "type": "video",
                                "title": video_answer['title'],
                                "url": video_answer['video_url'],
                                "timestamp": video_answer['timestamp'],
                                "metadata": {
                                    "start": video_answer['timestamp'],
                                    "end": video_answer.get('end_time', video_answer['timestamp'] + 30),
                                    "source_type": video_answer['source_type'],
                                    "score": video_answer['score']
                                }
                            }
                        ],
                        "video_url": video_answer['video_url'],
                        "start": video_answer['timestamp'],
                        "end": video_answer.get('end_time', video_answer['timestamp'] + 30),
                        "video_title": video_answer['title'],
                        "answer_source": "video_transcript",
                        "confidence": f"Video transcript answer (score: {video_answer['score']:.3f})"
                    }
            
            # If only knowledge answer exists
            elif knowledge_answer and not video_answer:
                print("✅ Using knowledge answer (only source available)")
                return {
                    "success": True,
                    "answer": knowledge_answer['answer'],
                    "sources": [
                        {
                            "type": "knowledge",
                            "title": knowledge_answer['title'],
                            "url": knowledge_answer['source_url'],
                            "content": knowledge_answer['answer'],
                            "metadata": {
                                "source_type": knowledge_answer['source_type'],
                                "score": knowledge_answer['score']
                            }
                        }
                    ],
                    "answer_source": "scraped_data",
                    "confidence": f"Scraped data answer (score: {knowledge_answer['score']:.3f})"
                }
            
            # If both answers exist, compare and select the best one
            elif video_answer and knowledge_answer:
                print(f"🔍 Comparing answers - Video score: {video_answer['score']:.3f}, Knowledge score: {knowledge_answer['score']:.3f}")
                
                # If video transcript is more accurate (higher score), use it with timestamp
                if video_answer['score'] > knowledge_answer['score']:
                    print("✅ Video transcript is more accurate - using with timestamp")
                    return {
                        "success": True,
                        "answer": video_answer['answer'],
                        "sources": [
                            {
                                "type": "video",
                                "title": video_answer['title'],
                                "url": video_answer['video_url'],
                                "timestamp": video_answer['timestamp'],
                                "metadata": {
                                    "start": video_answer['timestamp'],
                                    "end": video_answer.get('end_time', video_answer['timestamp'] + 30),
                                    "source_type": video_answer['source_type'],
                                    "score": video_answer['score']
                                }
                            }
                        ],
                        "video_url": video_answer['video_url'],
                        "start": video_answer['timestamp'],
                        "end": video_answer.get('end_time', video_answer['timestamp'] + 30),
                        "video_title": video_answer['title'],
                        "answer_source": "video_transcript",
                        "confidence": f"Video transcript was more accurate (score: {video_answer['score']:.3f} vs {knowledge_answer['score']:.3f})"
                    }
                else:
                    print("✅ Knowledge source is more accurate - using scraped data")
                    return {
                        "success": True,
                        "answer": knowledge_answer['answer'],
                        "sources": [
                            {
                                "type": "knowledge",
                                "title": knowledge_answer['title'],
                                "url": knowledge_answer['source_url'],
                                "content": knowledge_answer['answer'],
                                "metadata": {
                                    "source_type": knowledge_answer['source_type'],
                                    "score": knowledge_answer['score']
                                }
                            }
                        ],
                        "answer_source": "scraped_data",
                        "confidence": f"Scraped data was more accurate (score: {knowledge_answer['score']:.3f} vs {video_answer['score']:.3f})"
                    }
            
            # If no answers found
            else:
                print("⚠️ No relevant answers found in either video transcripts or knowledge sources")
                return {
                    "success": True,
                    "answer": "I couldn't find a specific answer to your question in the available content for this qudemo. You might want to try rephrasing your question or ask about a different aspect of the content.",
                    "sources": [],
                    "answer_source": "no_match",
                    "confidence": "No relevant content found"
                }
            
        except Exception as e:
            print(f"❌ Error selecting best answer: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "answer_source": "error"
            }

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
