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
        # Note: Knowledge integration is now handled by the Node.js backend
        # This class focuses on web scraping and content processing
        self.final_scraper = FinalGeminiScraper(gemini_api_key)
        
        # Initialize knowledge integrator for Q&A functionality
        from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator
        self.knowledge_integrator = EnhancedKnowledgeIntegrator(
            openai_api_key=openai_api_key,
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            pinecone_index=os.getenv('PINECONE_INDEX')
        )
        
    async def process_website_knowledge(self, url: str, company_name: str, qudemo_id: str = None) -> Dict:
        """Process website knowledge with semantic chunking (no Q&A generation) for specific qudemo"""
        try:
            print(f"🚀 Processing website: {url} for company: {company_name} qudemo: {qudemo_id}")
            
            # Extract content using the final scraper - comprehensive website crawling with path-based filtering
            try:
                extracted_contents = await self.final_scraper.scrape_website_comprehensive(
                    url, 
                    max_collections=50,  # Allow up to 50 collections for comprehensive coverage
                    max_articles_per_collection=100,  # Allow up to 100 articles per collection
                    smart_filtering=False,  # Use path-based filtering instead
                    exclude_patterns=[]  # Use built-in basic filtering only
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
                # Prepare source information
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
                    'source': 'web_scraping'
                }
                
                # Store content using semantic chunking
                # Create a chunk dictionary from the content
                chunk_data = {
                    'text': content.get('content', ''),
                    'full_context': content.get('content', ''),
                    'source': source_info.get('source', 'web_scraping'),
                    'title': source_info.get('title', 'Untitled'),
                    'url': source_info.get('url', url),
                    'processed_at': source_info.get('processed_at', ''),
                }
                
                stored_result = self.knowledge_integrator.store_semantic_chunks(
                    chunks=[chunk_data],
                    company_name=company_name,
                    qudemo_id=qudemo_id
                )
                
                # Add the chunk data to our tracking list if storage was successful
                if stored_result.get('success', False):
                    total_stored_chunks.append(chunk_data)
            
            # If we couldn't store any chunks, clean up and return error
            if not total_stored_chunks:
                print("❌ Failed to store any chunks in Pinecone")
                await self.cleanup_failed_website_data(url, company_name, qudemo_id)
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
            faqs = sum(1 for chunk in total_stored_chunks if 'faq' in chunk.get('text_preview', '').lower())
            beginner = sum(1 for chunk in total_stored_chunks if 'beginner' in chunk.get('text_preview', '').lower())
            intermediate = sum(1 for chunk in total_stored_chunks if 'intermediate' in chunk.get('text_preview', '').lower())
            advanced = sum(1 for chunk in total_stored_chunks if 'advanced' in chunk.get('text_preview', '').lower())
            
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
            # Clean up any data that might have been stored
            await self.cleanup_failed_website_data(url, company_name, qudemo_id)
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

    async def cleanup_failed_website_data(self, url: str, company_name: str, qudemo_id: str = None):
        """Clean up failed website data from Pinecone for specific qudemo"""
        try:
            print(f"🧹 Cleaning up failed website data for: {url} (company: {company_name}, qudemo: {qudemo_id})")
            
            # Delete vectors by company name, qudemo_id and source type
            # Note: This is a simplified cleanup - in a production system you might want more granular deletion
            print(f"🗑️ Cleaning up Pinecone data for company: {company_name} qudemo: {qudemo_id}")
            
            # The actual deletion will be handled by the Node.js backend calling the Python API
            # This is just a placeholder for any local cleanup needed
            
        except Exception as cleanup_error:
            print(f"❌ Cleanup failed: {cleanup_error}")
    
    async def ask_question(self, question: str, company_name: str, qudemo_id: str = None) -> Dict:
        """Ask a question and get comprehensive answer using semantic search with video navigation for specific qudemo"""
        try:
            print(f"🔍 Processing question: '{question}' for company: {company_name} qudemo: {qudemo_id}")
            
            # Use search with context for better results
            search_result = self.knowledge_integrator.search_with_context(
                query=question,
                company_name=company_name,
                qudemo_id=qudemo_id,
                top_k=20  # Get more results to ensure we have both video and website sources
            )
            
            if not search_result.get('success', False) or not search_result.get('results'):
                return {
                    "success": False,
                    "message": "No relevant information found",
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }
            
            search_results = search_result['results']
            
            # Build context from search results
            context_parts = []
            sources = []
            video_timestamps = []
            video_sources = []
            website_sources = []
            
            for result in search_results:
                # Use full context for better answer generation
                text = result.metadata.get('text', '') if hasattr(result, 'metadata') and result.metadata else ''
                context_parts.append(text)
                
                # Add source information - use metadata directly
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                
                source_data = {
                    'text': text[:200] + "..." if len(text) > 200 else text,
                    'score': result.score if hasattr(result, 'score') else 0,
                    'source_type': metadata.get('source', 'unknown'),
                    'title': metadata.get('title', 'Unknown'),
                    'url': metadata.get('url', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'total_chunks': metadata.get('total_chunks', 1)
                }
                
                print(f"🔍 Source: {source_data['source_type']} - {source_data['title']} - {source_data['url']}")
                print(f"🔍 Chunk {source_data['chunk_index']} selected with score {source_data['score']}")
                
                # Extract video timestamp if available
                if metadata.get('source') == 'video':
                    print(f"🔍 Processing video source: {metadata.get('title', 'Unknown')}")
                    # Look for timestamp in the chunk text or metadata
                    timestamp = self._extract_video_timestamp(text, metadata)
                    if timestamp is not None:
                        source_data['start_timestamp'] = timestamp
                        video_timestamps.append({
                            'url': metadata.get('url', ''),
                            'start': timestamp,
                            'title': metadata.get('title', 'Video')
                        })
                        print(f"🔍 Added video timestamp: {timestamp}s for {metadata.get('title', 'Unknown')}")
                    else:
                        print(f"🔍 No timestamp found for video: {metadata.get('title', 'Unknown')}")
                    video_sources.append(source_data)
                else:
                    print(f"🔍 Processing website source: {metadata.get('title', 'Unknown')} (source: {metadata.get('source', 'unknown')})")
                    website_sources.append(source_data)
            
            # Determine source composition and prioritize accordingly
            has_video_sources = len(video_sources) > 0
            has_website_sources = len(website_sources) > 0
            
            print(f"🔍 Found {len(video_sources)} video sources and {len(website_sources)} website sources")
            
            # Logic based on source availability:
            # 1. If both video and website sources → prioritize video sources + include some website sources
            # 2. If only video sources → use video sources only
            # 3. If only website sources → use website sources only (no timestamps)
            
            if has_video_sources and has_website_sources:
                # Both available: prioritize video sources (top 3) + include top website sources (top 2)
                final_sources = []
                final_sources.extend(video_sources[:3])  # Top 3 video sources
                final_sources.extend(website_sources[:2])  # Top 2 website sources
                print("🔍 Using combined video + website sources")
            elif has_video_sources:
                # Only video sources available
                final_sources = video_sources[:5]  # Top 5 video sources
                print("🔍 Using video sources only")
            else:
                # Only website sources available
                final_sources = website_sources[:5]  # Top 5 website sources
                print("🔍 Using website sources only (no timestamps)")
                # Clear any video timestamps since we're only using website sources
                video_timestamps = []
            
            sources = final_sources
            
            # Combine context
            full_context = "\n\n".join(context_parts)
            
            # Generate answer using the context
            prompt = f"""Based on the following information, please answer the question: "{question}"

Information:
{full_context}

Please provide a comprehensive and accurate answer based only on the information provided above. If the information doesn't contain enough details to answer the question completely, please say so."""

            print(f"🔍 Debug: knowledge_integrator type: {type(self.knowledge_integrator)}")
            print(f"🔍 Debug: knowledge_integrator is None: {self.knowledge_integrator is None}")
            
            if self.knowledge_integrator is None:
                print("❌ Error: knowledge_integrator is None!")
                return {
                    "success": False,
                    "message": "Knowledge integrator not initialized",
                    "answer": "I'm sorry, the knowledge system is not properly initialized. Please try again later.",
                    "sources": []
                }
            
            try:
                # First search for relevant context
                search_result = self.knowledge_integrator.search_with_context(
                    query=question,
                    company_name=company_name,
                    qudemo_id=qudemo_id
                )
                
                if search_result.get('success', False) and search_result.get('results'):
                    # Generate answer using the search results
                    answer_result = self.knowledge_integrator.generate_answer(
                        query=question,
                        context_results=search_result['results'],
                        company_name=company_name,
                        qudemo_id=qudemo_id
                    )
                    
                    if answer_result.get('success', False):
                        answer = answer_result['answer']
                    else:
                        answer = "I'm sorry, I couldn't generate an answer based on the available information."
                else:
                    answer = "I'm sorry, I couldn't find relevant information to answer your question."
                    
                print(f"🔍 Debug: Generated answer: {answer[:100]}...")
            except Exception as gen_error:
                print(f"❌ Error generating answer: {gen_error}")
                answer = "I'm sorry, I couldn't generate an answer at this time."
            
            # Prepare response with video navigation data
            response_data = {
                    "success": True,
                "answer": answer,
                "sources": sources,
                "context_used": len(context_parts),
                "search_results_count": len(search_results)
            }
            
            print(f"🔍 Final sources being sent to frontend:")
            for i, source in enumerate(sources):
                print(f"  Source {i+1}: {source['source_type']} - {source['title']} - {source['url']}")
            
            # Check if any of the final sources are actually video sources
            final_sources_include_video = any(source.get('source_type') == 'video' for source in sources)
            print(f"🔍 Final sources include video: {final_sources_include_video}")
            
            # Add video navigation data only if video sources were used AND they have valid timestamps
            if video_timestamps and has_video_sources and len(video_timestamps) > 0 and final_sources_include_video:
                print(f"🔍 Found {len(video_timestamps)} video timestamps, checking for valid ones...")
                
                # Filter for valid video timestamps with URLs
                valid_video_timestamps = [
                    vt for vt in video_timestamps 
                    if vt.get('url') and vt.get('start') is not None and vt.get('url').strip()
                ]
                
                if valid_video_timestamps:
                    # Use the highest scoring video timestamp
                    best_video = max(valid_video_timestamps, key=lambda x: next(
                        (s['score'] for s in sources if s.get('start_timestamp') == x['start']), 0
                    ))
                    
                    response_data.update({
                        "video_url": best_video['url'],
                        "start": best_video['start'],
                        "end": best_video['start'] + 30,  # 30 second window
                        "video_title": best_video['title']
                    })
                    print(f"🔍 Added video timestamp: {best_video['start']}s from {best_video['title']}")
                else:
                    print("🔍 Video sources found but no valid timestamps with URLs - not adding video data")
            else:
                print("🔍 No video timestamps added (website sources only or no timestamps found)")
            
            print(f"✅ Generated answer with {len(sources)} sources, {len(video_timestamps)} video timestamps")
            return response_data
            
        except Exception as e:
            print(f"❌ Error asking question: {e}")
            return {
                "success": False,
                "message": f"Error processing question: {str(e)}",
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": []
            }

    async def ask_qudemo_question(self, question: str, qudemo_id: str) -> Dict:
        """Ask a question specific to a particular qudemo using its videos and knowledge sources"""
        try:
            print(f"🔍 Processing qudemo-specific question: '{question}' for qudemo: {qudemo_id}")
            
            # Import required modules
            import os
            import aiohttp
            import json
            from typing import List, Dict, Optional
            
            # Get Node.js API URL from environment
            node_api_url = os.getenv('NODE_API_URL', 'http://localhost:5000')
            
            # Fetch qudemo data from Node.js backend
            async with aiohttp.ClientSession() as session:
                qudemo_url = f"{node_api_url}/api/qudemos/data/{qudemo_id}"
                print(f"🔍 Fetching qudemo data from: {qudemo_url}")
                
                async with session.get(qudemo_url, timeout=30) as response:
                    if response.status == 404:
                        return {
                            "success": True,
                            "answer": "I couldn't find the specific qudemo you're asking about. This might be because the qudemo hasn't been created yet or the ID is incorrect. Please make sure you're asking about an existing qudemo, or contact support if you believe this is an error.",
                            "sources": []
                        }
                    elif response.status != 200:
                        return {
                            "success": False,
                            "message": f"Failed to fetch qudemo data: {response.status}",
                            "answer": "Unable to retrieve qudemo information. Please try again.",
                            "sources": []
                        }
                    
                    qudemo_data = await response.json()
                    
                    if not qudemo_data.get('success'):
                        error_msg = qudemo_data.get('error', 'Failed to fetch qudemo data')
                        print(f"❌ Qudemo data fetch failed: {error_msg}")
                        
                        # Provide a helpful response for missing qudemo
                        if "not found" in error_msg.lower():
                            return {
                                "success": True,
                                "answer": "I couldn't find the specific qudemo you're asking about. This might be because the qudemo hasn't been created yet or the ID is incorrect. Please make sure you're asking about an existing qudemo, or contact support if you believe this is an error.",
                                "sources": []
                            }
                        else:
                            return {
                                "success": True,
                                "answer": "I'm having trouble accessing the qudemo data right now. This could be because the qudemo doesn't exist or there's a temporary connection issue. Please try asking about a different qudemo or contact support if you believe this is an error.",
                                "sources": []
                            }
                    
                    qudemo = qudemo_data['data']
                    videos = qudemo.get('videos', [])
                    knowledge_sources = qudemo.get('knowledge_sources', [])
                    qudemo_title = qudemo.get('title', 'Untitled Qudemo')
                    qudemo_description = qudemo.get('description', 'No description available')
                    company_name = qudemo.get('company_name', 'Unknown Company')
                    
                    print(f"📊 Found {len(videos)} videos and {len(knowledge_sources)} knowledge sources for qudemo {qudemo_id}")
                    
                    # Step 1: Search video transcripts first (highest priority)
                    video_answer = await self._search_video_transcripts(question, videos, company_name, qudemo_id)
                    
                    # Step 2: Search scraped knowledge sources
                    knowledge_answer = await self._search_knowledge_sources(question, knowledge_sources, company_name, qudemo_id)
                    
                    # Step 3: Implement priority-based answer selection
                    final_answer = self._select_best_answer(video_answer, knowledge_answer, question)
                    
                    return final_answer
                
        except Exception as e:
            print(f"❌ Error asking qudemo question: {e}")
            return {
                "success": False,
                "answer": "I encountered an error while processing your question. This might be because the qudemo doesn't exist or there's a temporary issue. Please try asking about a different qudemo or contact support if the problem persists.",
                "sources": []
            }

    async def _search_video_transcripts(self, question: str, videos: List[Dict], company_name: str, qudemo_id: str) -> Optional[Dict]:
        """Search through video transcripts for the answer using vector database"""
        try:
            if not videos:
                return None
                
            print(f"🎬 Searching {len(videos)} video transcripts for: '{question}'")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            
            # Get the index
            index = pc.Index(index_name)
            
            # Create namespace from company name
            namespace = company_name.lower().replace(' ', '-')
            
            # Create embedding for the question
            question_embedding = self._get_embedding(question)
            
            # Search for relevant video chunks
            query_results = index.query(
                vector=question_embedding,
                top_k=6,  # Get top 6 similar chunks
                include_metadata=True,
                namespace=namespace,
                filter={
                    "source_type": "video_transcript",
                    "qudemo_id": qudemo_id
                }
            )
            
            if not query_results.matches:
                print("⚠️ No video transcript chunks found")
                return None
            
            # Get the best match
            best_match = query_results.matches[0]
            
            if best_match.score > 0.3:  # Minimum relevance threshold
                print(f"✅ Found video answer with score {best_match.score:.2f}")
                
                # Extract video information from metadata
                video_url = best_match.metadata.get('video_url', '')
                start_time = best_match.metadata.get('start_time', 0)
                end_time = best_match.metadata.get('end_time', 0)
                
                return {
                    'type': 'video',
                    'answer': best_match.metadata.get('text', ''),
                    'video_url': video_url,
                    'title': best_match.metadata.get('title', 'Video'),
                    'timestamp': start_time,
                    'end_time': end_time,
                    'score': best_match.score,
                    'source_type': 'video_transcript'
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error searching video transcripts: {e}")
            return None

    async def _search_knowledge_sources(self, question: str, knowledge_sources: List[Dict], company_name: str, qudemo_id: str) -> Optional[Dict]:
        """Search through scraped knowledge sources for the answer"""
        try:
            if not knowledge_sources:
                return None
                
            print(f"📚 Searching {len(knowledge_sources)} knowledge sources for: '{question}'")
            
            # For now, we'll use a simple approach since knowledge sources are stored in the database
            # In a full implementation, you would search through the vector database
            
            # Check if we have access to Pinecone for semantic search
            try:
                import pinecone
                from openai import OpenAI
                
                # Initialize Pinecone
                pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
                index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
                
                # Get the index
                index = pc.Index(index_name)
                
                # Create namespace from company name
                namespace = company_name.lower().replace(' ', '-')
                
                # Search for relevant chunks
                query_results = index.query(
                    vector=self._get_embedding(question),
                    top_k=6,
                    include_metadata=True,
                    namespace=namespace,
                    filter={
                        "source_type": "web_scraping",
                        "qudemo_id": qudemo_id
                    }
                )
                
                if query_results.matches:
                    # Get the best match
                    best_match = query_results.matches[0]
                    
                    if best_match.score > 0.7:  # High relevance threshold
                        return {
                            'type': 'knowledge',
                            'answer': best_match.metadata.get('text', ''),
                            'source_url': best_match.metadata.get('url', ''),
                            'title': best_match.metadata.get('title', 'Knowledge Source'),
                            'score': best_match.score,
                            'source_type': 'web_scraping'
                        }
                
            except Exception as e:
                print(f"⚠️ Pinecone search failed: {e}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error searching knowledge sources: {e}")
            return None

    async def get_video_chunks(self, video_id: str, company_name: str, video_url: str) -> List[Dict]:
        """Get video transcript chunks from vector database"""
        try:
            print(f"🔍 Getting chunks for video: {video_id} (company: {company_name})")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX', 'qudemo-index')
            
            # Get the index
            index = pc.Index(index_name)
            
            # Create namespace from company name
            namespace = company_name.lower().replace(' ', '-')
            
            # Search for chunks related to this video
            query_results = index.query(
                vector=[0.0] * 1536,  # Dummy vector to get all chunks
                top_k=100,
                include_metadata=True,
                namespace=namespace,
                filter={
                    "source_type": "video_transcript",
                    "video_id": video_id
                }
            )
            
            chunks = []
            if query_results.matches:
                for match in query_results.matches:
                    chunk = {
                        'id': match.id,
                        'text': match.metadata.get('text', ''),
                        'start_time': match.metadata.get('start_time', 0),
                        'end_time': match.metadata.get('end_time', 0),
                        'score': match.score,
                        'metadata': match.metadata
                    }
                    chunks.append(chunk)
                
                # Sort by start_time
                chunks.sort(key=lambda x: x['start_time'])
                
                print(f"✅ Found {len(chunks)} chunks for video: {video_id}")
            else:
                print(f"⚠️ No chunks found for video: {video_id}")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error getting video chunks: {e}")
            return []

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
            # If only video answer exists
            if video_answer and not knowledge_answer:
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
                                "end": video_answer['timestamp'] + 30,
                                "source_type": video_answer['source_type']
                            }
                        }
                    ],
                    "video_url": video_answer['video_url'],
                    "start": video_answer['timestamp'],
                    "end": video_answer['timestamp'] + 30,
                    "video_title": video_answer['title'],
                    "answer_source": "video_transcript"
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
                                "source_type": knowledge_answer['source_type']
                            }
                        }
                    ],
                    "answer_source": "scraped_data"
                }
            
            # If both answers exist, compare and select the best one
            elif video_answer and knowledge_answer:
                print(f"🔍 Comparing answers - Video score: {video_answer['score']:.2f}, Knowledge score: {knowledge_answer['score']:.2f}")
                
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
                                    "end": video_answer['timestamp'] + 30,
                                    "source_type": video_answer['source_type']
                                }
                            }
                        ],
                        "video_url": video_answer['video_url'],
                        "start": video_answer['timestamp'],
                        "end": video_answer['timestamp'] + 30,
                        "video_title": video_answer['title'],
                        "answer_source": "video_transcript",
                        "confidence": f"Video transcript was more accurate (score: {video_answer['score']:.2f} vs {knowledge_answer['score']:.2f})"
                    }
                else:
                    print("✅ Scraped data is more accurate - using without timestamp")
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
                                    "source_type": knowledge_answer['source_type']
                                }
                            }
                        ],
                        "answer_source": "scraped_data",
                        "confidence": f"Scraped data was more accurate (score: {knowledge_answer['score']:.2f} vs {video_answer['score']:.2f})"
                    }
            
            # If no answers found
            else:
                print("⚠️ No relevant answers found in any source")
                return {
                    "success": True,
                    "answer": "I couldn't find a specific answer to your question in the available content for this qudemo. You might want to try rephrasing your question or ask about a different aspect of the content.",
                    "sources": [],
                    "answer_source": "none"
                }
                
        except Exception as e:
            print(f"❌ Error selecting best answer: {e}")
            return {
                "success": False,
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": []
            }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking"""
        import re
        
        # Split by sentence endings, but be careful with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _calculate_relevance_score(self, question: str, text: str) -> float:
        """Calculate relevance score between question and text"""
        question_words = set(question.lower().split())
        text_words = set(text.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        question_words = question_words - stop_words
        text_words = text_words - stop_words
        
        if not question_words:
            return 0.0
        
        # Calculate intersection
        intersection = question_words & text_words
        
        # Calculate score based on word overlap and text length
        word_score = len(intersection) / len(question_words)
        length_penalty = min(1.0, len(text) / 200)  # Prefer shorter, more focused text
        
        return word_score * length_penalty

    def _estimate_timestamp(self, sentence_index: int, total_sentences: int, video_type: str) -> int:
        """Estimate timestamp based on sentence position and video type"""
        if total_sentences == 0:
            return 0
        
        # Different video types may have different pacing
        if video_type == 'youtube':
            # YouTube videos often have more structured content
            seconds_per_sentence = 8
        elif video_type == 'loom':
            # Loom videos are often more conversational
            seconds_per_sentence = 6
        else:
            # Default assumption
            seconds_per_sentence = 7
        
        timestamp = sentence_index * seconds_per_sentence
        
        # Ensure timestamp is reasonable (not too high)
        max_timestamp = 3600  # 1 hour max
        return min(timestamp, max_timestamp)
    
    def _clean_text_formatting(self, text: str) -> str:
        """Clean up text formatting and remove unwanted symbols"""
        import re
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove multiple empty lines
        text = re.sub(r' +', ' ', text)  # Remove multiple spaces
        text = re.sub(r'\t+', ' ', text)  # Replace tabs with spaces
        
        # Remove all markdown formatting completely
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Remove markdown headers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold** markers
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove *italic* markers
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove `code` markers
        
        # Remove brackets and unwanted symbols
        text = re.sub(r'\[([^\]]+)\]', r'\1', text)  # Remove [brackets] but keep content
        text = re.sub(r'\{([^}]+)\}', r'\1', text)  # Remove {braces} but keep content
        text = re.sub(r'\(([^)]+)\)', r'\1', text)  # Remove (parentheses) but keep content
        
        # Clean up numbered lists - convert inline numbered lists to proper line-separated lists
        # Find patterns like "1 First Name 2 Last Name 3 Work Email" and split them
        text = re.sub(r'(\d+\.\s+[^0-9]+?)(?=\d+\.\s+)', r'\1\n', text)  # Add line breaks between numbered items
        
        # Clean up list formatting - convert * and - to proper bullet points
        text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)  # Convert * and - to •
        
        # Remove unwanted symbols but keep important punctuation and letters/numbers
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\n\r•]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Add proper line breaks for better readability
        # Add line breaks after numbered lists and bullet points
        text = re.sub(r'(\d+\.\s+[^\n]+)', r'\1\n', text)  # Add line break after numbered items
        text = re.sub(r'(•\s+[^\n]+)', r'\1\n', text)  # Add line break after bullet points
        
        # Add line breaks after section headers (text that ends with :)
        text = re.sub(r'([^:]+:\s*)\n', r'\1\n\n', text)
        
        # Remove extra whitespace at start and end
        text = text.strip()
        
        return text

    def _extract_video_timestamp(self, text: str, source_info: Dict) -> Optional[float]:
        """Extract video timestamp from text or metadata"""
        try:
            # Look for timestamp patterns in the text
            import re
            
            # Pattern for [HH:MM:SS] or [MM:SS] timestamps (bracketed format)
            timestamp_patterns = [
                r'\[(\d{1,2}):(\d{2}):(\d{2})\]',  # [HH:MM:SS]
                r'\[(\d{1,2}):(\d{2})\]',          # [MM:SS]
                r'(\d{1,2}):(\d{2}):(\d{2})',     # HH:MM:SS (without brackets)
                r'(\d{1,2}):(\d{2})'               # MM:SS (without brackets)
            ]
            
            # For the specific case of 1:24:24, this should be MM:SS format
            # Let's add a special pattern for this case
            special_pattern = r'(\d{1}):(\d{2}):(\d{2})'  # Single digit:MM:SS (likely MM:SS)
            special_match = re.search(special_pattern, text)
            if special_match:
                first, second, third = map(int, special_match.groups())
                # If first digit is 1-9, it's likely MM:SS format
                if first <= 9:
                    timestamp = first * 60 + second
                    print(f"🔍 Extracted timestamp [MM:SS] from special pattern: {first}:{second} = {timestamp}s")
                    return self._validate_timestamp(timestamp)
            
            for pattern in timestamp_patterns:
                match = re.search(pattern, text)
                if match:
                    groups = match.groups()
                    if len(groups) == 3:  # HH:MM:SS
                        hours, minutes, seconds = map(int, groups)
                        timestamp = hours * 3600 + minutes * 60 + seconds
                        print(f"🔍 Extracted timestamp [HH:MM:SS]: {hours}:{minutes}:{seconds} = {timestamp}s")
                        
                        # If hours is 1 or less, it's likely MM:SS format incorrectly parsed as HH:MM:SS
                        if hours <= 1 and timestamp > 120:  # If it's more than 2 minutes, probably MM:SS
                            print(f"⚠️ Timestamp {hours}:{minutes}:{seconds} seems like MM:SS format, reinterpreting")
                            # Reinterpret as MM:SS
                            timestamp = hours * 60 + minutes
                            print(f"🔍 Reinterpreted as MM:SS: {hours}:{minutes} = {timestamp}s")
                        
                        return self._validate_timestamp(timestamp)
                    elif len(groups) == 2:  # MM:SS
                        minutes, seconds = map(int, groups)
                        timestamp = minutes * 60 + seconds
                        print(f"🔍 Extracted timestamp [MM:SS]: {minutes}:{seconds} = {timestamp}s")
                        return self._validate_timestamp(timestamp)
            
            # Check metadata for timestamp information
            if 'start_timestamp' in source_info:
                timestamp = float(source_info['start_timestamp'])
                print(f"🔍 Extracted timestamp from metadata: {timestamp}s")
                return self._validate_timestamp(timestamp)
            
            print(f"🔍 No timestamp found in text: {text[:100]}...")
            return None
            
        except Exception as e:
            print(f"⚠️ Error extracting timestamp: {e}")
            return None
    
    def _validate_timestamp(self, timestamp: float) -> float:
        """Validate and cap timestamp to reasonable values"""
        if timestamp is None:
            return 0
        
        # Cap at 30 minutes (1800 seconds) for most videos
        if timestamp > 1800:
            print(f"⚠️ Timestamp {timestamp}s capped at 1800s (30 minutes)")
            return 1800
        
        return timestamp
    
    def get_knowledge_summary(self, company_name: str, qudemo_id: str = None) -> Dict:
        """Get summary of knowledge base for a company or specific qudemo"""
        try:
            print(f"📊 Getting knowledge summary for company: {company_name} qudemo: {qudemo_id}")
            
            # Query Pinecone for all content for this company/qudemo
            results = self.knowledge_integrator.search_with_context(
                query="",  # Empty query to get all content
                company_name=company_name,
                qudemo_id=qudemo_id,
                top_k=1000  # Get a large number to get statistics
            )
            
            if not results:
                return {
                "success": True,
                "data": {
                    "company_name": company_name,
                        "qudemo_id": qudemo_id,
                    "total_chunks": 0,
                    "last_updated": "Unknown",
                    "sources": []
                    }
                }
            
            # Calculate statistics
            total_chunks = len(results)
            sources = {}
            last_updated = "Unknown"
            
            # Group Settle Help Center content under one source
            settle_help_center_data = {
                "type": "website",
                "title": "Settle Help Center",
                "url": "https://help.settle.com/en/",
                "chunks": 0,
                "total_articles": 0
            }
            
            for result in results:
                metadata = result.get('metadata', {})
                
                # Get source information from metadata
                source_type = metadata.get('source', 'web_scraping')
                source_title = metadata.get('title', 'Unknown')
                source_url = metadata.get('url', '')
                
                # Skip video content and sources without meaningful titles
                if source_type == 'video' or not source_title or source_title == 'Unknown':
                    continue
                
                # Check if this is Settle Help Center content
                if 'help.settle.com' in source_url or 'settle' in source_title.lower():
                    settle_help_center_data["chunks"] += 1
                    settle_help_center_data["total_articles"] += 1
                    continue
                
                # Track other unique sources
                source_key = f"{source_type}:{source_title}"
                if source_key not in sources:
                    sources[source_key] = {
                        "type": source_type,
                        "title": source_title,
                        "url": source_url,
                        "chunks": 0
                    }
                sources[source_key]["chunks"] += 1
                
                # Track last updated
                if 'processed_at' in metadata:
                    if last_updated == "Unknown" or metadata['processed_at'] > last_updated:
                        last_updated = metadata['processed_at']
            
            # Add Settle Help Center as a single source if it has content
            final_sources = list(sources.values())
            if settle_help_center_data["chunks"] > 0:
                final_sources.insert(0, settle_help_center_data)  # Put it first
            
            return {
                "success": True,
                "data": {
                    "company_name": company_name,
                    "qudemo_id": qudemo_id,
                    "total_chunks": total_chunks,
                    "last_updated": last_updated,
                    "sources": final_sources
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting knowledge summary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_source_content(self, source_id: str, company_name: str, qudemo_id: str = None) -> Optional[Dict]:
        """Get content from a specific source for a specific qudemo"""
        try:
            # URL decode the source_id if it's URL-encoded
            import urllib.parse
            decoded_source_id = urllib.parse.unquote(source_id)
            print(f"🔍 Retrieving source content: {source_id} (decoded: {decoded_source_id}) for company: {company_name} qudemo: {qudemo_id}")
            
            # Query Pinecone for the specific source content
            results = self.knowledge_integrator.search_with_context(
                query="",  # Empty query to get all content
                company_name=company_name,
                qudemo_id=qudemo_id,
                top_k=1000  # Get more results to find all chunks for the source
            )
            
            # Find all chunks for the specific source
            source_chunks = []
            source_metadata = None
            
            for result in results:
                metadata = result.get('metadata', {})
                
                # Check if this chunk belongs to the source we're looking for
                # Special handling for Settle Help Center
                if (decoded_source_id == "https://help.settle.com/en/" or 
                    decoded_source_id == "Settle Help Center" or
                    source_id == "https://help.settle.com/en/" or 
                    source_id == "Settle Help Center" or
                    decoded_source_id == "settle_help_center" or
                    source_id == "settle_help_center"):
                    # Return all Settle Help Center content
                    if 'help.settle.com' in metadata.get('url', '') or 'settle' in metadata.get('title', '').lower():
                        source_chunks.append({
                            "chunk_index": result.get('chunk_index', 0),
                            "text": result.get('text', ''),
                            "full_context": result.get('full_context', ''),
                            "metadata": metadata
                        })
                        
                        # Use the first chunk's metadata as the source metadata
                        if source_metadata is None:
                            source_metadata = metadata
                elif (metadata.get('url', '') == source_id or 
                    metadata.get('url', '').endswith(source_id) or 
                    metadata.get('title', '').lower() in source_id.lower() or
                    str(metadata.get('chunk_id', '')).startswith(source_id) or
                    metadata.get('source', '').lower() in source_id.lower()):
                    
                    source_chunks.append({
                        "chunk_index": result.get('chunk_index', 0),
                        "text": result.get('text', ''),
                        "full_context": result.get('full_context', ''),
                        "metadata": metadata
                    })
                    
                    # Use the first chunk's metadata as the source metadata
                    if source_metadata is None:
                        source_metadata = metadata
            
            if source_chunks:
                # Sort chunks by chunk_index
                source_chunks.sort(key=lambda x: int(x['chunk_index']))
                
                # Combine all text and clean it up
                combined_text = "\n\n".join([chunk['text'] for chunk in source_chunks])
                
                # Clean up the text formatting
                combined_text = self._clean_text_formatting(combined_text)
                
                return {
                    "success": True,
                    "data": {
                        "source_id": source_id,
                        "company_name": company_name,
                        "qudemo_id": qudemo_id,
                        "text": combined_text,
                        "chunks": source_chunks,
                        "total_chunks": len(source_chunks),
                        "metadata": source_metadata,
                        "title": source_metadata.get('title', 'Unknown') if source_metadata else 'Unknown',
                        "url": source_metadata.get('url', '') if source_metadata else ''
                    }
                }
            else:
                print(f"⚠️ Source content not found: {source_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting source content: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return None

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
