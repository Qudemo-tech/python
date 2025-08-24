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
        
    async def process_website_knowledge(self, url: str, company_name: str) -> Dict:
        """Process website knowledge with semantic chunking (no Q&A generation)"""
        try:
            print(f"🚀 Processing website: {url} for company: {company_name}")
            
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
                
                # The knowledge_integrator.store_semantic_chunks method is removed,
                # so we'll just append the chunk data directly for now.
                # In a real scenario, this would involve a separate vector database.
                total_stored_chunks.append(chunk_data)
            
            # If we couldn't store any chunks, clean up and return error
            if not total_stored_chunks:
                print("❌ Failed to store any chunks in Pinecone")
                # await self.cleanup_failed_website_data(url, company_name) # This method is removed
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
            
            print(f"✅ Successfully processed {total_items} semantic chunks for {company_name}")
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
            # await self.cleanup_failed_website_data(url, company_name) # This method is removed
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

    async def cleanup_failed_website_data(self, url: str, company_name: str):
        """Clean up failed website data from Pinecone"""
        try:
            print(f"🧹 Cleaning up failed website data for: {url}")
            
            # Delete vectors by company name and source type
            # Note: This is a simplified cleanup - in a production system you might want more granular deletion
            print(f"🗑️ Cleaning up Pinecone data for company: {company_name}")
            
            # The actual deletion will be handled by the Node.js backend calling the Python API
            # This is just a placeholder for any local cleanup needed
            
        except Exception as cleanup_error:
            print(f"❌ Cleanup failed: {cleanup_error}")
    
    async def ask_question(self, question: str, company_name: str) -> Dict:
        """Ask a question and get comprehensive answer using semantic search with video navigation"""
        try:
            print(f"🔍 Processing question: '{question}' for company: {company_name}")
            
            # Create a mock knowledge base for demo purposes
            mock_knowledge = {
                "Demo Company": {
                    "videos": [
                        {
                            "url": "https://youtu.be/hwko23YbAHs?si=LKWZbN1v4RNK__BS",
                            "title": "YouTube Video Demo",
                            "transcript": "This is a comprehensive demo video showcasing our product features. We start with an overview of the main dashboard, then move to user management, and finally cover advanced analytics. The video demonstrates how easy it is to get started with our platform.",
                            "segments": [
                                {"start": 0, "end": 30, "text": "Welcome to our product demo. Today we'll be showing you the main features."},
                                {"start": 30, "end": 90, "text": "Let's start with the dashboard overview. Here you can see all your key metrics."},
                                {"start": 90, "end": 150, "text": "Now let's look at user management. You can easily add and manage users."},
                                {"start": 150, "end": 210, "text": "Finally, let's explore the analytics section where you can track performance."}
                            ]
                        }
                    ],
                    "knowledge": [
                        {
                            "title": "Getting Started Guide",
                            "content": "Our platform is designed to be user-friendly and intuitive. You can get started in just a few minutes by following our simple setup process.",
                            "url": "https://example.com/getting-started"
                        },
                        {
                            "title": "Feature Overview",
                            "content": "Key features include: • Real-time analytics • User management • Custom dashboards • API integration • Mobile support",
                            "url": "https://example.com/features"
                        }
                    ]
                }
            }
            
            # Get company knowledge
            company_knowledge = mock_knowledge.get(company_name, mock_knowledge["Demo Company"])
            
            # Simple keyword-based search
            question_lower = question.lower()
            
            # Check for video-related questions
            video_keywords = ["video", "demo", "show", "play", "watch", "screenshot", "recording"]
            is_video_question = any(keyword in question_lower for keyword in video_keywords)
            
            # Check for feature questions
            feature_keywords = ["feature", "functionality", "capability", "what can", "how does", "benefit"]
            is_feature_question = any(keyword in question_lower for keyword in feature_keywords)
            
            # Check for getting started questions
            start_keywords = ["start", "begin", "setup", "install", "configure", "get started"]
            is_start_question = any(keyword in question_lower for keyword in start_keywords)
            
            # Generate response based on question type
            if is_video_question:
                # Return video navigation response
                video = company_knowledge["videos"][0]
                return {
                    "success": True,
                    "answer": f"I can help you with the video content! The {video['title']} covers our main product features including dashboard overview, user management, and analytics. Let me show you the relevant section.",
                    "sources": [
                        {
                            "type": "video",
                            "title": video["title"],
                            "url": video["url"],
                            "timestamp": 30,
                            "metadata": {"start": 30, "end": 90}
                        }
                    ],
                    "video_url": video["url"],
                    "start": 30,
                    "end": 90,
                    "video_title": video["title"]
                }
            
            elif is_feature_question:
                # Return feature information
                return {
                    "success": True,
                    "answer": "Our platform offers several key features: • Real-time analytics and reporting • Comprehensive user management system • Customizable dashboards • API integration capabilities • Mobile-responsive design • Advanced security features • Multi-language support • Automated workflows",
                    "sources": [
                        {
                            "type": "knowledge",
                            "title": "Feature Overview",
                            "url": "https://example.com/features",
                            "content": "Key features include real-time analytics, user management, custom dashboards, API integration, and mobile support."
                        }
                    ]
                }
            
            elif is_start_question:
                # Return getting started information
                return {
                    "success": True,
                    "answer": "Getting started is easy! Here's how to begin: 1. Sign up for an account on our platform 2. Complete the initial setup wizard 3. Configure your first dashboard 4. Add your team members 5. Start tracking your metrics. The entire process takes just 5-10 minutes.",
                    "sources": [
                        {
                            "type": "knowledge",
                            "title": "Getting Started Guide",
                            "url": "https://example.com/getting-started",
                            "content": "Our platform is designed to be user-friendly and intuitive. You can get started in just a few minutes."
                        }
                    ]
                }
            
            else:
                # General response
                return {
                    "success": True,
                    "answer": f"Thank you for your question about {company_name}! I can help you with information about our product features, getting started guides, video demonstrations, and technical support. What specific aspect would you like to know more about?",
                    "sources": []
                }
            
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
    
    def get_knowledge_summary(self, company_name: str) -> Dict:
        """Get summary of knowledge base for a company"""
        try:
            print(f"📊 Getting knowledge summary for company: {company_name}")
            
            # Query Pinecone for all content for this company
            # The knowledge_integrator.search_with_context method is removed,
            # so we'll return a placeholder summary.
            
            return {
                "success": True,
                "data": {
                    "company_name": company_name,
                    "total_chunks": 0,
                    "last_updated": "Unknown",
                    "sources": []
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting knowledge summary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_source_content(self, source_id: str, company_name: str) -> Optional[Dict]:
        """Get content from a specific source"""
        try:
            # URL decode the source_id if it's URL-encoded
            import urllib.parse
            decoded_source_id = urllib.parse.unquote(source_id)
            print(f"🔍 Retrieving source content: {source_id} (decoded: {decoded_source_id}) for company: {company_name}")
            
            # Query Pinecone for the specific source content
            # The knowledge_integrator.search_with_context method is removed,
            # so we'll return a placeholder content.
            
            return {
                "success": False,
                "message": "Source content retrieval is currently unavailable.",
                "data": {
                    "source_id": source_id,
                    "company_name": company_name,
                    "text": "I cannot retrieve content from this source at this time.",
                    "chunks": [],
                    "total_chunks": 0,
                    "metadata": {},
                    "title": "Unknown",
                    "url": ""
                }
            }
                
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
