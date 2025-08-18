#!/usr/bin/env python3
"""
Enhanced QA System with Intelligent Scraping
Integrates web scraping with video transcripts for comprehensive support bot knowledge
"""

import asyncio
import json
import os
from typing import List, Dict, Optional
from enhanced_knowledge_integration import EnhancedKnowledgeIntegrator
from final_gemini_scraper import FinalGeminiScraper

class EnhancedQASystem:
    def __init__(self, gemini_api_key: str, openai_api_key: str):
        """Initialize enhanced QA system"""
        self.knowledge_integrator = EnhancedKnowledgeIntegrator(
            openai_api_key=openai_api_key,
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            pinecone_index=os.getenv('PINECONE_INDEX')
        )
        self.final_scraper = FinalGeminiScraper(gemini_api_key)
        
    async def process_website_knowledge(self, url: str, company_name: str) -> Dict:
        """Process website knowledge with semantic chunking (no Q&A generation)"""
        try:
            print(f"🚀 Processing website: {url} for company: {company_name}")
            
            # Extract content using the final scraper - comprehensive website crawling
            try:
                extracted_contents = await self.final_scraper.scrape_website_comprehensive(
                    url, 
                    max_collections=50,  # Allow up to 50 collections for comprehensive coverage
                    max_articles_per_collection=100  # Allow up to 100 articles per collection
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
                stored_chunks = self.knowledge_integrator.store_semantic_chunks(
                    text=content.get('content', ''),
                    company_name=company_name,
                    source_info=source_info
                )
                
                total_stored_chunks.extend(stored_chunks)
            
            # If we couldn't store any chunks, clean up and return error
            if not total_stored_chunks:
                print("❌ Failed to store any chunks in Pinecone")
                await self.cleanup_failed_website_data(url, company_name)
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
            await self.cleanup_failed_website_data(url, company_name)
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
            
            # Use hybrid search for better results
            search_results = self.knowledge_integrator.hybrid_search(
                query=question,
                company_name=company_name,
                top_k=20  # Get more results to ensure we have both video and website sources
            )
            
            if not search_results:
                return {
                    "success": False,
                    "message": "No relevant information found",
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }
            
            # Build context from search results
            context_parts = []
            sources = []
            video_timestamps = []
            video_sources = []
            website_sources = []
            
            for result in search_results:
                # Use full context for better answer generation
                context_parts.append(result['full_context'])
                
                # Add source information - use metadata directly
                metadata = result.get('metadata', {})
                
                source_data = {
                    'text': result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                    'score': result['score'],
                    'source_type': metadata.get('source', 'unknown'),
                    'title': metadata.get('title', 'Unknown'),
                    'url': metadata.get('url', ''),
                    'chunk_index': result.get('chunk_index', 0),
                    'total_chunks': result.get('total_chunks', 1)
                }
                
                print(f"🔍 Source: {source_data['source_type']} - {source_data['title']} - {source_data['url']}")
                print(f"🔍 Chunk {source_data['chunk_index']} selected with score {source_data['score']}")
                
                # Extract video timestamp if available
                if metadata.get('source') == 'video':
                    print(f"🔍 Processing video source: {metadata.get('title', 'Unknown')}")
                    # Look for timestamp in the chunk text or metadata
                    timestamp = self._extract_video_timestamp(result['text'], metadata)
                    if timestamp:
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
                answer = self.knowledge_integrator.generate_answer(prompt)
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
            results = self.knowledge_integrator.search_with_context(
                query="",  # Empty query to get all content
                company_name=company_name,
                top_k=1000  # Get a large number to get statistics
            )
            
            if not results:
                return {
                    "success": True,
                    "data": {
                        "company_name": company_name,
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
    
    def get_source_content(self, source_id: str, company_name: str) -> Optional[Dict]:
        """Get content from a specific source"""
        try:
            # URL decode the source_id if it's URL-encoded
            import urllib.parse
            decoded_source_id = urllib.parse.unquote(source_id)
            print(f"🔍 Retrieving source content: {source_id} (decoded: {decoded_source_id}) for company: {company_name}")
            
            # Query Pinecone for the specific source content
            results = self.knowledge_integrator.search_with_context(
                query="",  # Empty query to get all content
                company_name=company_name,
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
                        "text": combined_text,
                        "chunks": source_chunks,
                        "total_chunks": len(source_chunks),
                        "metadata": source_metadata,
                        "title": source_metadata.get('title', 'Unknown') if source_metadata else 'Unknown',
                        "url": source_metadata.get('url', '') if source_metadata else ''
                    }
                }
            
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
