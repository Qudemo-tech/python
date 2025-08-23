#!/usr/bin/env python3
"""
Enhanced Knowledge Integration System
Handles Pinecone vector database operations for knowledge storage and retrieval
"""

import requests
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pinecone import Pinecone
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
NODE_API_URL = os.getenv('NODE_API_URL', 'http://localhost:3001')
PYTHON_API_URL = os.getenv('PYTHON_API_URL', 'http://localhost:5000')

class EnhancedKnowledgeIntegrator:
    """Enhanced Knowledge Integrator for Pinecone operations with qudemo isolation"""
    
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index: str = None):
        """Initialize the knowledge integrator"""
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index = pinecone_index or os.getenv('PINECONE_INDEX', 'qudemo-index')
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
            logger.info(f"✅ Pinecone initialized with index: {self.pinecone_index}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {e}")
            raise
    
    def store_semantic_chunks(self, chunks: List[Dict], company_name: str, qudemo_id: str = None) -> Dict:
        """Store semantic chunks in Pinecone with qudemo isolation"""
        try:
            logger.info(f"🔧 Storing {len(chunks)} semantic chunks for {company_name} qudemo {qudemo_id}")
            
            # Create namespace for qudemo isolation
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}" if qudemo_id else company_name.lower().replace(' ', '-')
            
            # Get Pinecone index
            index = self.pc.Index(self.pinecone_index)
            
            vectors_to_upsert = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding for the chunk text
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk['text']
                    )
                    embedding = response.data[0].embedding
                    
                    # Prepare metadata
                    metadata = {
                        'text': chunk['text'],
                        'source': chunk.get('source', 'unknown'),
                        'title': chunk.get('title', ''),
                        'url': chunk.get('url', ''),
                        'processed_at': chunk.get('processed_at', datetime.now().isoformat()),
                        'company_name': company_name,
                        'qudemo_id': qudemo_id,
                        'chunk_type': 'semantic',
                        'start_timestamp': chunk.get('start_timestamp', 0),
                        'end_timestamp': chunk.get('end_timestamp', 0),
                        'chunk_index': chunk.get('chunk_index', 0),
                        'total_chunks': chunk.get('total_chunks', 1)
                    }
                    
                    # Create vector record
                    vector_record = {
                        'id': f"{namespace}-chunk-{i}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'values': embedding,
                        'metadata': metadata
                    }
                    
                    vectors_to_upsert.append(vector_record)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {i}: {e}")
                    continue
            
            if vectors_to_upsert:
                # Upsert vectors to Pinecone
                index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                logger.info(f"✅ Successfully stored {len(vectors_to_upsert)} chunks in namespace: {namespace}")
                
                return {
                    'success': True,
                    'chunks_stored': len(vectors_to_upsert),
                    'namespace': namespace,
                    'company_name': company_name,
                    'qudemo_id': qudemo_id
                }
            else:
                logger.warning("⚠️ No chunks to store")
                return {
                    'success': False,
                    'error': 'No chunks to store',
                    'chunks_stored': 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error storing semantic chunks: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunks_stored': 0
            }
    
    def search_with_context(self, query: str, company_name: str, qudemo_id: str = None, top_k: int = 5) -> Dict:
        """Search for relevant chunks with context"""
        try:
            # Create namespace for qudemo isolation
            namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}" if qudemo_id else company_name.lower().replace(' ', '-')
            
            # Generate query embedding
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Search in Pinecone
            index = self.pc.Index(self.pinecone_index)
            results = index.query(
                vector=query_embedding,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True
            )
            
            return {
                'success': True,
                'results': results.matches,
                'namespace': namespace,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"❌ Error searching with context: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def generate_answer(self, query: str, context_results: List, company_name: str, qudemo_id: str = None) -> Dict:
        """Generate answer using context from search results"""
        try:
            # Prepare context from search results
            context_texts = []
            for match in context_results:
                if hasattr(match, 'metadata') and match.metadata:
                    text = match.metadata.get('text', '')
                    source = match.metadata.get('source', '')
                    title = match.metadata.get('title', '')
                    url = match.metadata.get('url', '')
                    
                    context_texts.append(f"Source: {source}\nTitle: {title}\nURL: {url}\nContent: {text}\n")
            
            context = "\n".join(context_texts)
            
            # Generate answer using OpenAI
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Always cite sources when possible."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            return {
                'success': True,
                'answer': answer,
                'context_sources': len(context_results),
                'company_name': company_name,
                'qudemo_id': qudemo_id
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': 'Sorry, I encountered an error while generating the answer.'
            }

def process_qudemo_knowledge_source(qudemo_id, source_id, source_url, source_type, title):
    """
    Process a knowledge source for a specific qudemo
    """
    try:
        print(f"🔍 Processing knowledge source {source_id} for qudemo {qudemo_id}")
        
        # Call Python backend for processing
        python_payload = {
            "source_url": source_url,
            "source_type": source_type,
            "title": title,
            "qudemo_id": str(qudemo_id),
            "source_id": str(source_id)
        }
        
        response = requests.post(
            f"{PYTHON_API_URL}/process_knowledge_source",
            json=python_payload,
            headers={'Content-Type': 'application/json'},
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Update source status in Node.js database
            update_payload = {
                "status": "processed",
                "processed_at": datetime.now().isoformat(),
                "metadata": result.get("metadata", {})
            }
            
            update_response = requests.put(
                f"{NODE_API_URL}/api/qudemos/{qudemo_id}/knowledge/{source_id}/status",
                json=update_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if update_response.status_code == 200:
                print(f"✅ Successfully processed knowledge source {source_id}")
                return True
            else:
                print(f"❌ Failed to update source status: {update_response.text}")
                return False
        else:
            print(f"❌ Python processing failed: {response.text}")
            
            # Update source status to failed
            update_payload = {
                "status": "failed",
                "processed_at": datetime.now().isoformat(),
                "metadata": {"error": response.text}
            }
            
            requests.put(
                f"{NODE_API_URL}/api/qudemos/{qudemo_id}/knowledge/{source_id}/status",
                json=update_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            return False
            
    except Exception as e:
        print(f"❌ Error processing knowledge source: {str(e)}")
        
        # Update source status to failed
        try:
            update_payload = {
                "status": "failed",
                "processed_at": datetime.now().isoformat(),
                "metadata": {"error": str(e)}
            }
            
            requests.put(
                f"{NODE_API_URL}/api/qudemos/{qudemo_id}/knowledge/{source_id}/status",
                json=update_payload,
                headers={'Content-Type': 'application/json'}
            )
        except:
            pass
            
        return False

def get_pending_knowledge_sources():
    """
    Get all pending knowledge sources from Node.js backend
    """
    try:
        response = requests.get(
            f"{NODE_API_URL}/api/knowledge/pending",
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"❌ Failed to get pending sources: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error getting pending sources: {str(e)}")
        return []

def process_all_pending_sources():
    """
    Process all pending knowledge sources
    """
    pending_sources = get_pending_knowledge_sources()
    
    print(f"🔍 Found {len(pending_sources)} pending knowledge sources")
    
    for source in pending_sources:
        process_qudemo_knowledge_source(
            source["qudemo_id"],
            source["id"],
            source["source_url"],
            source["source_type"],
            source["title"]
        )

if __name__ == "__main__":
    process_all_pending_sources()
