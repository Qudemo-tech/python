# QuDemo AI-Powered Q&A System

## 🚀 System Overview

QuDemo is an intelligent Q&A system that processes video content and web-scraped data to provide contextual, timestamped answers. The system leverages OpenAI embeddings, Gemini AI for video transcription, and Pinecone for vector storage to deliver accurate, source-attributed responses.

## 🏗️ System Architecture

### High-Level Architecture
```
User Question → Frontend → Backend API → Q&A Engine → Pinecone Search → Answer Generation → Response
                                    ↓
                            Video Processing + Web Scraping → Storage → Vector Embeddings
```

### Core Components
1. **FastAPI Backend** (`main.py`) - RESTful API server
2. **Enhanced Q&A System** (`enhanced_qa_simple.py`) - Intelligent answer selection
3. **Video Processor** (`enhanced_video_processor.py`) - Video transcription & chunking
4. **Web Scraper** (`final_gemini_scraper.py`) - Website content extraction
5. **Pinecone Manager** (`enhanced_pinecone_manager.py`) - Vector database operations
6. **Knowledge Integration** (`enhanced_knowledge_integration.py`) - Content storage & retrieval

---

## 🎬 Video Data Processing

### Video Processing Pipeline

#### 1. **Video Input & Validation**
- Supports YouTube and Loom videos
- URL validation and format detection
- Automatic video type identification

#### 2. **Transcription Process**
```python
# Using Gemini AI for transcription
gemini_processor = GeminiTranscriptionProcessor()
transcript = await gemini_processor.process_video_with_qudemo(
    video_url, company_name, qudemo_id
)
```

#### 3. **Content Chunking Strategy**
- **Chunk Size**: 1000 characters per chunk
- **Overlap**: 200 characters between chunks
- **Metadata**: Rich context including timestamps, video type, company info

#### 4. **Chunk Processing**
```python
chunks = [
    {
        'text': 'chunk_content',
        'start_timestamp': 120,  # 2:00
        'end_timestamp': 150,    # 2:30
        'chunk_index': 0,
        'total_chunks': 15,
        'video_url': 'https://...',
        'video_type': 'youtube|loom'
    }
]
```

#### 5. **Storage Optimization**
- **Pinecone Index**: `qudemo-video-index`
- **Namespace**: `{company_name}-{qudemo_id}`
- **Metadata**: Comprehensive video context for efficient retrieval

---

## 🌐 Web Scraping & Data Processing

### Scraping Architecture

#### 1. **Multi-Strategy Approach**
- **Primary**: Playwright-based dynamic scraping
- **Fallback**: BeautifulSoup static extraction
- **Enhanced**: Multi-selector article detection

#### 2. **Content Extraction Methods**
```python
# Primary scraping with Playwright
async def scrape_website_comprehensive(self, url: str):
    # Dynamic content rendering
    # JavaScript execution
    # Interactive element handling
    
# Fallback with BeautifulSoup
async def scrape_with_fallback(self, url: str):
    # Static HTML parsing
    # Multiple selector strategies
    # Content cleaning & formatting
```

#### 3. **Article Detection**
- **Selectors**: `article`, `.article`, `.post`, `.help-item`
- **Content Filtering**: Minimum 100 words, relevant content
- **Quality Scoring**: Relevance, completeness, step-by-step content

#### 4. **Data Processing**
- **HTML Cleaning**: Remove navigation, headers, footers
- **Content Structuring**: Organize into logical sections
- **Metadata Extraction**: Title, URL, content type, word count

---

## 🗄️ Pinecone Storage Architecture

### Storage Strategy

#### 1. **Multi-Index Architecture (Standard Plan)**
```python
self.indexes = {
    'video': 'qudemo-video-index',      # Video transcripts
    'knowledge': 'qudemo-knowledge-index', # Web scraped content
    'web': 'qudemo-web-index',          # Additional web content
    'legacy': 'qudemo-index'            # Fallback for existing content
}
```

#### 2. **Namespace Isolation**
```python
# Company and QuDemo specific namespaces
namespace = f"{company_name.lower().replace(' ', '-')}-{qudemo_id}"

# Example: "acme-corp-qudemo-123"
# Ensures complete data separation between companies and QuDemos
```

#### 3. **Content Routing**
```python
def route_content_to_index(self, content_type: str) -> str:
    if content_type in ['video_transcript', 'youtube_transcript', 'loom_transcript']:
        return 'qudemo-video-index'
    elif content_type in ['web_scraping', 'article']:
        return 'qudemo-knowledge-index'
    else:
        return 'qudemo-index'  # Fallback
```

#### 4. **Vector Embeddings**
- **Model**: `text-embedding-3-large` (3072 dimensions)
- **Optimization**: High-dimensional vectors for better semantic understanding
- **Fallback**: Dummy embeddings (3072 dimensions) if OpenAI fails

---

## 🔍 Answer Retrieval Logic

### Intelligent Answer Selection

#### 1. **Relevance Scoring System**
```python
def _calculate_relevance_score(self, question: str, content: str) -> float:
    # Semantic score (base weight: 0.6)
    semantic_score = 0.6
    
    # Keyword matching (weight: 0.2)
    keyword_score = min(keyword_matches / total_terms, 1.0) * 0.2
    
    # Required terms (weight: 0.2)
    required_score = (required_matches / total_required) * 0.2
    
    # Negative penalty (up to -0.45)
    negative_penalty = sum(0.15 for term in negative_terms if term in content)
    
    return max(0.0, min(1.0, semantic_score + keyword_score + required_score - negative_penalty))
```

#### 2. **Decision Matrix**
```python
# Relevance thresholds
HIGH_RELEVANCE = 0.7      # Excellent match
MEDIUM_RELEVANCE = 0.6    # Good match
LOW_RELEVANCE = 0.4       # Acceptable match

# Decision logic
if knowledge_score >= HIGH_RELEVANCE and video_score < MEDIUM_RELEVANCE:
    # Use knowledge only (highly relevant, video not relevant enough)
elif video_score >= HIGH_RELEVANCE and knowledge_score >= HIGH_RELEVANCE:
    # Combine both sources (both highly relevant)
elif video_score >= HIGH_RELEVANCE and knowledge_score < MEDIUM_RELEVANCE:
    # Use video only (video highly relevant, knowledge not relevant enough)
```

#### 3. **Answer Generation Strategies**

##### **Knowledge-Only Answer**
- **Trigger**: Knowledge score ≥ 0.7, Video score < 0.6
- **Process**: GPT formatting of scraped content
- **Output**: Structured, step-by-step instructions
- **Source**: `[SOURCE: Scraped Data]`

##### **Video-Only Answer**
- **Trigger**: Video score ≥ 0.7, Knowledge score < 0.6
- **Process**: GPT-guided answer from video transcript
- **Output**: Clear, actionable guidance
- **Source**: `[SOURCE: Video Data]`

##### **Combined Answer**
- **Trigger**: Both scores ≥ 0.7
- **Process**: GPT combination of both sources
- **Output**: Comprehensive, non-repetitive answer
- **Source**: `[SOURCE: Video + Scraped Data]`

#### 4. **Fallback Logic**
```python
# Score difference analysis
score_difference = knowledge_score - video_score

if score_difference >= 0.1:  # Knowledge significantly better
    return knowledge_answer
elif video_score < 0.65:     # Video not relevant enough
    return knowledge_answer
else:                        # Both similarly relevant
    return combined_answer
```

---

## 📊 Pinecone Standard Plan vs Free Trial

### **Free Trial Limitations**
- **Single Index**: All content in one index
- **Limited Namespaces**: Basic namespace support
- **Storage Constraints**: Limited vector storage
- **Performance**: Basic query performance
- **Scalability**: Limited to small datasets

### **Standard Plan Benefits**

#### 1. **Multi-Index Architecture**
```python
# Separate indexes for different content types
'qudemo-video-index'      # Video transcripts (optimized for video search)
'qudemo-knowledge-index'  # Web content (optimized for knowledge search)
'qudemo-web-index'        # Additional web content
'qudemo-index'            # Legacy/fallback content
```

#### 2. **Enhanced Namespace Management**
```python
# Company-specific isolation
namespace = f"{company_name}-{qudemo_id}"

# Benefits:
# - Complete data separation between companies
# - Multiple QuDemos per company
# - Secure, isolated data access
# - No cross-contamination of content
```

#### 3. **Performance Improvements**
- **Faster Queries**: Index-specific optimizations
- **Better Relevance**: Content-type specific vector models
- **Reduced Latency**: Targeted search in relevant indexes
- **Scalability**: Handle multiple companies and QuDemos

#### 4. **Storage Efficiency**
- **Optimized Chunking**: Content-type specific chunk sizes
- **Metadata Optimization**: Rich context for better retrieval
- **Vector Dimensions**: 3072 dimensions for superior semantic understanding

---

## 🔄 Data Flow Architecture

### **Content Processing Flow**

#### 1. **Video Processing Pipeline**
```
Video URL → Validation → Gemini Transcription → Chunking → Pinecone Storage
    ↓
Metadata: timestamps, video_type, company, qudemo_id, chunk_index
```

#### 2. **Web Scraping Pipeline**
```
Website URL → Multi-Strategy Scraping → Content Extraction → Cleaning → Storage
    ↓
Metadata: title, url, content_type, word_count, quality_score
```

#### 3. **Q&A Processing Flow**
```
User Question → Embedding → Multi-Index Search → Relevance Scoring → Answer Selection
    ↓
Response: answer, sources, timestamps, source attribution
```

---

## 🎯 System Benefits

### **1. Intelligent Answer Selection**
- **Context-Aware**: Considers both video and knowledge relevance
- **Source Attribution**: Clear indication of answer sources
- **Quality Optimization**: Always selects the most relevant content

### **2. Scalable Architecture**
- **Multi-Company Support**: Complete isolation between companies
- **Multiple QuDemos**: Independent content management per QuDemo
- **Performance**: Optimized for large-scale deployments

### **3. Rich Content Processing**
- **Video Intelligence**: Timestamp-based video navigation
- **Web Content**: Comprehensive website scraping
- **Metadata Rich**: Extensive context for better search

### **4. Production Ready**
- **Error Handling**: Comprehensive fallback mechanisms
- **Logging**: Detailed system monitoring
- **API Design**: RESTful, scalable endpoints

---

## 🚀 Getting Started

### **1. Environment Setup**
```bash
# Install dependencies
pip install -r requirements-python312.txt

# Set environment variables
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
```

### **2. Initialize System**
```python
# Start the FastAPI server
python main.py

# System will automatically initialize:
# - Enhanced Pinecone Manager
# - Enhanced Knowledge Integration
# - Enhanced Q&A System
# - Enhanced Video Processor
```

### **3. API Endpoints**
- `POST /ask/{company_name}/{qudemo_id}` - Ask questions
- `POST /process-qudemo-content/{company_name}/{qudemo_id}` - Process content
- `GET /health` - System health check

---

## 📈 Performance Metrics

### **Processing Times**
- **Video Processing**: 1-3 minutes per video
- **Web Scraping**: 3-10 minutes per website
- **Q&A Response**: 2-5 seconds per question

### **Storage Efficiency**
- **Video Chunks**: ~1000 characters with 200 character overlap
- **Web Content**: Optimized for knowledge retrieval
- **Vector Storage**: 3072 dimensions for superior semantic search

### **Scalability**
- **Companies**: Unlimited (namespace isolation)
- **QuDemos per Company**: Unlimited
- **Content Types**: Video, web, knowledge, combined

---

## 🔧 Technical Specifications

### **System Requirements**
- **Python**: 3.12+
- **Memory**: 4GB+ RAM
- **Storage**: SSD recommended for Pinecone operations
- **Network**: Stable internet for API calls

### **Dependencies**
- **FastAPI**: Modern web framework
- **OpenAI**: Embeddings and GPT models
- **Gemini**: Video transcription
- **Pinecone**: Vector database
- **Playwright**: Web scraping
- **BeautifulSoup**: HTML parsing

### **API Rate Limits**
- **OpenAI**: Based on your plan
- **Gemini**: Based on your plan
- **Pinecone**: Standard Plan limits

---

## 🎉 Conclusion

The QuDemo system represents a significant advancement in AI-powered Q&A technology, combining:

- **Intelligent Content Processing**: Video and web content with context
- **Advanced Vector Storage**: Pinecone Standard Plan multi-index architecture
- **Smart Answer Selection**: Relevance-based answer generation
- **Scalable Architecture**: Multi-company, multi-QuDemo support
- **Production Ready**: Comprehensive error handling and monitoring

The upgraded Pinecone Standard Plan provides the foundation for enterprise-grade scalability, performance, and data isolation that the free trial simply cannot match.

---

*For technical support or questions, please refer to the system logs and health check endpoints.*
