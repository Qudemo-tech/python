# QuDemo Gemini Branch

This branch integrates Google Gemini API for YouTube video transcription while maintaining Loom video processing functionality.

## 🚀 What's New

### ✅ **Gemini Integration**
- **YouTube Video Processing**: Uses Google Gemini API to extract transcriptions from YouTube videos
- **No Bot Detection**: Eliminates the need for complex bot detection bypass techniques
- **Reliable Transcription**: Direct API access to YouTube content through Gemini

### ✅ **Enhanced Loom Processing**
- **Dedicated Loom Processor**: Separate module for Loom video processing
- **Whisper Integration**: Uses OpenAI Whisper for Loom video transcription
- **Consistent Vector Storage**: Both YouTube and Loom videos stored in Pinecone

### ✅ **Clean Architecture**
- **Modular Design**: Separate processors for different video types
- **Unified Interface**: Both processors use the same vector storage and search interface
- **Removed Dependencies**: Eliminated yt-dlp, Selenium, and other complex bypass tools

## 📁 File Structure

```
backend/pythonn/
├── main.py                    # Main FastAPI application
├── gemini_transcription.py    # YouTube video processing with Gemini
├── loom_processor.py          # Loom video processing with Whisper
├── requirements-python312.txt # Updated dependencies
├── test_integration.py        # Integration tests
└── README_GEMINI_BRANCH.md   # This file
```

## 🔧 Setup

### 1. Environment Variables
Create a `.env` file with the following API keys:

```env
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for Supabase integration)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 2. Install Dependencies
```bash
pip install -r requirements-python312.txt
```

### 3. Test Integration
```bash
python test_integration.py
```

## 🎯 How It Works

### YouTube Video Processing
1. **URL Detection**: Automatically detects YouTube URLs (`youtube.com` or `youtu.be`)
2. **Gemini Transcription**: Sends YouTube URL to Gemini API for transcription
3. **Text Chunking**: Splits transcription into overlapping chunks
4. **Vector Creation**: Creates embeddings using OpenAI
5. **Pinecone Storage**: Stores vectors in company-specific Pinecone index

### Loom Video Processing
1. **URL Detection**: Detects Loom URLs (`loom.com`)
2. **Video Download**: Downloads Loom video using their API
3. **Whisper Transcription**: Uses OpenAI Whisper for transcription
4. **Text Chunking**: Splits transcription into overlapping chunks
5. **Vector Creation**: Creates embeddings using OpenAI
6. **Pinecone Storage**: Stores vectors in company-specific Pinecone index

### Question Answering
1. **Query Processing**: Creates embedding for user question
2. **Vector Search**: Searches Pinecone for similar chunks
3. **Context Assembly**: Combines relevant chunks as context
4. **Answer Generation**: Uses OpenAI GPT to generate answer
5. **Source Attribution**: Returns sources with video URLs and titles

## 🚀 API Endpoints

### Process Video
```http
POST /process-video/{company_name}
Content-Type: application/json

{
  "video_url": "https://www.youtube.com/watch?v=...",
  "company_name": "MyCompany",
  "bucket_name": "optional-bucket-name",
  "source": "optional-source",
  "meeting_link": "optional-meeting-link"
}
```

### Ask Question
```http
POST /ask-question
Content-Type: application/json

{
  "question": "What features does this product have?",
  "company_name": "MyCompany"
}
```

### Ask Question (Company-specific)
```http
POST /ask/{company_name}
Content-Type: application/json

{
  "question": "What features does this product have?"
}
```

### Generate Summary
```http
POST /generate-summary
Content-Type: application/json

{
  "questions_and_answers": [
    {"question": "Q1", "answer": "A1"},
    {"question": "Q2", "answer": "A2"}
  ],
  "company_name": "MyCompany",
  "buyer_name": "John Doe"
}
```

### Status Check
```http
GET /status
```

## 🧪 Testing

### Integration Test
```bash
python test_integration.py
```

### Manual Testing
1. Start the server: `python main.py`
2. Test with a YouTube URL: `POST /process-video/TestCompany`
3. Ask a question: `POST /ask-question`
4. Check status: `GET /status`

## 🔍 Debug Endpoints

- `GET /debug/qa-test/{company_name}` - Test Q&A functionality
- `GET /debug/videos` - Show video mappings
- `GET /memory-status` - Check memory usage

## 📊 Monitoring

### Status Endpoint Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "processors": {
    "gemini": "✅ Available",
    "loom": "✅ Available"
  },
  "api_keys": {
    "gemini": "✅ Set",
    "pinecone": "✅ Set",
    "openai": "✅ Set"
  },
  "memory_mb": 245.6,
  "video_mappings": 15,
  "python_version": "3.12.0",
  "environment": "development"
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing API Keys**
   - Ensure all required API keys are set in `.env`
   - Check `/status` endpoint for key status

2. **Processor Initialization Failed**
   - Verify API keys are valid
   - Check network connectivity
   - Review logs for specific errors

3. **Video Processing Fails**
   - Check video URL format
   - Ensure video is publicly accessible
   - Review processor-specific logs

4. **Pinecone Storage Issues**
   - Verify Pinecone API key
   - Check index creation permissions
   - Review vector storage logs

### Log Levels
- `INFO`: Normal operation logs
- `WARNING`: Non-critical issues
- `ERROR`: Processing failures
- `DEBUG`: Detailed debugging (if enabled)

## 🔄 Migration from Previous Version

### What Changed
- ✅ Removed complex bot detection bypass
- ✅ Eliminated yt-dlp and Selenium dependencies
- ✅ Added Gemini API integration
- ✅ Separated Loom processing into dedicated module
- ✅ Simplified architecture and dependencies

### What Remains the Same
- ✅ Question answering functionality
- ✅ Pinecone vector storage
- ✅ Supabase integration
- ✅ API endpoints structure
- ✅ Video URL mapping system

## 📈 Performance

### Memory Usage
- **Gemini Processing**: ~50-100MB per video
- **Loom Processing**: ~200-500MB per video (includes Whisper model)
- **Vector Storage**: ~1KB per chunk

### Processing Speed
- **YouTube (Gemini)**: 10-30 seconds per video
- **Loom (Whisper)**: 30-120 seconds per video (depends on length)

### Scalability
- **Concurrent Processing**: Rate limited to 1 request per 5 seconds per company
- **Vector Storage**: Pinecone handles up to millions of vectors
- **Memory Management**: Automatic cleanup of temporary files

## 🔐 Security

### API Key Management
- Store API keys in environment variables
- Never commit keys to version control
- Use different keys for development and production

### Data Privacy
- Video content is processed but not permanently stored
- Only transcription chunks are stored in Pinecone
- Temporary video files are automatically cleaned up

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production (Render)
- Use `render.yaml` for deployment configuration
- Set environment variables in Render dashboard
- Monitor memory usage and logs

### Docker (Optional)
```bash
docker build -t qudemo-gemini .
docker run -p 5001:5001 qudemo-gemini
```

## 📝 License

This project is part of the QuDemo platform. See main project license for details.

---

**🎉 The Gemini branch is ready for production use!**
