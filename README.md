# QuDemo Python Backend

Production-ready Python backend for the QuDemo application, handling video processing, transcription, and Q&A functionality using AI with optimized settings for Render deployment.

## 🚀 Features

- **Video Processing**: Download and transcribe videos using yt-dlp and Whisper
- **Audio Conversion**: Convert video to audio for memory optimization
- **Q&A System**: Answer questions about company videos using OpenAI GPT-4 and FAISS
- **FAISS Indexing**: Fast similarity search for video transcript chunks
- **Google Cloud Storage**: Store transcripts and indexes in GCS buckets
- **Supabase Integration**: Fetch video metadata from Supabase
- **Memory Optimization**: Optimized for Render 2GB RAM deployment

## 📁 Project Structure

```
pythonn/
├── main.py                    # FastAPI server
├── loom_video_processor.py    # Loom video processing with audio conversion
├── render_deployment_config.py # Production configuration
├── requirements-python312.txt  # Dependencies
├── README.md                  # This file
├── PRODUCTION_DEPLOYMENT_2GB.md # Production deployment guide
└── venv/                     # Virtual environment
```

## 🛠️ Setup

### 1. Environment Variables

Create a `.env` file in the `backend/pythonn/` directory:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Google Cloud Storage
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json

# Supabase
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Server Configuration
HOST=0.0.0.0
PORT=5001
DEBUG=False

# Render Environment (for production)
RENDER=true
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements-python312.txt
```

### 3. Start the API Server

```bash
# Development
python main.py

# Production (with RENDER=true)
RENDER=true python main.py
```

The API will be available at:
- **API**: http://localhost:5001
- **Docs**: http://localhost:5001/docs

## 🎯 Production Optimizations

### Memory Management
- **Audio Conversion**: Convert video to 16kHz mono audio before Whisper processing
- **Memory Limits**: 1.8GB max for Render 2GB RAM plan
- **Video Quality**: Force 240p downloads for minimal memory usage
- **Whisper Model**: 'tiny' model for fastest processing

### Performance Settings
```python
{
    'max_memory_mb': 1800,
    'ytdlp_format': 'worst[height<=240]',
    'whisper_model': 'tiny',
    'embedding_batch_size': 8,
    'audio_sample_rate': 16000,
    'audio_channels': 1,
}
```

## 📊 API Endpoints

### Process Video
```bash
POST /process-video/{company_name}
```

Process a video for a specific company:
```json
{
  "video_url": "https://www.loom.com/share/...",
  "company_name": "Company Name",
  "is_youtube": true,
  "source": null,
  "meeting_link": ""
}
```

### Ask Question
```bash
POST /ask/{company_name}
```

Ask a question about a company's video content:
```json
{
  "question": "What are the main features of this product?"
}
```

## 🔧 Production Deployment

### Render Deployment
1. Set `RENDER=true` environment variable
2. Deploy with Python 3.12
3. Install FFmpeg dependency
4. Use `requirements-python312.txt`

### Expected Performance
- **Memory Usage**: 800-1000MB peak
- **Processing Time**: 25-65 seconds per video
- **Success Rate**: >95%
- **Concurrent Requests**: 2-3 videos simultaneously

## 📋 Dependencies

- **FastAPI**: Web framework
- **Whisper**: Audio transcription
- **yt-dlp**: Video downloading
- **FFmpeg**: Audio conversion
- **OpenAI**: Embeddings and Q&A
- **Supabase**: Database integration
- **Pinecone**: Vector storage

## 🚨 Troubleshooting

### Memory Issues
- Check video file size (max 30MB for production)
- Verify 240p format is being used
- Monitor memory usage in logs

### Processing Failures
- Check Loom video accessibility
- Verify FFmpeg installation
- Check OpenAI API key

### Performance Issues
- Ensure `RENDER=true` is set for production
- Check network connectivity
- Monitor CPU usage

## 📈 Monitoring

Key metrics to monitor:
- Memory usage (should stay under 1.5GB)
- Processing time (25-65 seconds)
- Success rate (>95%)
- Error rate (<5%)

For detailed deployment information, see `PRODUCTION_DEPLOYMENT_2GB.md`. 