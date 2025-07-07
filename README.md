# QuDemo Python Backend

This is the Python backend for the QuDemo application, handling video processing, transcription, and Q&A functionality using AI.

## Features

- **Video Processing**: Download and transcribe videos using yt-dlp and Whisper
- **Q&A System**: Answer questions about company videos using OpenAI GPT-4 and FAISS
- **FAISS Indexing**: Fast similarity search for video transcript chunks
- **Google Cloud Storage**: Store transcripts and indexes in GCS buckets
- **Supabase Integration**: Fetch video metadata from Supabase

## Setup

### 1. Environment Variables

Create a `.env` file in the `backend/python/` directory:

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
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Google Cloud Setup

1. Create a Google Cloud project
2. Enable Cloud Storage API
3. Create a service account and download the JSON key
4. Set the path to the key in `GOOGLE_APPLICATION_CREDENTIALS`

## Usage

### Start the API Server

```bash
python run.py
```

The API will be available at:
- **API**: http://localhost:5001
- **Docs**: http://localhost:5001/docs

### API Endpoints

#### Process Video
```bash
POST /process-video/{company_name}
```

Process a video for a specific company:
```json
{
  "video_url": "https://youtube.com/watch?v=...",
  "company_name": "Company Name",
  "bucket_name": "optional_bucket_name",
  "source": "optional_source",
  "meeting_link": "optional_meeting_link"
}
```

#### Ask Question
```bash
POST /ask/{company_name}
```

Ask a question about a company's video content:
```json
{
  "question": "What are the main features of this product?"
}
```

#### Health Check
```bash
GET /health
```

### Standalone Scripts

#### Process a Video
```bash
python process_video.py <video_url> <company_name> [source] [meeting_link]
```

#### Ask a Question
```bash
python ask_question.py <company_name> <question>
```

#### Rebuild FAISS Index
```bash
python rebuild_index.py <company_name>
```

List all companies:
```bash
python rebuild_index.py --list
```

## Architecture

### Core Components

1. **Video Processing Pipeline**:
   - Download video using yt-dlp
   - Transcribe using OpenAI Whisper
   - Create transcript chunks with timestamps
   - Upload to Google Cloud Storage

2. **Q&A System**:
   - Create embeddings for questions using OpenAI
   - Search FAISS index for relevant chunks
   - Rerank chunks using GPT-3.5
   - Generate answers using GPT-4

3. **Storage**:
   - Google Cloud Storage for transcripts and indexes
   - FAISS for fast similarity search
   - Supabase for video metadata

### File Structure

```
backend/python/
├── main.py              # Main FastAPI application
├── run.py               # Server startup script
├── requirements.txt     # Python dependencies
├── ask_question.py      # Standalone Q&A script
├── process_video.py     # Standalone video processing script
├── rebuild_index.py     # FAISS index rebuild script
├── README.md           # This file
├── key.json            # Google Cloud service account key
└── env/                # Virtual environment
```

## Troubleshooting

### Common Issues

1. **NumPy Version Conflicts**: If you get numpy-related errors, try:
   ```bash
   pip uninstall numpy faiss-cpu
   pip install numpy==1.24.4
   pip install faiss-cpu
   ```

2. **Google Cloud Authentication**: Ensure your service account key is valid and has the necessary permissions.

3. **OpenAI API**: Make sure your OpenAI API key is valid and has sufficient credits.

4. **Port Conflicts**: If port 5001 is in use, change the PORT environment variable.

### Logs

The application uses structured logging. Check the console output for detailed information about:
- Video processing progress
- Q&A operations
- API requests
- Error messages

## Development

### Adding New Features

1. Add new endpoints in `main.py`
2. Update the Pydantic models for request/response validation
3. Add any new dependencies to `requirements.txt`
4. Update this README with new usage instructions

### Testing

Test the API using the interactive docs at http://localhost:5001/docs

### Deployment

For production deployment:
1. Set `DEBUG=False`
2. Use a proper WSGI server like Gunicorn
3. Set up proper CORS configuration
4. Use environment-specific configuration 