# 🚀 Deploy Your Existing Python Backend

## 📁 Push This Directory: `backend/pythonn/`

Your existing Python backend is **perfect** for video processing! It already has:
- ✅ Video download with yt-dlp
- ✅ Cookie support for restricted videos
- ✅ Multiple video processors (Loom, Vimeo)
- ✅ FastAPI framework
- ✅ AI integration (OpenAI, Whisper, Pinecone)

## 🚀 Quick Deploy Steps:

### Step 1: Navigate to Python Backend
```bash
cd backend/pythonn
```

### Step 2: Initialize Git (if not already done)
```bash
git init
git add .
git commit -m "Deploy video processing backend"
```

### Step 3: Create GitHub Repository
```bash
git remote add origin https://github.com/YOUR_USERNAME/qudemo-video-backend.git
git push -u origin main
```

### Step 4: Deploy to Render
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Use these settings:
   - **Name**: `qudemo-video-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-python312.txt`
   - **Start Command**: `python main.py`
   - **Plan**: `Starter` (or higher)

### Step 5: Set Environment Variables in Render
Add these environment variables:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `SUPABASE_URL`
- `SUPABASE_KEY`

## 🎯 Test Your Deployment:

```bash
# Health check
curl "https://your-app.onrender.com/health"

# Process a video
curl -X POST "https://your-app.onrender.com/process-video/test-company" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://youtu.be/ZAGxqOT2l2U?si=uB03UNTGKGzgIJ7L",
    "company_name": "test-company",
    "is_loom": false
  }'

# Ask a question about the video
curl -X POST "https://your-app.onrender.com/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this video about?",
    "company_name": "test-company"
  }'
```

## 🔧 Why This is Better:

1. **Already Built** - Your backend is production-ready
2. **Video Processing** - Handles YouTube, Loom, Vimeo, and more
3. **AI Integration** - Transcribes and analyzes videos
4. **Cookie Support** - Can handle restricted videos
5. **Scalable** - FastAPI with async support

## 📊 Supported Features:
- ✅ Video download from multiple platforms
- ✅ Video transcription with Whisper
- ✅ AI-powered Q&A about videos
- ✅ Vector search with Pinecone
- ✅ Company-specific video organization
- ✅ Cookie authentication for restricted content

## 🛠️ Troubleshooting:
- Check Render logs for any issues
- Ensure all environment variables are set
- For restricted videos, make sure `cookies.txt` is uploaded 