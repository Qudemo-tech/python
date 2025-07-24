# Memory Optimization Guide for 512MB Render Instance

## Problem
Your Python backend is running out of memory (512MB limit) when processing videos, causing the application to crash.

## Solutions Implemented

### 1. **Video Download Optimization**
- **YouTube videos**: Limited to 480p quality (`worst[height<=480]`)
- **File size limit**: Maximum 50MB per video
- **Streaming downloads**: Avoid loading entire files into memory
- **Timeout**: 30-second timeout for downloads

### 2. **Whisper Model Optimization**
- **Model size**: Switched from `base` to `tiny` model
- **Memory usage**: Tiny model uses ~39MB vs Base model ~74MB
- **Accuracy**: Slightly reduced but still good for most use cases

### 3. **Memory Management**
- **Immediate cleanup**: Delete video files after transcription
- **Smaller batches**: Reduced embedding batch size from 10 to 5
- **Memory monitoring**: Added `psutil` for real-time memory tracking
- **Error cleanup**: Ensure files are deleted even on errors

### 4. **Processing Flow Optimization**
```
1. Download video (max 50MB, 480p for YouTube)
2. Transcribe with tiny Whisper model
3. Delete video file immediately
4. Process embeddings in small batches (5 texts)
5. Upload to Pinecone
6. Clean up any remaining files
```

## Memory Usage Breakdown

| Component | Memory Usage |
|-----------|-------------|
| FastAPI + Dependencies | ~50MB |
| Whisper Tiny Model | ~39MB |
| Video file (50MB max) | ~50MB |
| Embeddings processing | ~20-50MB |
| System overhead | ~50MB |
| **Total** | **~209-239MB** |

## Testing

Run the memory test script:
```bash
cd backend/pythonn
python test_memory.py
```

## Additional Recommendations

### 1. **Upgrade Render Plan**
Consider upgrading to a 1GB or 2GB plan for better performance:
- **1GB plan**: $7/month
- **2GB plan**: $15/month

### 2. **Alternative Solutions**
- **Use external video processing**: Process videos on a separate service
- **Stream processing**: Process videos in chunks without downloading
- **Cloud transcription**: Use Google Speech-to-Text or AWS Transcribe

### 3. **Monitoring**
The app now logs memory usage at key points:
```
💾 Memory usage: 150.2MB
📥 Downloading video: https://youtu.be/...
🗑️ Cleaned up video file: downloaded_video_abc123.mp4
💾 Memory usage: 120.1MB
```

## Configuration

### Environment Variables
```bash
# Memory limits
MAX_VIDEO_SIZE_MB=50
WHISPER_MODEL=tiny
EMBEDDING_BATCH_SIZE=5
```

### Render Configuration
```yaml
# render.yaml
services:
  - type: web
    name: python-backend
    env: python
    plan: starter  # 512MB
    # Consider upgrading to: plan: standard  # 1GB
```

## Troubleshooting

### If you still get memory errors:

1. **Check memory usage**:
   ```bash
   python test_memory.py
   ```

2. **Reduce video quality further**:
   ```python
   ydl_opts = {
       'format': 'worst[height<=360]',  # Even lower quality
       'max_filesize': '25M',  # Smaller file size
   }
   ```

3. **Use audio-only processing**:
   ```python
   ydl_opts = {
       'extractaudio': True,
       'audioformat': 'mp3',
       'max_filesize': '10M',
   }
   ```

4. **Process in background**:
   - Queue videos for processing
   - Process one at a time
   - Use a job queue system

## Performance Tips

1. **Monitor logs** for memory usage patterns
2. **Test with small videos** first
3. **Consider video length limits** (e.g., max 10 minutes)
4. **Use compression** for uploaded videos
5. **Implement retry logic** for failed processing

## Success Metrics

After optimization, you should see:
- ✅ Memory usage stays under 400MB
- ✅ Videos process successfully
- ✅ No out-of-memory crashes
- ✅ Faster processing times
- ✅ More reliable service 