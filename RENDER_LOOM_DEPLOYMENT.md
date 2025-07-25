# Render Deployment Guide for Loom Video Processing

## **🎯 Overview**

This guide covers deploying the QuDemo Python backend on Render with optimized Loom video processing capabilities.

## **✅ What Will Work on Render**

### **Loom Video Processing**
- ✅ **yt-dlp download**: Primary method (most reliable)
- ✅ **File validation**: Ensures downloaded files are actual videos
- ✅ **Whisper transcription**: Using tiny model for memory efficiency
- ✅ **Embedding generation**: Optimized batch sizes for Render
- ✅ **Memory management**: Stays under 400MB limit

### **Optimizations for Render**
- 🎬 **Lower video quality**: `worst[height<=480]` to save memory
- 📏 **File size limits**: 50MB max for Loom videos
- ⏱️ **Duration limits**: 5 minutes max for processing
- 🤖 **Tiny Whisper model**: Fast and memory-efficient
- 📦 **Smaller batches**: 3 embeddings per batch

## **🚀 Deployment Steps**

### **1. Update Render Configuration**

The `render.yaml` file has been updated with:
- FFmpeg installation for video processing
- Memory and environment optimizations
- Loom-specific configurations

### **2. Environment Variables**

Set these in your Render dashboard:

**Required:**
```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=your_pinecone_index
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
```

**Optional (for Loom):**
```bash
LOOM_API_KEY=your_loom_api_key  # Optional, yt-dlp works without it
```

**Render Optimizations:**
```bash
MAX_MEMORY_MB=400
NODE_ENV=production
```

### **3. Deploy to Render**

```bash
# Push your changes to GitHub
git add .
git commit -m "Add Render-optimized Loom video processing"
git push origin main

# Render will automatically deploy
```

## **🔧 Render-Specific Optimizations**

### **Memory Management**
- **Limit**: 400MB max memory usage
- **Monitoring**: Automatic memory tracking
- **Cleanup**: Temporary files removed after processing

### **Video Processing Limits**
- **Size**: 50MB max file size
- **Duration**: 5 minutes max
- **Quality**: 480p max resolution
- **Format**: MP4 preferred

### **Network Optimizations**
- **yt-dlp**: Primary download method
- **Fallbacks**: Disabled on Render to save memory
- **Timeouts**: 60 seconds for downloads

## **📊 Expected Performance**

### **Loom Video Processing**
- **Download time**: 10-30 seconds (depending on size)
- **Transcription time**: 30-60 seconds (tiny model)
- **Embedding time**: 10-20 seconds (3 batches)
- **Total time**: 1-2 minutes per video

### **Memory Usage**
- **Peak usage**: 300-400MB
- **Average usage**: 200-300MB
- **Cleanup**: Returns to ~150MB after processing

## **⚠️ Limitations on Render**

### **Video Size**
- **Loom videos**: Usually small (< 50MB) ✅
- **Large videos**: Will fail gracefully
- **Long videos**: Limited to 5 minutes

### **Concurrent Processing**
- **Single video**: One at a time
- **Queue**: Sequential processing
- **Timeout**: 10 minutes per request

### **Network Access**
- **yt-dlp**: Works reliably
- **Loom API**: May have rate limits
- **CDN access**: Restricted (handled by yt-dlp)

## **🔄 Fallback Behavior**

### **On Render (Optimized)**
1. **yt-dlp download** (primary)
2. **Fail gracefully** if yt-dlp fails
3. **No API/CDN fallbacks** (saves memory)

### **Development (Full Features)**
1. **yt-dlp download** (primary)
2. **API fallback** (if available)
3. **CDN fallback** (if needed)
4. **Standard video processing** (final fallback)

## **📝 Monitoring and Debugging**

### **Logs to Watch**
```bash
# Successful Loom processing
✅ Video info extracted: [Title] - Duration: [X]s
✅ Valid video file signature detected: [filename]
✅ Successfully downloaded using yt-dlp: [filename] ([size] bytes)
✅ Processed embedding batch [X]/[Y]

# Memory monitoring
💾 Memory usage: [X]MB
💾 After yt-dlp download - Memory: [X]MB
💾 After transcription - Memory: [X]MB

# Errors to watch for
❌ yt-dlp download failed: [error]
❌ Memory limit exceeded during Loom download
❌ Downloaded file failed validation
```

### **Common Issues**

#### **1. Memory Exceeded**
```
❌ Memory limit exceeded during Loom download
```
**Solution**: Video too large, will fail gracefully

#### **2. yt-dlp Failed**
```
❌ yt-dlp download failed: [error]
```
**Solution**: Check if Loom URL is accessible

#### **3. File Validation Failed**
```
❌ Downloaded file failed validation
```
**Solution**: Loom video may be restricted or corrupted

## **🎉 Success Indicators**

When Loom video processing works correctly on Render, you'll see:

1. **✅ Download successful**: yt-dlp downloads the video
2. **✅ File validated**: Downloaded file is confirmed as video
3. **✅ Transcription complete**: Whisper processes the audio
4. **✅ Embeddings generated**: OpenAI creates embeddings
5. **✅ Cleanup complete**: Temporary files removed
6. **✅ Memory stable**: Stays under 400MB throughout

## **🚀 Next Steps**

After successful deployment:

1. **Test with small Loom videos** (< 10MB, < 2 minutes)
2. **Monitor memory usage** in Render logs
3. **Check processing times** for optimization
4. **Scale up gradually** to larger videos
5. **Consider upgrading** to Render Pro for larger videos

## **📞 Support**

If you encounter issues:

1. **Check Render logs** for detailed error messages
2. **Verify environment variables** are set correctly
3. **Test with a simple Loom video** first
4. **Monitor memory usage** during processing
5. **Check yt-dlp version** is up to date

The system is designed to fail gracefully and provide detailed logging for troubleshooting. 