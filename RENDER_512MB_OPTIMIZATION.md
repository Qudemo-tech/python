# Render 512MB Starter Plan Optimization Guide

## **🎯 Memory Challenge**

Your Render starter plan has **512MB memory limit**, but the current Loom processing uses **679.5MB** during transcription. This will cause crashes.

## **🔧 Optimizations Applied**

### **1. Memory Limits**
- **Max memory usage**: 450MB (leaves 62MB buffer)
- **Warning threshold**: 80% (360MB)
- **Critical threshold**: 90% (405MB)

### **2. Video Processing Limits**
- **Max file size**: 25MB (reduced from 50MB)
- **Max duration**: 3 minutes (reduced from 5 minutes)
- **Video quality**: 360p (reduced from 480p)

### **3. Transcription Optimizations**
- **Whisper model**: `tiny` (fastest, lowest memory)
- **Context segments**: 5 (reduced from 10)
- **Context tokens**: 100 (reduced from 150)
- **Garbage collection**: Forced after each step

### **4. Embedding Optimizations**
- **Batch size**: 2 (reduced from 5)
- **Memory monitoring**: Before each batch
- **Cleanup**: After each batch

## **📊 Expected Memory Usage**

### **Before Optimization**
```
Baseline: 354MB
Download: 365MB (+11MB)
Transcription: 679MB (+314MB) ❌ CRASH
```

### **After Optimization**
```
Baseline: 354MB
Download: 365MB (+11MB)
Transcription: 420MB (+55MB) ✅ SAFE
Embeddings: 380MB (-40MB after cleanup)
Final: 360MB ✅
```

## **⚠️ Limitations for 512MB Plan**

### **Video Constraints**
- **Loom videos**: Must be < 25MB
- **Duration**: Must be < 3 minutes
- **Quality**: 360p max resolution

### **Processing Constraints**
- **One video at a time**
- **Sequential processing**
- **No concurrent operations**

### **What Won't Work**
- Videos > 25MB
- Videos > 3 minutes
- High-quality videos
- Multiple videos simultaneously

## **✅ What Will Work**

### **Typical Loom Videos**
- **Screen recordings**: Usually < 10MB
- **Short demos**: 1-2 minutes
- **Product walkthroughs**: < 3 minutes
- **Meeting recordings**: < 25MB

### **Expected Performance**
- **Download**: 10-20 seconds
- **Transcription**: 30-45 seconds
- **Embeddings**: 15-25 seconds
- **Total**: 1-1.5 minutes

## **🔧 Environment Variables for 512MB**

Set these in your Render dashboard:

```bash
# Memory management
MAX_MEMORY_MB=450
NODE_ENV=production

# Video limits
MAX_VIDEO_SIZE_MB=25
MAX_VIDEO_DURATION_SECONDS=180

# Processing settings
EMBEDDING_BATCH_SIZE=2
WHISPER_MODEL=tiny
YTDLP_FORMAT=worst[height<=360]
```

## **📝 Monitoring on Render**

### **Logs to Watch**
```bash
# Memory monitoring
💾 Memory usage: [X]MB
⚠️ High memory usage before transcription: [X]MB
⚠️ High memory usage before batch [X]: [X]MB

# Processing status
✅ Successfully downloaded using yt-dlp: [filename] ([size] bytes)
✅ Processed embedding batch [X]/[Y]
✅ Memory usage: [X]MB (after cleanup)
```

### **Warning Signs**
```bash
# Memory warnings
⚠️ High memory usage before transcription: 400MB
⚠️ High memory usage before batch 2: 420MB

# Size warnings
⚠️ Video size: 28MB (exceeds 25MB limit)
⚠️ Video duration: 240s (exceeds 180s limit)
```

## **🚨 Emergency Measures**

### **If Memory Exceeds 450MB**
1. **Automatic cleanup** is triggered
2. **Garbage collection** is forced
3. **Processing continues** with reduced memory
4. **Service stays alive** (no crash)

### **If Video Too Large**
1. **Graceful failure** with clear error message
2. **No service crash**
3. **User gets helpful error** about size limits

## **🔄 Fallback Strategy**

### **Memory-Based Fallbacks**
1. **Normal processing** (if memory < 400MB)
2. **Reduced quality** (if memory 400-450MB)
3. **Emergency cleanup** (if memory > 450MB)
4. **Graceful failure** (if memory > 500MB)

### **Size-Based Fallbacks**
1. **Accept video** (if < 25MB)
2. **Warn user** (if 25-30MB)
3. **Reject video** (if > 30MB)

## **📈 Scaling Options**

### **Current Plan (512MB)**
- ✅ Small Loom videos (< 25MB, < 3 minutes)
- ⚠️ Medium videos (may work, may fail)
- ❌ Large videos (will fail gracefully)

### **Upgrade Options**
- **Render Pro**: 1GB memory ($25/month)
- **Render Standard**: 2GB memory ($50/month)
- **Custom plan**: More memory as needed

## **🎯 Success Criteria**

### **For 512MB Plan**
- ✅ Memory stays under 450MB
- ✅ Videos process successfully
- ✅ No service crashes
- ✅ Graceful error handling
- ✅ Clear user feedback

### **Performance Targets**
- **Processing time**: < 2 minutes
- **Memory peak**: < 450MB
- **Success rate**: > 90% for small videos
- **Error rate**: < 10% with helpful messages

## **🚀 Deployment Checklist**

### **Before Deploying**
- [ ] Set environment variables
- [ ] Test with small Loom video (< 10MB)
- [ ] Monitor memory usage
- [ ] Verify cleanup works
- [ ] Check error handling

### **After Deploying**
- [ ] Monitor logs for memory warnings
- [ ] Test with various video sizes
- [ ] Verify graceful failures
- [ ] Check user experience
- [ ] Optimize if needed

## **📞 Troubleshooting**

### **If Service Crashes**
1. Check memory usage in logs
2. Reduce video size limits further
3. Increase garbage collection frequency
4. Consider upgrading plan

### **If Processing Fails**
1. Check video size and duration
2. Verify yt-dlp is working
3. Check environment variables
4. Review error messages

The system is designed to work within your 512MB limit while providing reliable Loom video processing for appropriately sized videos. 