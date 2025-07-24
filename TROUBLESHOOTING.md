# Deployment Troubleshooting Guide

## **Current Issues & Solutions**

### **Issue 1: Python Version Mismatch**
**Problem**: Render is using Python 3.13 despite runtime.txt specifying 3.11
**Solution**: 
- Updated runtime.txt to use Python 3.11.9
- Added .python-version file
- Using minimal requirements to avoid compatibility issues

### **Issue 2: Package Compatibility**
**Problem**: Several packages don't support Python 3.13
**Solution**:
- Updated all packages to use `>=` instead of `==`
- Removed problematic packages temporarily
- Created minimal requirements file

### **Issue 3: Pinecone Version**
**Problem**: pinecone-client==3.0.0 not available for Python 3.13
**Solution**:
- Updated to pinecone-client>=4.0.0
- Updated API usage to be compatible

## **Deployment Strategy**

### **Phase 1: Minimal Deployment (Current)**
```bash
# Use minimal requirements
pip install -r requirements-minimal.txt
```

**What's included:**
- ✅ Core FastAPI dependencies
- ✅ Video processing (yt-dlp, whisper)
- ✅ Basic data science (numpy, pandas)
- ✅ Vector DBs (supabase, pinecone)
- ✅ System monitoring (psutil)

**What's excluded:**
- ❌ scikit-learn (can add later)
- ❌ faiss-cpu (can add later)
- ❌ google-cloud-storage (can add later)

### **Phase 2: Full Features (After Basic Deployment)**
```bash
# Add back full requirements
pip install -r requirements.txt
```

## **Testing Steps**

### **Step 1: Deploy Minimal Version**
1. Push current changes
2. Monitor deployment logs
3. Check if basic deployment succeeds

### **Step 2: Test Core Functionality**
1. Test health endpoint: `GET /health`
2. Test basic video processing
3. Verify memory usage

### **Step 3: Add Features Back**
1. Add scikit-learn if needed
2. Add faiss-cpu if needed
3. Add google-cloud-storage if needed

## **Alternative Solutions**

### **Option A: Use Docker**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Run application
CMD ["python", "run.py"]
```

### **Option B: Use Poetry**
```toml
# pyproject.toml
[tool.poetry]
name = "qudemo-python"
version = "0.1.0"
description = "QuDemo Python Backend"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
# ... other dependencies
```

### **Option C: Use Conda**
```yaml
# environment.yml
name: qudemo
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - fastapi>=0.104.0
    - uvicorn>=0.24.0
    # ... other pip packages
```

## **Monitoring Deployment**

### **Key Log Messages to Watch:**
```
✅ Python 3.11.9 detected
✅ All packages installed successfully
✅ FastAPI server starting
✅ Health check endpoint responding
```

### **Error Messages to Monitor:**
```
❌ Python version mismatch
❌ Package not found
❌ Import error
❌ Memory limit exceeded
```

## **Fallback Plan**

If deployment continues to fail:

1. **Use Render's built-in Python environment**
   - Remove runtime.txt
   - Let Render choose Python version
   - Update requirements for that version

2. **Use external video processing**
   - Process videos on a separate service
   - Use cloud transcription services
   - Reduce local processing requirements

3. **Use different hosting platform**
   - Heroku (supports Python 3.11)
   - Railway (good Python support)
   - DigitalOcean App Platform

## **Success Criteria**

Deployment is successful when:
- ✅ No build errors
- ✅ Health endpoint responds
- ✅ Basic video processing works
- ✅ Memory usage stays under 400MB
- ✅ No import errors in logs

## **Next Steps After Successful Deployment**

1. **Test video processing** with small files
2. **Monitor memory usage** during processing
3. **Add missing packages** one by one
4. **Test large video processing** if needed
5. **Optimize performance** based on usage

The minimal deployment should work and provide a foundation for adding more features gradually. 