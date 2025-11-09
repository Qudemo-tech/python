"""
Cut specific time segments from Loom video
Video: https://www.loom.com/share/0b0d966f3cdd4d609b983e46d0aec403
Segments:
1. 1:59 to 3:40 - "How to create qudemo"
2. 4:30 to 5:45 - "View interactions"
"""

import os
import subprocess
import requests
from datetime import datetime

# Configuration
LOOM_VIDEO_URL = "https://www.loom.com/share/0b0d966f3cdd4d609b983e46d0aec403"
LOOM_VIDEO_ID = "0b0d966f3cdd4d609b983e46d0aec403"
OUTPUT_DIR = "loom_qudemo_segments"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video file path
VIDEO_FILE = os.path.join(OUTPUT_DIR, "loom_video.mp4")

# Segments to cut
SEGMENTS = [
    {
        "name": "create_qudemo",
        "start": "00:01:59",
        "end": "00:03:40",
        "duration": "00:01:41",  # 3:40 - 1:59 = 1:41
        "description": "How to create a Qudemo"
    },
    {
        "name": "view_interactions",
        "start": "00:04:30",
        "end": "00:05:45",
        "duration": "00:01:15",  # 5:45 - 4:30 = 1:15
        "description": "View interactions feature"
    }
]

print("=" * 80)
print("🎬 LOOM VIDEO SEGMENT CUTTER")
print("=" * 80)
print(f"Video URL: {LOOM_VIDEO_URL}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)
print()

def download_loom_video():
    """Download Loom video using yt-dlp"""
    print("📥 Downloading Loom video...")
    
    try:
        # Try using yt-dlp
        print("🔧 Using yt-dlp to download...")
        cmd = [
            'yt-dlp',
            '-f', 'best',
            '-o', VIDEO_FILE,
            LOOM_VIDEO_URL
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(VIDEO_FILE):
            print(f"✅ Video downloaded to: {VIDEO_FILE}")
            
            # Get file size
            size_mb = os.path.getsize(VIDEO_FILE) / (1024 * 1024)
            print(f"📊 File size: {size_mb:.2f} MB")
            return True
        
        print(f"⚠️ yt-dlp failed: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        return False


def cut_video_segment(segment):
    """Cut a specific segment from the video using FFmpeg"""
    print(f"\n✂️  Cutting segment: {segment['name']}")
    print(f"   Description: {segment['description']}")
    print(f"   Time: {segment['start']} → {segment['end']} (duration: {segment['duration']})")
    
    output_file = os.path.join(OUTPUT_DIR, f"{segment['name']}.mp4")
    
    try:
        # FFmpeg command to cut segment
        cmd = [
            'ffmpeg',
            '-i', VIDEO_FILE,
            '-ss', segment['start'],  # Start time
            '-to', segment['end'],    # End time
            '-c', 'copy',             # Copy codec (fast, no re-encoding)
            '-y',                     # Overwrite output file
            output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"   ✅ Saved: {output_file}")
            print(f"   📊 Size: {size_mb:.2f} MB")
            return True
        else:
            print(f"   ❌ FFmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("   ❌ FFmpeg not found. Please install ffmpeg.")
        return False
    except Exception as e:
        print(f"   ❌ Error cutting segment: {e}")
        return False


def main():
    """Main execution"""
    
    # Step 1: Download video (if not already downloaded)
    if not os.path.exists(VIDEO_FILE):
        if not download_loom_video():
            print("\n❌ Failed to download video. Exiting.")
            return
    else:
        print(f"✅ Video already exists: {VIDEO_FILE}")
        size_mb = os.path.getsize(VIDEO_FILE) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
    
    # Step 2: Cut segments
    print("\n" + "=" * 80)
    print("✂️  CUTTING VIDEO SEGMENTS")
    print("=" * 80)
    
    success_count = 0
    for segment in SEGMENTS:
        if cut_video_segment(segment):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ SEGMENT CUTTING COMPLETE!")
    print("=" * 80)
    print(f"Successfully cut: {success_count}/{len(SEGMENTS)} segments")
    print()
    print("📁 Output files:")
    for segment in SEGMENTS:
        output_file = os.path.join(OUTPUT_DIR, f"{segment['name']}.mp4")
        if os.path.exists(output_file):
            print(f"   ✅ {output_file}")
        else:
            print(f"   ❌ {output_file} (failed)")
    print("=" * 80)


if __name__ == "__main__":
    main()

