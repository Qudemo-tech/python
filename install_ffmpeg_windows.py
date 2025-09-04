#!/usr/bin/env python3
"""
FFmpeg Installation Script for Windows
Automatically downloads and installs FFmpeg for Whisper compatibility
"""

import os
import sys
import subprocess
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path

def download_ffmpeg():
    """Download FFmpeg for Windows"""
    try:
        print("🔽 Downloading FFmpeg for Windows...")
        
        # FFmpeg download URL (latest release)
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        
        # Download to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📥 Downloading: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ Downloaded FFmpeg: {downloaded / (1024*1024):.1f} MB")
            return tmp_file.name
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def install_ffmpeg():
    """Install FFmpeg to system"""
    try:
        print("🔧 Installing FFmpeg...")
        
        # Download FFmpeg
        zip_path = download_ffmpeg()
        if not zip_path:
            return False
        
        # Extract to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print("📦 Extracting FFmpeg...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the extracted folder
            extracted_folders = [f for f in os.listdir(temp_dir) if f.startswith('ffmpeg')]
            if not extracted_folders:
                print("❌ Could not find extracted FFmpeg folder")
                return False
            
            ffmpeg_folder = os.path.join(temp_dir, extracted_folders[0])
            ffmpeg_bin = os.path.join(ffmpeg_folder, 'bin')
            
            # Install to user directory
            user_home = Path.home()
            install_dir = user_home / 'ffmpeg'
            install_bin = install_dir / 'bin'
            
            print(f"📁 Installing to: {install_dir}")
            
            # Create installation directory
            install_dir.mkdir(exist_ok=True)
            install_bin.mkdir(exist_ok=True)
            
            # Copy FFmpeg files
            for file in os.listdir(ffmpeg_bin):
                src = os.path.join(ffmpeg_bin, file)
                dst = os.path.join(install_bin, file)
                shutil.copy2(src, dst)
                print(f"📄 Copied: {file}")
            
            # Add to PATH (Windows)
            ffmpeg_path = str(install_bin)
            current_path = os.environ.get('PATH', '')
            
            if ffmpeg_path not in current_path:
                print("🔧 Adding FFmpeg to PATH...")
                
                # Add to user PATH
                import winreg
                
                # Open user environment variables
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                   r"Environment", 
                                   0, 
                                   winreg.KEY_ALL_ACCESS)
                
                try:
                    # Get current PATH
                    current_path, _ = winreg.QueryValueEx(key, "PATH")
                except FileNotFoundError:
                    current_path = ""
                
                # Add FFmpeg to PATH
                if ffmpeg_path not in current_path:
                    new_path = f"{current_path};{ffmpeg_path}" if current_path else ffmpeg_path
                    winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                    print("✅ Added FFmpeg to PATH")
                
                winreg.CloseKey(key)
            
            # Test installation
            ffmpeg_exe = os.path.join(install_bin, 'ffmpeg.exe')
            if os.path.exists(ffmpeg_exe):
                print("🧪 Testing FFmpeg installation...")
                result = subprocess.run([ffmpeg_exe, '-version'], 
                                      capture_output=True, 
                                      timeout=10)
                if result.returncode == 0:
                    print("✅ FFmpeg installed successfully!")
                    print(f"📍 Location: {ffmpeg_exe}")
                    return True
                else:
                    print("❌ FFmpeg test failed")
                    return False
            else:
                print("❌ FFmpeg executable not found")
                return False
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False
    
    finally:
        # Clean up download
        if 'zip_path' in locals() and os.path.exists(zip_path):
            try:
                os.unlink(zip_path)
            except:
                pass

def check_ffmpeg():
    """Check if FFmpeg is already installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is already installed and working")
            return True
        else:
            print("⚠️ FFmpeg found but not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
        return False
    except Exception as e:
        print(f"⚠️ FFmpeg check failed: {e}")
        return False

def main():
    """Main installation function"""
    print("🎬 FFmpeg Installation for Windows")
    print("=" * 40)
    
    # Check if already installed
    if check_ffmpeg():
        print("🎉 FFmpeg is ready to use!")
        return True
    
    # Install FFmpeg
    if install_ffmpeg():
        print("\n🎉 FFmpeg installation completed!")
        print("🔄 Please restart your terminal/IDE for PATH changes to take effect")
        print("🧪 You can test with: ffmpeg -version")
        return True
    else:
        print("\n❌ FFmpeg installation failed")
        print("💡 You can manually install FFmpeg from: https://ffmpeg.org/download.html")
        return False

if __name__ == "__main__":
    main()
