#!/usr/bin/env python3
"""
Progress Tracker - Real-time scraping progress updates
"""

import time
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self, task_id: str, total_estimated: int = 0):
        self.task_id = task_id
        self.start_time = time.time()
        self.total_estimated = total_estimated
        self.current_step = 0
        self.total_steps = 0
        self.current_url = ""
        self.status = "initializing"
        self.messages = []
        self.errors = []
        self.warnings = []
        self.stats = {
            "urls_scraped": 0,
            "urls_skipped": 0,
            "urls_failed": 0,
            "articles_found": 0,
            "collections_processed": 0,
            "current_collection": "",
            "current_article": ""
        }
        
    def update_status(self, status: str, message: str = ""):
        """Update current status"""
        self.status = status
        if message:
            self.messages.append({
                "timestamp": datetime.now().isoformat(),
                "message": message
            })
        logger.info(f"📊 Progress [{self.task_id}]: {status} - {message}")
    
    def update_progress(self, current: int, total: int, url: str = ""):
        """Update progress counters"""
        self.current_step = current
        self.total_steps = total
        if url:
            self.current_url = url
        
        if total > 0:
            percentage = (current / total) * 100
            elapsed = time.time() - self.start_time
            eta = (elapsed / current) * (total - current) if current > 0 else 0
            
            logger.info(f"📊 Progress [{self.task_id}]: {current}/{total} ({percentage:.1f}%) - ETA: {eta/60:.1f}min")
    
    def update_stats(self, **kwargs):
        """Update statistics"""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value
    
    def add_error(self, error: str):
        """Add error message"""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error
        })
        logger.error(f"❌ Error [{self.task_id}]: {error}")
    
    def add_warning(self, warning: str):
        """Add warning message"""
        self.warnings.append({
            "timestamp": datetime.now().isoformat(),
            "warning": warning
        })
        logger.warning(f"⚠️ Warning [{self.task_id}]: {warning}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        elapsed = time.time() - self.start_time
        
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": {
                "current": self.current_step,
                "total": self.total_steps,
                "percentage": (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
                "elapsed_minutes": elapsed / 60,
                "eta_minutes": self._calculate_eta()
            },
            "current_url": self.current_url,
            "stats": self.stats,
            "messages": self.messages[-10:],  # Last 10 messages
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_eta(self) -> float:
        """Calculate estimated time to completion"""
        if self.current_step == 0 or self.total_steps == 0:
            return 0
        
        elapsed = time.time() - self.start_time
        rate = self.current_step / elapsed
        remaining = self.total_steps - self.current_step
        
        return remaining / rate / 60 if rate > 0 else 0  # Return in minutes
    
    def is_complete(self) -> bool:
        """Check if task is complete"""
        return self.status in ["completed", "failed", "timeout"]
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time

class ProgressManager:
    """Manages multiple progress trackers"""
    
    def __init__(self):
        self.trackers = {}
    
    def create_tracker(self, task_id: str, total_estimated: int = 0) -> ProgressTracker:
        """Create a new progress tracker"""
        tracker = ProgressTracker(task_id, total_estimated)
        self.trackers[task_id] = tracker
        return tracker
    
    def get_tracker(self, task_id: str) -> Optional[ProgressTracker]:
        """Get existing tracker"""
        return self.trackers.get(task_id)
    
    def remove_tracker(self, task_id: str):
        """Remove completed tracker"""
        if task_id in self.trackers:
            del self.trackers[task_id]
    
    def get_all_progress(self) -> Dict[str, Any]:
        """Get progress for all active trackers"""
        return {
            task_id: tracker.get_progress_summary()
            for task_id, tracker in self.trackers.items()
        }
    
    def cleanup_old_trackers(self, max_age_hours: int = 24):
        """Remove old completed trackers"""
        current_time = time.time()
        to_remove = []
        
        for task_id, tracker in self.trackers.items():
            if tracker.is_complete() and tracker.get_elapsed_time() > (max_age_hours * 3600):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            self.remove_tracker(task_id)

# Global progress manager instance
progress_manager = ProgressManager()
