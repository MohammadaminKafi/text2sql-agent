"""
Logs router for streaming live logs to the frontend.
"""

import os
import glob
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])


class LogEntry(BaseModel):
    """Log entry model"""
    timestamp: str
    level: str
    module: str
    message: str
    thread_id: str


class LogsResponse(BaseModel):
    """Response model for logs"""
    logs: List[LogEntry]
    total_entries: int
    has_more: bool


@router.get("/recent", response_model=LogsResponse)
async def get_recent_logs(limit: int = 50, thread_id: Optional[str] = None):
    """
    Get recent log entries from all.log files
    
    Args:
        limit: Maximum number of log entries to return
        thread_id: Optional thread ID to filter logs (e.g., system-run-abc123)
    """
    try:
        log_base_dir = os.getenv('LOG_BASE_DIR', './log')
        
        # Safety check
        if not os.path.exists(log_base_dir):
            return LogsResponse(logs=[], total_entries=0, has_more=False)
        
        # Find all log directories
        if thread_id:
            # Filter by specific thread ID
            log_dirs = glob.glob(os.path.join(log_base_dir, f"*{thread_id}*"))
        else:
            # Get all system-run logs, sorted by modification time (newest first)
            try:
                log_dirs = glob.glob(os.path.join(log_base_dir, "system-run-*"))
                if log_dirs:
                    log_dirs.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
            except Exception as e:
                logger.warning(f"Error sorting log directories: {e}")
                log_dirs = []
        
        all_logs = []
        
        # Read logs from each directory
        for log_dir in log_dirs[:5]:  # Limit to 5 most recent directories for performance
            if not os.path.exists(log_dir):
                continue
                
            all_log_path = os.path.join(log_dir, "all.log")
            if os.path.exists(all_log_path):
                thread_name = os.path.basename(log_dir)
                
                try:
                    # Read only recent lines to avoid memory issues
                    with open(all_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()[-50:]  # Only read last 50 lines per file
                        
                    for line in lines:
                        line = line.strip()
                        if line and len(all_logs) < limit * 2:  # Prevent excessive processing
                            parsed_entry = parse_log_line(line, thread_name)
                            if parsed_entry:
                                all_logs.append(parsed_entry)
                                
                except Exception as e:
                    logger.warning(f"Error reading log file {all_log_path}: {e}")
                    continue
        
        # Sort by timestamp (newest first) and limit
        try:
            # Since timestamp is a string in HH:MM:SS format, we can sort lexicographically for same-day logs
            all_logs.sort(key=lambda x: x.timestamp, reverse=True)
        except Exception as e:
            logger.warning(f"Error sorting logs: {e}")
            
        limited_logs = all_logs[:limit]
        
        return LogsResponse(
            logs=limited_logs,
            total_entries=len(all_logs),
            has_more=len(all_logs) > limit
        )
        
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        # Return empty response instead of error to prevent breaking the UI
        return LogsResponse(logs=[], total_entries=0, has_more=False)


@router.get("/threads")
async def get_active_threads():
    """Get list of active log threads"""
    try:
        log_base_dir = os.getenv('LOG_BASE_DIR', './log')
        
        # Find all system-run log directories
        log_dirs = glob.glob(os.path.join(log_base_dir, "system-run-*"))
        log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        threads = []
        for log_dir in log_dirs[:20]:  # Show last 20 threads
            thread_name = os.path.basename(log_dir)
            all_log_path = os.path.join(log_dir, "all.log")
            
            if os.path.exists(all_log_path):
                # Get basic info about the thread
                stat = os.stat(all_log_path)
                threads.append({
                    "thread_id": thread_name,
                    "last_modified": stat.st_mtime,
                    "size_bytes": stat.st_size
                })
        
        return {"threads": threads}
        
    except Exception as e:
        logger.error(f"Error fetching threads: {e}")
        raise HTTPException(status_code=500, detail="Error fetching threads")


@router.get("/flow/{thread_id}")
async def get_flow_logs(thread_id: str):
    """Get flow-specific logs for a thread"""
    try:
        log_base_dir = os.getenv('LOG_BASE_DIR', './log')
        log_dir = os.path.join(log_base_dir, thread_id)
        
        if not os.path.exists(log_dir):
            raise HTTPException(status_code=404, detail="Thread not found")
        
        logs = {}
        
        # Read different log files
        for log_type in ['all.log', 'flow.log', 'flow_time.log', 'flow_tokens.log']:
            log_path = os.path.join(log_dir, log_type)
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        logs[log_type.replace('.log', '')] = content.split('\n') if content else []
                except Exception as e:
                    logger.warning(f"Error reading {log_path}: {e}")
                    logs[log_type.replace('.log', '')] = []
            else:
                logs[log_type.replace('.log', '')] = []
        
        return {
            "thread_id": thread_id,
            "logs": logs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching flow logs for {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching flow logs")


def parse_log_line(line: str, thread_id: str) -> Optional[LogEntry]:
    """Parse a log line into structured format"""
    try:
        # Format: "13:30:37  [SYSTEM_INFO]  core.smartlog.core › Activated log thread: ..."
        parts = line.split("  ", 2)
        if len(parts) < 3:
            return None
            
        timestamp = parts[0].strip()
        level_part = parts[1].strip()
        message_part = parts[2].strip()
        
        # Extract level (remove brackets)
        if level_part.startswith('[') and level_part.endswith(']'):
            level = level_part[1:-1]
        else:
            level = "UNKNOWN"
        
        # Split module and message
        if ' › ' in message_part:
            module, message = message_part.split(' › ', 1)
        else:
            module = "unknown"
            message = message_part
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            module=module.strip(),
            message=message.strip(),
            thread_id=thread_id
        )
        
    except Exception as e:
        logger.warning(f"Error parsing log line '{line}': {e}")
        return None