"""
Utility functions for the trading system.
Includes atomic file operations to prevent race conditions.
"""
import json
import os
import tempfile
import shutil
import logging
import time
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

def save_json_atomic(filepath: str, data: Any, indent: int = 2):
    """
    Save JSON data atomically to avoid partial writes or race conditions.
    Writes to a temp file first, then renames it to the target file.
    """
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    # Create temp file in the same directory to ensure atomic rename works
    # (rename across filesystems might not be atomic)
    fd, temp_path = tempfile.mkstemp(dir=dir_name, text=True)
    
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
            
        # Atomic rename
        shutil.move(temp_path, filepath)
        
    except Exception as e:
        logger.error(f"Error saving JSON atomically to {filepath}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e

def load_json_safe(filepath: str, default: Any = None, retries: int = 3, delay: float = 0.1) -> Any:
    """
    Load JSON data safely with retries.
    """
    if default is None:
        default = {}
        
    if not os.path.exists(filepath):
        return default
        
    for i in range(retries):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            if i == retries - 1:
                logger.error(f"JSON decode error in {filepath}")
                return default
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return default
            
    return default
