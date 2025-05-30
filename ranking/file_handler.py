"""
File handler module for reading and writing JSON files
"""

import os
import json
import shutil
import tempfile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_topics(file_path):
    """Safely read JSON file with error handling"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return []

def save_topics(file_path, data):
    """Safely write data to JSON file using atomic write pattern"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tmp:
            # Write data to temp file
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp_name = tmp.name
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Safely replace original file with temp file
        shutil.move(tmp_name, file_path)
        return True
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        # Clean up temp file if it still exists
        if 'tmp_name' in locals() and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except:
                pass
        return False

def create_processed_topics_index(processed_topics):
    """Create an index of processed topics by their IDs for quick lookup"""
    processed_ids = {}
    for topic in processed_topics:
        if 'id' in topic:
            processed_ids[topic['id']] = topic
    return processed_ids