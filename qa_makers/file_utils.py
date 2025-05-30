"""
File processing utilities module, including file read/write and path handling functions
"""

import os
import glob
import re
import json
import logging
import time
from typing import List, Dict, Any, Tuple, Optional

from qa_makers.config import DATA_DIR

logger = logging.getLogger(__name__)

def get_latest_modified_topics_file():
    """Gets the latest modified_topics file"""
    files = glob.glob(os.path.join(DATA_DIR, "modified_topics_*.json"))
    if not files:
        return None  # No matching file found
    # Sort by file modification time, return the latest file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def get_timestamp_from_filename(filename):
    """Extracts timestamp from filename"""
    match = re.search(r'modified_topics_(\d+)\.json', filename)
    if match:
        return match.group(1)
    return None

def determine_io_files(args):
    """Determines input and output file paths"""
    input_file = None
    timestamp = None
    
    if args.timestamp:
        # Use timestamp specified in command line
        timestamp = args.timestamp
        specified_file = os.path.join(DATA_DIR, f"modified_topics_{timestamp}.json")
        if os.path.exists(specified_file):
            input_file = specified_file
        else:
            logger.error(f"Error: File with timestamp {timestamp} not found: {specified_file}")
            return None, None, None
    elif args.input:
        # Use input file specified in command line
        input_file = args.input
        # Try to extract timestamp from input filename
        timestamp = get_timestamp_from_filename(input_file)
    else:
        # Use the latest file
        input_file = get_latest_modified_topics_file()
        if input_file:
            timestamp = get_timestamp_from_filename(input_file)
        else:
            logger.error("Error: No modified_topics file found")
            return None, None, None
    
    # Set output file, keep the same timestamp
    if args.output:
        # If output file is specified, prioritize it
        output_file = args.output
    elif timestamp:
        # Use the same timestamp
        output_file = os.path.join(DATA_DIR, f"l1_topics_{timestamp}.json")
    else:
        # Use default output filename
        output_file = os.path.join(DATA_DIR, "l1_topics.json")
    
    return input_file, output_file, timestamp

def load_topics(file_path):
    """Safely loads a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading topics: {str(e)}")
        return []

def save_results(output_file, output_data, last_save_time):
    """Thread-safely saves results to the output file"""
    # Reduce save frequency to avoid too frequent I/O
    current_time = time.time()
    if current_time - last_save_time >= 1:  # Save at least once per second
        try:
            # Create output directory (if it doesn't exist)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Real-time saving results to {output_file}")
            return current_time
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    return last_save_time
