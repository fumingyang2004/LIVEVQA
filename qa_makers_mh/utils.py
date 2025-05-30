"""
Utility functions for the multi-hop question generator
"""

import os
import re
import json
import glob
import logging
import traceback
from qa_makers_mh.config import DATA_DIR

logger = logging.getLogger(__name__)

def get_latest_filtered_topics_file():
    """Gets the latest l1_filtered_topics file."""
    files = glob.glob(os.path.join(DATA_DIR, "l1_filtered_topics_*.json"))
    if not files:
        return None  # No matching files found
    # Sort by file modification time and return the latest file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def get_timestamp_from_filename(filename):
    """Extracts the timestamp from a filename."""
    match = re.search(r'l1_filtered_topics_(\d+)\.json', filename)
    if match:
        return match.group(1)
    return None

def determine_io_files(args):
    """Determines input and output file paths."""
    input_file = None
    timestamp = None
    
    if args.timestamp:
        # Use timestamp specified via command line
        timestamp = args.timestamp
        specified_file = os.path.join(DATA_DIR, f"l1_filtered_topics_{timestamp}.json")
        if os.path.exists(specified_file):
            input_file = specified_file
        else:
            logger.error(f"Error: File with timestamp {timestamp} not found at {specified_file}")
            return None, None, None
    elif args.input:
        # Use input file specified via command line
        input_file = args.input
        # Try to extract timestamp from input filename
        timestamp = get_timestamp_from_filename(input_file)
    else:
        # Use the latest file
        input_file = get_latest_filtered_topics_file()
        if input_file:
            timestamp = get_timestamp_from_filename(input_file)
        else:
            logger.error("Error: No l1_filtered_topics file found.")
            return None, None, None
    
    # Set output file, maintaining the same timestamp
    if args.output:
        # If output file is specified, prioritize it
        output_file = args.output
    elif timestamp:
        # Use the same timestamp
        output_file = os.path.join(DATA_DIR, f"l23_topics_{timestamp}.json")
    else:
        # Use a default output filename
        output_file = os.path.join(DATA_DIR, "l23_topics.json")
    
    return input_file, output_file, timestamp

def load_input_file(input_file):
    """
    Loads the input file.
    
    Args:
        input_file: Path to the input file.
        
    Returns:
        list: A list of topics.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        logger.info(f"Loaded {len(topics)} topics from {input_file}")
        return topics
    except Exception as e:
        logger.error(f"Error loading input file: {str(e)}")
        return []

def save_results(output_file, results):
    """
    Saves results to a file.
    
    Args:
        output_file: Path to the output file.
        results: The list of results.
        
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return False

def encode_image_to_base64(image_path):
    """Encodes an image to a base64 string."""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image file does not exist: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            import base64
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None