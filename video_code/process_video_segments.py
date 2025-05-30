'''
Read image paths and topic content
Generate JSON program
Next step will be generating Q&A
'''
import os
import json
import glob
import time
import random
import datetime
from pathlib import Path
import requests
import logging
import traceback

# Generate timestamp
timestamp = datetime.datetime.now().strftime("%m%d%H%M")

# Create logs directory
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(logs_dir, f"process_segments_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_segments(segments_dir):
    """
    Process all subdirectories in segments4 directory to generate JSON in specified format
    """
    all_segments = []
    
    # Get all subdirectories
    try:
        subdirs = [d for d in os.listdir(segments_dir) if os.path.isdir(os.path.join(segments_dir, d))]
        logger.info(f"Found {len(subdirs)} subdirectories in {segments_dir}")
    except Exception as e:
        logger.error(f"Error reading segments directory {segments_dir}: {e}")
        return all_segments
    
    for i, subdir in enumerate(subdirs):
        subdir_path = os.path.join(segments_dir, subdir)
        logger.info(f"Processing [{i+1}/{len(subdirs)}] directory: {subdir}")
        
        # Find metadata.json file
        metadata_files = glob.glob(os.path.join(subdir_path, "*final_metadata.json"))
        if not metadata_files:
            logger.warning(f"No metadata.json found in {subdir_path}")
            continue
            
        metadata_file = metadata_files[0]
        logger.info(f"Using metadata file: {os.path.basename(metadata_file)}")
        
        try:
            # Read metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Find images in selected_tag1 directory
            selected_dir = os.path.join(subdir_path, "selected0")
            if not os.path.exists(selected_dir):
                logger.warning(f"No selected0 directory found in {subdir_path}")

                
            image_paths = glob.glob(os.path.join(selected_dir, "*.jpg")) + glob.glob(os.path.join(selected_dir, "*.png"))
            
            if not image_paths:
                logger.warning(f"No images found in {selected_dir}")
                continue
            
            logger.info(f"Found {len(image_paths)} images in {selected_dir}")
            
            # Default caption
            default_caption = "A visual representation related to the content."
            
            # Get content, ensure it's not empty
            content = metadata.get('content', '')
            if not content:
                logger.warning(f"Content is empty in {metadata_file}")
                continue
                
            # Generate topic
            topic = metadata.get('topic', '')
            
            for img_path in image_paths:
                # Construct absolute path to ensure images can be located
                img_fullpath = os.path.abspath(img_path)
                
                # Generate unique ID
                current_timestamp = int(time.time() * 1000)
                time.sleep(0.001)  # Ensure timestamp uniqueness
                unique_id = f"1_{current_timestamp}"
                
                # Build output format - one entry per image
                segment_data = {
                    "id": unique_id,
                    "topic": topic,
                    "text": content,
                    "img_urls": [],
                    "img_paths": [img_fullpath],
                    "captions": 'null',
                    "source": "YouTube"
                }
                
                all_segments.append(segment_data)
            
            logger.info(f"Processed {subdir} successfully")
            
        except Exception as e:
            logger.error(f"Error processing {subdir}: {e}")
            logger.error(traceback.format_exc())
    
    logger.info(f"Total processed segments: {len(all_segments)}")
    return all_segments

def save_json(data, output_file):
    """
    Save JSON data to file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving data to {output_file}: {e}")
        logger.error(traceback.format_exc())

def main():
    """
    Main function
    """
    import argparse
    
    # Create command line parser
    parser = argparse.ArgumentParser(description="Process video segments and generate JSON data")
    parser.add_argument('--segments_dir', type=str, default="",
                        help="Path to video segments directory")
    
    # Parse command line arguments
    args = parser.parse_args()
    segments_dir = args.segments_dir
    
    # Set output path
    output_dir = ""
    output_file = os.path.join(output_dir, f"modified_topics_{timestamp}.json")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(segments_dir):
        logger.error(f"Directory {segments_dir} does not exist!")
        return
        
    logger.info(f"Using segments directory: {segments_dir}")
    
    # Process segments
    logger.info(f"Starting to process segments in {segments_dir}")
    segments_data = process_segments(segments_dir)
    
    # Save results
    if segments_data:
        save_json(segments_data, output_file)
        logger.info(f"Successfully processed {len(segments_data)} segments and saved to {output_file}")
    else:
        logger.warning("No valid segments were processed")

if __name__ == "__main__":
    main()