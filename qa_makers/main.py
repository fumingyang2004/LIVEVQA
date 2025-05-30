"""
Level 1 Question Generator Main Entry File

This script generates level 1 questions that require social knowledge to answer.
"""

import os
import sys
import logging
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root path to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import component modules
from qa_makers.config import CONFIG, LOG_DIR
from qa_makers.file_utils import determine_io_files, load_topics, save_results
from qa_makers.question_processor import process_topic_thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "l1_question_generator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Synchronization locks
output_lock = threading.Lock()
save_lock = threading.Lock()
last_save_time = time.time()

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Level 1 Multi-Hop Questions')
    parser.add_argument('--workers', type=int, default=CONFIG["max_workers"], 
                        help='Number of threads for parallel processing')
    parser.add_argument('--input', type=str, 
                        help='Input file path (overrides automatic selection of the latest file)')
    parser.add_argument('--output', type=str,
                        help='Output file path (overrides automatically generated path)')
    parser.add_argument('--continue', dest='continue_processing', action='store_true',
                        help='Continue processing an existing output file')
    parser.add_argument('--timestamp', '-t', type=str, 
                        help='Specify timestamp of the modified_topics file to process, e.g., 04181718')
    args = parser.parse_args()
    
    # Determine input and output files
    input_file, output_file, timestamp = determine_io_files(args)
    if not input_file or not output_file:
        logger.error("Could not determine input or output file")
        return 1
    
    # Update configuration
    CONFIG["max_workers"] = args.workers
    CONFIG["input_file"] = input_file
    CONFIG["output_file"] = output_file
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    if timestamp:
        logger.info(f"Timestamp: {timestamp}")
    
    # Load input topics
    input_topics = load_topics(CONFIG["input_file"])
    if not input_topics:
        logger.error(f"Could not load input file or file is empty: {CONFIG['input_file']}")
        return 1
    
    logger.info(f"Loaded {len(input_topics)} topics from {CONFIG['input_file']}")
    
    # Initialize output data and set of processed IDs
    output_data = []
    processed_ids = set()
    
    # Check if output file exists and load processed IDs
    if os.path.exists(CONFIG["output_file"]):
        existing_output = load_topics(CONFIG["output_file"])
        if existing_output:
            # If --continue argument is used, use existing output as base
            if args.continue_processing:
                output_data = existing_output
                logger.info(f"Loaded {len(output_data)} processed topics from {CONFIG['output_file']}")
            
            # Regardless of --continue, extract processed IDs for deduplication
            processed_ids = {topic.get('id') for topic in existing_output if topic.get('id') is not None}
            logger.info(f"Found {len(processed_ids)} processed IDs, these will be skipped")
    
    # Filter out already processed topics
    topics_to_process = []
    for topic in input_topics:
        topic_id = topic.get('id')
        if topic_id is None:
            # Topics without an ID also need to be processed
            topics_to_process.append(topic)
        elif topic_id not in processed_ids:
            # Only process unprocessed IDs
            topics_to_process.append(topic)
        else:
            logger.info(f"Skipping already processed topic ID: {topic_id}")
    
    logger.info(f"Filtered, {len(topics_to_process)} new topics to process, using {CONFIG['max_workers']} threads")
    
    # If no new topics to process, return directly
    if not topics_to_process:
        logger.info("No new topics to process, program finished")
        return 0
    
    # Use thread pool to process topics in parallel
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        # Submit all tasks
        future_to_topic = {
            executor.submit(process_topic_thread, topic, output_data, CONFIG, last_save_time): topic 
            for topic in topics_to_process
        }
        
        # Process completed tasks
        with tqdm(total=len(topics_to_process), desc="Processing topics") as pbar:
            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    success = future.result()
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Worker thread failed: {str(e)}")
    
    # Ensure final save
    try:
        os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
        with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
            import json
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"All processing completed and saved to {CONFIG['output_file']}")
    except Exception as e:
        logger.error(f"Error saving final results: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
