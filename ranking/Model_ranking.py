"""
Image Processing and Enhancement Tool for News Articles
Main entry point that coordinates all ranking modules
"""

import os
import sys
import logging
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

# Add module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking.config import CONFIG
from ranking.client import setup_client, get_thread_client
from ranking.file_handler import load_topics, save_topics, create_processed_topics_index
from ranking.image_processor import process_topic_images
from ranking.topic_manager import is_discarded_topic, process_and_update_realtime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add synchronization locks
output_lock = threading.Lock()
save_lock = threading.Lock()
completed_count = 0
last_save_time = time.time()

# python script.py --timestamp 04181718
def process_topic_thread(topic, output_data, processed_ids):
    """Thread function: Processes a single topic
    
    Args:
        topic: The topic to process
        output_data: Shared list of output data
        processed_ids: Index of processed topic IDs
        
    Returns:
        True if processed successfully, False otherwise
    """
    global completed_count
    
    # Get thread-local client
    client = get_thread_client()
    
    # Process topic
    success = False
    try:
        # Call the processing function, but temporarily store results without directly adding to shared data
        temp_output = []
        success = process_and_update_realtime(client, topic, temp_output, processed_ids)
        
        # Use a lock to protect access to shared data
        with output_lock:
            # Add processing results to the shared output list
            if temp_output:
                output_data.extend(temp_output)
                
            # Update counter
            if success:
                completed_count += 1
        
        # Save results after processing each topic - removed conditional to ensure real-time saving
        save_results(output_data)
            
    except Exception as e:
        logger.error(f"Error in worker thread: {str(e)}")
        
    return success

def save_results(output_data):
    """Thread-safely saves results to the output file"""
    global last_save_time
    
    # Use a lock to prevent multiple threads from writing to the file simultaneously
    with save_lock:
        # Reduce the time limit for save frequency from 5 seconds to 1 second,
        # but still maintain a limit to avoid excessive disk I/O
        current_time = time.time()
        if current_time - last_save_time >= 1:  # Save at least once every 1 second to avoid overly frequent I/O
            if save_topics(CONFIG["output_file"], output_data):
                logger.info(f"Updated output file with {len(output_data)} topics (realtime)")
                last_save_time = current_time

def main():
    """Main function that orchestrates the entire process"""
    global completed_count
    
    # Create OpenAI client
    client = setup_client()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
    
    # Load input data
    try:
        logger.info(f"Loading data from {CONFIG['input_file']}")
        input_topics = load_topics(CONFIG["input_file"])
        logger.info(f"Loaded {len(input_topics)} topics from input file")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return
    
    # Load processed data and create index
    processed_topics = []
    processed_ids = {}
    if os.path.exists(CONFIG["output_file"]):
        processed_topics = load_topics(CONFIG["output_file"])
        if processed_topics:
            logger.info(f"Loaded {len(processed_topics)} existing processed topics")
            processed_ids = create_processed_topics_index(processed_topics)
            logger.info(f"Created index with {len(processed_ids)} processed topic IDs")
            
            # Count discarded topics
            discarded_count = sum(1 for t in processed_topics if is_discarded_topic(t))
            logger.info(f"Found {discarded_count} previously discarded topics with placeholders")
    
    # Create new output list
    output_data = list(processed_topics) if processed_topics else []
    
    # Limit processing quantity (if required)
    if CONFIG["max_items"] > 0:
        input_topics = input_topics[:CONFIG["max_items"]]
        logger.info(f"Limited processing to {CONFIG['max_items']} topics")
    
    # Calculate items to process (excluding already processed ones)
    topics_to_process = []
    for topic in input_topics:
        if 'id' in topic and topic['id'] in processed_ids:
            continue
        topics_to_process.append(topic)
    
    logger.info(f"Processing {len(topics_to_process)} new topics with {CONFIG['max_workers']} worker threads")
    
    # Process topics in parallel using a thread pool
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        # Submit all tasks
        future_to_topic = {
            executor.submit(process_topic_thread, topic, output_data, processed_ids): topic 
            for topic in topics_to_process
        }
        
        # Process completed tasks
        from tqdm import tqdm
        with tqdm(total=len(topics_to_process), desc="Processing topics") as pbar:
            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    success = future.result()
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Worker thread failed: {str(e)}")
    
    # Finalize processing
    if save_topics(CONFIG["output_file"], output_data):
        # Count discarded topics in the results
        discarded_count = sum(1 for t in output_data if is_discarded_topic(t))
        
        logger.info(f"Processing complete. Saved {len(output_data)} topics")
        logger.info(f"Final output includes {discarded_count} discarded topic placeholders")
    else:
        logger.error("Failed to save final output file")

if __name__ == "__main__":
    main()