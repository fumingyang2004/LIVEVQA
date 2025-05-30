"""
Multi-hop Question Generator

This script reads the l1_filtered_topics file and then generates Level 2 questions for each entry.
Question generation is based on the answers to Level 1 questions, building a high-difficulty multi-hop reasoning structure.
"""

import os
import re
import json
import glob
import sys
import base64
import logging
import argparse
import traceback
from config import BASE_DIR

# Add the project root directory to the Python path to allow package imports to work correctly
sys.path.insert(0, BASE_DIR)

from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Import internal modules
from qa_makers_mh.config import BASE_DIR, DATA_DIR, SAVE_INTERVAL, MAX_WORKERS
from qa_makers_mh.utils import (get_latest_filtered_topics_file, 
                                get_timestamp_from_filename, 
                                determine_io_files,
                                load_input_file, 
                                save_results,
                                encode_image_to_base64)
from qa_makers_mh.prompt_generator import create_multihop_prompt
from qa_makers_mh.api_client import create_client, generate_multihop_questions
from qa_makers_mh.processor import process_topic, process_topic_thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "multihop_generator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_multihop_questions_main(input_file=None, output_file=None, timestamp=None, workers=MAX_WORKERS):
    """
    Main function for generating multi-hop questions, which can be imported and called by other scripts.
    
    Args:
        input_file: Path to the input file, automatically found if None.
        output_file: Path to the output file, generated based on the input file if None.
        timestamp: Timestamp to find a specific input file.
        workers: Number of threads for parallel processing.
        
    Returns:
        str: Path to the output file if successful, None otherwise.
    """
    # Create an args object to reuse the determine_io_files function
    class Args:
        pass
    args = Args()
    args.input = input_file
    args.output = output_file
    args.timestamp = timestamp
    args.workers = workers
    
    # Determine input and output files
    input_file, output_file, timestamp = determine_io_files(args)
    if not input_file or not output_file:
        logger.error("Could not determine input or output files, exiting.")
        return None
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    if timestamp:
        logger.info(f"Timestamp: {timestamp}")
    
    # Create OpenAI client
    client = create_client()
    
    # Load input file
    topics = load_input_file(input_file)
    if not topics:
        logger.error("Failed to load topics, exiting.")
        return None
    
    # Create results list (copy of original topics)
    results = topics.copy()
    
    # Create thread pool
    max_workers = min(args.workers, len(topics))
    logger.info(f"Using {max_workers} threads to process {len(topics)} topics.")
    
    # Count topics that need processing
    topics_to_process = [topic for topic in topics if not topic.get('discarded', False)]
    logger.info(f"There are {len(topics_to_process)} non-discarded topics to process.")
    
    # Process topics using a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_index = {}
        for i, topic in enumerate(topics):
            if not topic.get('discarded', False):
                future = executor.submit(process_topic_thread, (client, topic, results, i))
                future_to_index[future] = i
        
        # Process results
        processed_count = 0
        with tqdm(total=len(future_to_index), desc="Generating Multi-hop Questions") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    success = future.result()
                    if success:
                        processed_count += 1
                    
                    # Periodically save results
                    if processed_count % SAVE_INTERVAL == 0:
                        save_results(output_file, results)
                        
                except Exception as e:
                    logger.error(f"Error processing topic at index {index}: {str(e)}")
                    logger.error(traceback.format_exc())
                
                pbar.update(1)
    
    # Save final results
    save_results(output_file, results)
    
    # Calculate statistics
    total_level2 = 0
    for topic in results:
        if topic.get('discarded', False):
            continue
        for key in topic.keys():
            if key.startswith('level2_qas_img'):
                total_level2 += len(topic[key])
    
    logger.info(f"Processing complete! Successfully processed {processed_count}/{len(topics_to_process)} topics.")
    logger.info(f"A total of {total_level2} Level 2 questions were generated.")
    logger.info(f"Results saved to {output_file}")
    
    return output_file

def main():
    """Command-line entry point function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate multi-hop questions')
    parser.add_argument('--timestamp', '-t', type=str, 
                        help='Specify the timestamp of the l1_filtered_topics file to process, e.g., 04191836')
    parser.add_argument('--input', '-i', type=str,
                        help='Input file path (overrides auto-selected latest file)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path (overrides auto-generated path)')
    parser.add_argument('--workers', '-w', type=int, default=MAX_WORKERS,
                        help='Number of threads for parallel processing')
    args = parser.parse_args()
    
    # Call the main function
    return generate_multihop_questions_main(
        input_file=args.input,
        output_file=args.output,
        timestamp=args.timestamp,
        workers=args.workers
    )

if __name__ == "__main__":
    main()