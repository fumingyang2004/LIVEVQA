import os
import sys
import logging
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root path to system path
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '')
# Add these lines before importing modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"root_dir: {root_dir}")

# Import component modules
from qa_makers.config import CONFIG, LOG_DIR
from qa_makers.file_utils import determine_io_files, load_topics, save_results
from qa_makers.question_processor import process_topic_thread
# Import text evaluation module
from qa_makers.text_evaluator import evaluate_text_data

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate first-level multi-hop questions')
    parser.add_argument('--workers', type=int, default=CONFIG["max_workers"], 
                        help='Number of threads for parallel processing')
    parser.add_argument('--input', type=str, 
                        help='Input file path (overrides automatically selected latest file)')
    parser.add_argument('--output', type=str,
                        help='Output file path (overrides automatically generated path)')
    parser.add_argument('--continue', dest='continue_processing', action='store_true',
                        help='Continue processing existing output file')
    parser.add_argument('--timestamp', '-t', type=str, 
                        help='Specify the timestamp of the modified_topics file to process, e.g., 04181718')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='Whether to evaluate text validity and filter meaningless texts')
    
    # Ensure argument parsing is correct
    try:
        args = parser.parse_args()
    except Exception as e:
        logger.error(f"Argument parsing error: {e}")
        return 1
    
    # Determine input and output files
    input_file, output_file, timestamp = determine_io_files(args)
    if not input_file or not output_file:
        logger.error("Unable to determine input or output file")
        return 1
    
    # Validate input file format
    input_filename = os.path.basename(input_file)
    if not (input_filename.startswith("modified_topics_") and 
            not any(suffix in input_filename for suffix in ["_evaluated", "_evaluated_all", "_filtered"])):
        logger.warning(f"Input file '{input_filename}' does not conform to 'modified_topics_{{timestamp}}.json' format")
        user_confirm = input("Input file format may be incorrect, continue processing? (y/n): ")
        if user_confirm.lower() != 'y':
            logger.info("User cancelled operation")
            return 0
    
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
        logger.error(f"Unable to load input file or file is empty: {CONFIG['input_file']}")
        return 1
    
    logger.info(f"Loaded {len(input_topics)} topics from {CONFIG['input_file']}")
    
    logger.info("Start evaluating text validity...")
    # Generate evaluation and filtered file paths with original timestamp
    evaluated_file = os.path.join(
        os.path.dirname(input_file), 
        f"modified_topics_{timestamp}_evaluated.json"
    )
    filtered_file = os.path.join(
        os.path.dirname(input_file), 
        f"modified_topics_{timestamp}_filtered.json"
    )
    
    # Evaluate text and get meaningful entries
    _, meaningful_topics = evaluate_text_data(
        input_file_path=input_file,
        output_file_path=evaluated_file,
        save_filtered=True,
        filtered_output_path=filtered_file,
        max_workers=args.workers
    )
    
    # Use filtered meaningful topics
    input_topics = meaningful_topics
    logger.info(f"Evaluation completed, retained {len(input_topics)} meaningful topics, saved to {filtered_file}")
    
    # Update input file to filtered file to ensure subsequent processing uses filtered file
    CONFIG["input_file"] = filtered_file
    
    # Initialize output data and processed ID set
    output_data = []
    processed_ids = set()
    
    # Check if output file exists and load processed IDs
    if os.path.exists(CONFIG["output_file"]):
        existing_output = load_topics(CONFIG["output_file"])
        if existing_output:
            # If using --continue parameter, use existing output as base
            if args.continue_processing:
                output_data = existing_output
                logger.info(f"Loaded {len(output_data)} processed topics from {CONFIG['output_file']}")
            # Extract processed IDs for deduplication regardless of --continue
            processed_ids = {topic.get('id') for topic in existing_output if topic.get('id') is not None}
            logger.info(f"Found {len(processed_ids)} processed IDs, will skip these IDs")
    
    # Filter out already processed topics
    topics_to_process = []
    for topic in input_topics:
        topic_id = topic.get('id')
        if topic_id is None:
            # Topics without ID also need processing
            topics_to_process.append(topic)
        elif topic_id not in processed_ids:
            # Only process IDs not processed before
            topics_to_process.append(topic)
        else:
            logger.info(f"Skipping already processed topic ID: {topic_id}")
    
    logger.info(f"After filtering, {len(topics_to_process)} new topics to process, using {CONFIG['max_workers']} threads")
    
    # If no new topics to process, return directly
    if not topics_to_process:
        logger.info("No new topics to process, program ends")
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
