import os
import json
import glob
import time
import random
import datetime
import argparse
from pathlib import Path
import requests
import logging
import traceback
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm

# Generate timestamp
timestamp = datetime.datetime.now().strftime("%m%d%H%M")

# Create logs directory
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(logs_dir, f"process_metadata_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenAI API configuration
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def generate_topic_with_gpt(original_topic):
    """
    Use ChatGPT 4.1 to generate a new topic (English translation)
    """
    try:
        logger.info(f"Translating topic to English. Original: {original_topic[:50]}...")
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
            {"role": "system", "content": "You are a professional translator specializing in creating clean, concise English news headlines. Your primary goal is to translate the provided original titles, ensuring the English output is **completely faithful to the meaning and tone of the original**, without adding any extra information or altering the original intent. The output language must strictly be English."},
            {"role": "user", "content": f"""Translate the following news headline into English.

            Please remove all emojis, unnecessary symbols, source mentions (e.g., 'Sun News', 'Geo News'), format indicators (e.g., 'Shorts'), and hashtags.
            Please replace or remove separators, such as '|' and '·'.
            Focus on the core subject and key entities of the headline, and condense it into a concise English headline.

            **CRITICAL INSTRUCTION:** In the translation process, **absolutely do not change the meaning, tone, or emotional slant of the original headline.** Translate and summarize the original headline based *only* on the information provided in the original title. Do not introduce any content based on speculation or external knowledge to avoid inaccurate or "hallucinated" translations.
            
            **Special Case:** If the original title is determined to be meaningless or an error message (like an apology from an AI model), return an empty string "".

            Your output must be *only* the translated headline, containing no explanations or extra punctuation beyond what is necessary for the headline itself.

            Original title: {original_topic}
            """}
        ],
            max_tokens=50,
            temperature=0
        )
        new_topic = response.choices[0].message.content.strip()
        
        # If the generated title has quotes, remove them
        if new_topic.startswith('"') and new_topic.endswith('"'):
            new_topic = new_topic[1:-1].strip()
            
        logger.info(f"Translated topic: {new_topic}")
        return new_topic
    except Exception as e:
        logger.error(f"Error occurred when translating topic with ChatGPT: {e}")
        logger.error(traceback.format_exc())
        # On failure, return the original topic, but remove hashtags
        cleaned_topic = original_topic.split('#')[0].strip() if original_topic else "News Update"
        return cleaned_topic

def process_metadata_file(metadata_file):
    """
    Process a single metadata.json file
    - Read the original data
    - Translate the topic field
    - Return the updated data
    """
    try:
        logger.info(f"Processing metadata file: {metadata_file}")
        
        # Read metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Ensure metadata is valid and not empty
        if not metadata:
            logger.warning(f"Metadata file is empty: {metadata_file}")
            return None
        
        # Store the original topic
        original_topic = metadata.get('topic', '')
        if not original_topic:
            logger.warning(f"No topic field in metadata file: {metadata_file}")
            metadata['original_topic'] = ""
            metadata['topic'] = "News Update"
            return metadata
        
        # Save the original topic and translate to English
        metadata['original_topic'] = original_topic
        metadata['topic'] = generate_topic_with_gpt(original_topic)
        
        return metadata
    
    except json.JSONDecodeError:
        logger.error(f"Unable to parse JSON file: {metadata_file}")
        return None
    except Exception as e:
        logger.error(f"Error occurred when processing metadata file: {metadata_file}, {e}")
        logger.error(traceback.format_exc())
        return None

def process_single_directory(args):
    """
    Process metadata files in a single subdirectory
    
    Args:
        args: Dictionary containing subdirectory path
    
    Returns:
        (processed_count, skipped_count): Number of successfully processed and skipped files
    """
    subdir_path = args['subdir_path']
    subdir = os.path.basename(subdir_path)
    processed_count = 0
    skipped_count = 0
    
    # Find metadata files
    metadata_files = glob.glob(os.path.join(subdir_path, "*_metadata.json"))
    
    if not metadata_files:
        logger.warning(f"No metadata files found in subdirectory: {subdir_path}")
        return 0, 1
    
    # Process each metadata file
    for metadata_file in metadata_files:
        # Define output file name
        file_name = os.path.basename(metadata_file)
        base_name = file_name.replace("_metadata.json", "")
        output_file = os.path.join(subdir_path, f"{base_name}_final_metadata.json")
        
        # If output file exists, skip processing
        if os.path.exists(output_file):
            logger.debug(f"Output file already exists, skipping: {output_file}")
            skipped_count += 1
            continue
        
        # Process metadata
        updated_metadata = process_metadata_file(metadata_file)
        
        # If processed successfully, save result
        if updated_metadata:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_metadata, f, ensure_ascii=False, indent=2)
                logger.debug(f"Successfully saved to: {output_file}")
                processed_count += 1
            except Exception as e:
                logger.error(f"Error occurred when saving to {output_file}: {e}")
                skipped_count += 1
        else:
            logger.warning(f"Failed to process: {metadata_file}")
            skipped_count += 1
    
    return processed_count, skipped_count

def process_segments_directory(segments_dir, workers=4, quiet=False):
    """
    Process all subdirectories in the segments directory in parallel
    Process metadata files for each subdirectory
    
    Args:
        segments_dir: Path to the segments directory
        workers: Number of parallel worker threads
        quiet: Whether to reduce output information
    
    Returns:
        (processed_count, skipped_count): Total number of successfully processed and skipped files
    """
    # Get all subdirectories
    subdirs = [d for d in os.listdir(segments_dir) if os.path.isdir(os.path.join(segments_dir, d))]
    logger.info(f"Found {len(subdirs)} subdirectories, using {workers} threads for parallel processing")
    
    # Prepare task arguments
    task_args = []
    for subdir in subdirs:
        subdir_path = os.path.join(segments_dir, subdir)
        task_args.append({'subdir_path': subdir_path})
    
    total_processed = 0
    total_skipped = 0
    
    # Use a thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Create progress bar
        results = list(tqdm(
            executor.map(process_single_directory, task_args),
            total=len(task_args),
            desc="Processing Progress",
            disable=quiet
        ))
        
        # Aggregate results
        for processed, skipped in results:
            total_processed += processed
            total_skipped += skipped
    
    return total_processed, total_skipped

def main():
    """
    Main function
    """
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Process video metadata files in parallel')
    parser.add_argument('--segments-dir', type=str, 
                        default="",
                        help='Path to the segments directory')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of worker threads for parallel processing')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output information')
    args = parser.parse_args()
    
    # If quiet mode is enabled, adjust log level
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    segments_dir = args.segments_dir
    
    # Check if input directory exists
    if not os.path.exists(segments_dir):
        logger.error(f"Directory does not exist: {segments_dir}")
        return
    
    start_time = time.time()
    
    # Process metadata for all subdirectories in parallel
    logger.info(f"Start processing metadata files in {segments_dir}")
    processed_count, skipped_count = process_segments_directory(
        segments_dir, 
        workers=args.workers,
        quiet=args.quiet
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Output processing result
    logger.info(f"Processing complete! Successfully processed: {processed_count}, Skipped: {skipped_count}")
    logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    if processed_count > 0:
        per_file_time = elapsed_time / processed_count
        logger.info(f"Average processing time per file: {per_file_time:.2f} seconds")

if __name__ == "__main__":
    main()