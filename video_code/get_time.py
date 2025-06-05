import re
import sys
import os
import glob
import argparse
import logging
import datetime

timestamp = datetime.datetime.now().strftime("%m%d%H%M")

os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/word_timestamps{timestamp}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_vtt_word_timestamps(vtt_path):
    """
    Parse VTT file to extract words with their timestamps
    Rules:
    - Each word's timestamp is the time tag that follows it
    - If a word has no following time tag, use the current block's end time
    - Ignore plain text lines without inline time tags
    """
    word_times = []
    block_end = None
    time_block_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")
    inline_pattern = re.compile(r"<(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})><c>(?P<word>[^<]+)</c>")

    try:
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Update block end time
                m = time_block_pattern.search(line)
                if m:
                    block_end = m.group(2)
                    continue
                
                # Skip empty lines or metadata lines
                if not line or line.startswith(('WEBVTT', 'Kind:', 'Language:', 'align:', 'position:')):
                    continue
                
                # Find all inline time tags
                found = list(inline_pattern.finditer(line))
                if not found:
                    # If no inline tags, ignore the entire text line
                    logger.debug(f"Ignoring plain text line without time tags: {line}")
                    continue
                
                # Process line with inline tags
                words_with_times = []
                
                # Extract all words with their following timestamps
                for j, match in enumerate(found):
                    word = match.group('word').strip()
                    next_time = None
                    
                    # Get next timestamp if exists
                    if j < len(found) - 1:
                        next_time = found[j+1].group('time')
                    else:
                        next_time = block_end  # Use block end time
                    
                    words_with_times.append((word, next_time))
                
                # Handle first word (might be before tags)
                first_match_start = found[0].start()
                if first_match_start > 0:
                    prefix_text = line[:first_match_start].strip()
                    if prefix_text:
                        # Prefix words use first tag's time
                        for w in prefix_text.split():
                            if w.strip():
                                word_times.append((w.strip(), found[0].group('time')))
                
                # Add main word list
                word_times.extend(words_with_times)
                
                # Handle text after last tag
                last_match_end = found[-1].end()
                if last_match_end < len(line):
                    suffix_text = line[last_match_end:].strip()
                    suffix_text = re.sub(r'<[^>]+>', '', suffix_text)  # Remove any remaining tags
                    if suffix_text:
                        # Suffix words use block end time
                        for w in suffix_text.split():
                            if w.strip():
                                word_times.append((w.strip(), block_end))
                
    except Exception as e:
        logger.error(f"Error parsing file {vtt_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return word_times

def process_vtt_file(vtt_file_path, output_path):
    """
    Process a single VTT file, extract word timestamps and save to output file
    
    Args:
        vtt_file_path: Path to VTT file
        output_path: Output file path
        
    Returns:
        bool: Returns True if successful, False otherwise
    """
    try:
        logger.info(f"Processing file: {vtt_file_path}")
        word_times = parse_vtt_word_timestamps(vtt_file_path)
        
        if not word_times:
            logger.warning(f"No word timestamps extracted from VTT file: {vtt_file_path}")
            return False
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for word, time in word_times:
                f.write(f"{word} <{time}>\n")
        
        logger.info(f"Successfully processed file: {vtt_file_path}, extracted {len(word_times)} timestamped words")
        return True
    
    except Exception as e:
        logger.error(f"Error processing file {vtt_file_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_directory(source_dir):
    """
    Process all .en.vtt files in subdirectories of the specified directory
    """
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    logger.info(f"Found {len(subdirs)} subdirectories")
    
    processed_count = 0
    error_count = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        vtt_file = os.path.join(subdir_path, f"{subdir}.en.vtt")
        output_file = os.path.join(subdir_path, "word_times.txt")
        
        if not os.path.exists(vtt_file):
            logger.warning(f"Skipping {subdir}: .en.vtt file not found")
            continue
        
        if process_vtt_file(vtt_file, output_file):
            processed_count += 1
        else:
            error_count += 1
    
    logger.info("=" * 50)
    logger.info(f"Processing complete, successfully processed {processed_count}/{len(subdirs)} subdirectories")
    logger.info(f"Failed to process: {error_count}")
    logger.info("=" * 50)
    
    return processed_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Extract word timestamps from .en.vtt files')
    parser.add_argument('--source_dir', type=str, default="",
                      help='Source directory containing video subdirectories')
    args = parser.parse_args()
    
    processed_count, error_count = process_directory(args.source_dir)
    
    print(f"Successfully processed {processed_count} subdirectories, failed {error_count}")

if __name__ == '__main__':
    main()