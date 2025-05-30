#!/usr/bin/env python3
"""
Video Splitting Script - Splits videos based on timestamps from text_metadata.json and retains detailed metadata
"""

import os
import json
import subprocess
import argparse
import glob
import shutil
import logging
import re
from datetime import datetime
import time 
import concurrent.futures
from functools import partial
import threading


# Generate timestamp
timestamp = datetime.now().strftime("%m%d%H%M")

# Create logs directory
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Setup logging
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

def sanitize_filename(name):
    """Sanitize filename by removing illegal characters"""
    # Replace illegal characters with underscore
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    # Limit length
    return name[:100] if len(name) > 100 else name

def convert_timestamp_to_seconds(timestamp):
    """Convert HH:MM:SS.SSS formatted timestamp to seconds"""
    hours, minutes, seconds = timestamp.split(':')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total_seconds

def find_video_file(dir_path):
    """Search for a video file in the directory"""
    video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
    for ext in video_exts:
        files = glob.glob(os.path.join(dir_path, f"*{ext}"))
        if files:
            return files[0]
    return None

def extract_video_resolution(video_path):
    """Extract video resolution"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        width = data['streams'][0].get('width', 0)
        height = data['streams'][0].get('height', 0)
        return f"{width}x{height}"
    except Exception as e:
        logger.warning(f"Failed to extract video resolution: {e}")
        return "unknown"

def extract_video_duration(video_path):
    """Extract total video duration"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        logger.warning(f"Failed to extract video duration: {e}")
        return 0

def cut_video(video_path, start_time, end_time, output_path):
    """Use ffmpeg to split the video"""
    start_seconds = convert_timestamp_to_seconds(start_time)
    end_seconds = convert_timestamp_to_seconds(end_time)
    duration = end_seconds - start_seconds
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', start_time,
        '-t', str(duration),
        '-c:v', 'libx264',  # video codec
        '-c:a', 'aac',      # audio codec
        '-strict', 'experimental',
        '-b:a', '128k',
        '-y',               # overwrite output file
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, {"start_seconds": start_seconds, "end_seconds": end_seconds, "duration": duration}
    except subprocess.CalledProcessError as e:
        logger.error(f"Video splitting failed: {e}")
        logger.error(f"ffmpeg error: {e.stderr.decode('utf-8', errors='ignore')}")
        return False, {}

def process_single_video(subdir, source_dir, output_dir):
    """Process a single video directory"""
    thread_id = f"[Thread-{threading.get_ident() % 1000:03d}]"
    logger.info(f"{thread_id} Starting processing {subdir}")
    subdir_path = os.path.join(source_dir, subdir)
    
    # Find metadata file
    metadata_files = glob.glob(os.path.join(subdir_path, f"*_text_metadata.json"))
    if not metadata_files:
        logger.warning(f"Skipped {subdir}: Metadata file not found")
        return None, "invalid", 0  # return tuple (None, status, segment count)
    
    metadata_file = metadata_files[0]
    
    # Read metadata
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read metadata {subdir}: {e}")
        return None, "error", 0
    
    # Check text validity
    if not metadata.get("is_text_valid", False):
        logger.warning(f"Skipped {subdir}: Invalid text")
        return None, "invalid", 0
    
    # Find video file
    video_file = find_video_file(subdir_path)
    if not video_file:
        logger.warning(f"Skipped {subdir}: Video file not found")
        return None, "error", 0
    
    logger.info(f"Processing video in {subdir}...")
    
    # Get original video metadata
    video_id = metadata.get("id", subdir)
    video_topic = metadata.get("topic", "Unknown Topic")
    video_url = metadata.get("url", "")
    video_source = metadata.get("source", "Unknown")
    
    # Extract video features
    video_resolution = extract_video_resolution(video_file)
    video_duration = extract_video_duration(video_file)
    
    # Process each segment
    segments = metadata.get("segments", [])
    valid_segments = [seg for seg in segments if not seg.get("is_outro", False)]
    
    if not valid_segments:
        logger.warning(f"Skipped {subdir}: No valid segments")
        return None, "invalid", 0
    
    # Collect all info of original video
    original_video_info = {
        "id": video_id,
        "topic": video_topic,
        "url": video_url,
        "source": video_source,
        "resolution": video_resolution,
        "total_duration": video_duration,
        "segments_count": len(valid_segments),
        "original_metadata_file": os.path.basename(metadata_file),
        "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    segment_count = 0
    # Iterate each segment and split video
    for i, segment in enumerate(valid_segments):
        # ... [original segment processing code] ...
        # (Your original code remains unchanged)
        
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")
        content = segment.get("content", "").strip()
        
        if not (start_time and end_time and content):
            logger.warning(f"Skipped segment {i+1}/{len(valid_segments)}: Missing required information")
            continue
        
        # Create folder for this segment
        segment_name = f"{video_id}_seg{i+1:02d}"
        segment_dir = os.path.join(output_dir, segment_name)
        os.makedirs(segment_dir, exist_ok=True)
        
        # Split video
        output_video = os.path.join(segment_dir, f"{segment_name}.mp4")
        success, cut_info = cut_video(video_file, start_time, end_time, output_video)
        
        if success:
            # Create detailed metadata file
            segment_metadata = {
                # Basic identification
                "id": segment_name,
                "original_id": video_id,
                
                # Content info
                "topic": video_topic,
                "content": content,
                "is_outro": segment.get("is_outro", False),
                
                # Source info
                "source": video_source,
                "url": video_url,
                
                # Video info
                "resolution": video_resolution,
                
                # Time info
                "start_time": start_time,
                "end_time": end_time,
                "duration": cut_info.get("duration", 0),
                "start_seconds": cut_info.get("start_seconds", 0),
                "end_seconds": cut_info.get("end_seconds", 0),
                
                # Original video info
                "original_video": {
                    "id": video_id,
                    "total_duration": video_duration,
                    "segments_count": len(valid_segments)
                },
                
                # Processing info
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "segment_number": i+1
            }
            
            # Add extra metadata fields from original video
            for key, value in metadata.items():
                if key not in ["segments", "is_text_valid", "transcript_source", "id", "url", "source", "topic", "contains_outro"]:
                    segment_metadata[f"original_{key}"] = value
            
            # Save metadata
            with open(os.path.join(segment_dir, f"{segment_name}_metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(segment_metadata, f, ensure_ascii=False, indent=2)
            
            # Create concise info file (for quick browsing)
            summary_info = {
                "id": segment_name,
                "topic": video_topic,
                "content": content,
                "duration": f"{cut_info.get('duration', 0):.2f} seconds",
                "source": video_source,
                "url": video_url
            }
            
            with open(os.path.join(segment_dir, f"{segment_name}_summary.txt"), 'w', encoding='utf-8') as f:
                for key, value in summary_info.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Successfully processed segment {i+1}/{len(valid_segments)} - {segment_name}")
            segment_count += 1
        else:
            logger.error(f"Failed to split segment {i+1}/{len(valid_segments)}")
    
    # Save summary information of the original video
    with open(os.path.join(output_dir, f"{video_id}_original_info.json"), 'w', encoding='utf-8') as f:
        json.dump(original_video_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Finished processing {subdir}, generated {segment_count} segments")
    return video_id, "processed", segment_count

def process_directory(source_dir, output_dir, max_threads=16):
    """Process all video directories using a thread pool"""
    if not os.path.exists(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        return 0, 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    logger.info(f"Found {len(subdirs)} subdirectories")
    
    processed_count = 0
    invalid_count = 0
    error_count = 0
    segment_count = 0
    
    # Create partial function fixing source_dir and output_dir parameters
    process_func = partial(process_single_video, source_dir=source_dir, output_dir=output_dir)
    
    # Use thread pool to execute processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Start tasks
        future_to_subdir = {executor.submit(process_func, subdir): subdir for subdir in subdirs}
        
        # Handle results
        for future in concurrent.futures.as_completed(future_to_subdir):
            subdir = future_to_subdir[future]
            try:
                video_id, status, segments = future.result()
                
                if status == "processed":
                    processed_count += 1
                    segment_count += segments
                elif status == "invalid":
                    invalid_count += 1
                elif status == "error":
                    error_count += 1
                    
                logger.info(f"Progress: {processed_count + invalid_count + error_count}/{len(subdirs)}")
            
            except Exception as e:
                logger.error(f"Exception occurred while processing {subdir}: {e}")
                error_count += 1
    
    logger.info("=" * 50)
    logger.info(f"Processing complete")
    logger.info(f"Successfully processed videos: {processed_count}/{len(subdirs)}")
    logger.info(f"Skipped videos: {invalid_count}")
    logger.info(f"Errored videos: {error_count}")
    logger.info(f"Total number of generated segments: {segment_count}")
    logger.info("=" * 50)
    
    return processed_count, invalid_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Split videos based on metadata')
    parser.add_argument('--source_dir', type=str, default="",
                       help='Source directory containing video subdirectories')
    parser.add_argument('--output_dir', type=str, default="",
                       help='Output directory')
    parser.add_argument('--threads', type=int, default=16,
                       help='Number of processing threads; use multithreading to speed up processing')
    args = parser.parse_args()
    
    # Log the number of threads
    logger.info(f"Starting processing with {args.threads} threads")
    
    start_time = datetime.now()
    processed, invalid, errors = process_directory(args.source_dir, args.output_dir, args.threads)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Total processing time: {duration:.2f} seconds")
    logger.info(f"Average processing time per video: {duration/max(processed, 1):.2f} seconds")

if __name__ == "__main__":
    main()
#     # Run the main function