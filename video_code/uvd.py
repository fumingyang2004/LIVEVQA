import os
import logging
import torch
import uvd
import decord
import cv2
import argparse
import numpy as np
from datetime import datetime
import glob
import traceback
import subprocess
import re
import shutil
import json
import concurrent.futures
import threading
import time
import psutil
import gc
from functools import lru_cache

# UVD model manager class, implements singleton pattern to ensure model sharing and reuse
class UVDModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(UVDModelManager, cls).__new__(cls)
                cls._instance.models = {}
                cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._instance.is_initializing = {}
                cls._instance.init_locks = {}
            return cls._instance
    
    def get_model(self, preprocessor_name):
        """Get or load the specified UVD model"""
        preprocessor_key = preprocessor_name.lower().replace("-", "")
        
        # If the model is already initializing, wait for initialization to complete
        if preprocessor_key in self.is_initializing:
            if preprocessor_key not in self.init_locks:
                self.init_locks[preprocessor_key] = threading.Lock()
            
            with self.init_locks[preprocessor_key]:
                if preprocessor_key in self.models:
                    return preprocessor_key
        
        # If the model is already loaded, return directly
        if preprocessor_key in self.models:
            return preprocessor_key
        
        # Mark the model as initializing
        self.is_initializing[preprocessor_key] = True
        if preprocessor_key not in self.init_locks:
            self.init_locks[preprocessor_key] = threading.Lock()
        
        with self.init_locks[preprocessor_key]:
            try:
                # Clean up GPU memory before loading
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                logging.info(f"Loading UVD model: {preprocessor_name}")
                # The model is not loaded directly here, but will be loaded and cached inside uvd on first use via get_uvd_subgoals
                self.models[preprocessor_key] = True
                logging.info(f"UVD model {preprocessor_name} is ready")
                return preprocessor_key
            except Exception as e:
                logging.error(f"Failed to load UVD model {preprocessor_name}: {e}")
                if preprocessor_key in self.is_initializing:
                    del self.is_initializing[preprocessor_key]
                return None
            finally:
                if preprocessor_key in self.is_initializing:
                    del self.is_initializing[preprocessor_key]
    
    def release_model(self, preprocessor_name=None):
        """Release resources for the specified model or all models"""
        if preprocessor_name:
            preprocessor_key = preprocessor_name.lower().replace("-", "")
            if preprocessor_key in self.models:
                del self.models[preprocessor_key]
                logging.info(f"Released model: {preprocessor_name}")
        else:
            self.models.clear()
            logging.info("Released all models")
        
        # Clean up GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

# GPU memory monitoring class
class GPUMemoryMonitor:
    def __init__(self, threshold_mb=4000):
        self.threshold_mb = threshold_mb
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        if self.device != "cuda":
            return 0
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0
    
    def get_system_memory_usage(self):
        """Get system memory usage"""
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 0
    
    def is_memory_available(self):
        """Check if enough GPU memory is available"""
        if self.device != "cuda":
            return True
        
        current_usage = self.get_gpu_memory_usage()
        max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        available = max_memory - current_usage
        
        if available < self.threshold_mb:
            logging.warning(f"Insufficient GPU memory: {available:.2f}MB available, below threshold {self.threshold_mb}MB")
            return False
        return True
    
    def wait_for_memory(self, max_wait_seconds=300):
        """Wait until enough GPU memory is available"""
        if self.device != "cuda":
            return True
        
        start_time = time.time()
        while not self.is_memory_available():
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                logging.error(f"Timeout waiting for GPU memory: {max_wait_seconds} seconds")
                return False
            
            # Force clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            logging.info(f"Waiting for GPU memory release: current usage {self.get_gpu_memory_usage():.2f}MB, system memory usage {self.get_system_memory_usage()}%")
            time.sleep(5)
        
        return True

# Use LRU cache decorator to optimize frequently called functions
@lru_cache(maxsize=128)
def extract_video_fps(video_path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                return num / den if den != 0 else None
            else:
                return float(fps_str)
        return None
    except Exception as e:
        logging.error(f"Failed to get video frame rate: {e}")
        return None

# Concurrent processing of video segments
def process_video_segments(segment_files, preprocessor, output_dir, max_workers=3):
    """Process multiple video segments in parallel, controlling concurrency"""
    model_manager = UVDModelManager()
    memory_monitor = GPUMemoryMonitor()
    all_keyframes_info = []
    segment_success = 0
    segment_error = 0
    
    # Ensure model is loaded
    preprocessor_key = model_manager.get_model(preprocessor)
    if not preprocessor_key:
        logging.error(f"Model {preprocessor} failed to load, unable to process video segments")
        return 0, 0, []
    
    # Use thread pool to control concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_segment = {
            executor.submit(
                process_single_segment, segment_path, preprocessor, output_dir, memory_monitor
            ): segment_path for segment_path in segment_files
        }
        
        # Handle completed tasks
        for future in concurrent.futures.as_completed(future_to_segment):
            segment_path = future_to_segment[future]
            segment_name = os.path.basename(segment_path)
            try:
                result, keyframes_info = future.result()
                if result:
                    if keyframes_info:
                        # Add segment info
                        for kf_info in keyframes_info:
                            kf_info["segment_file"] = segment_name
                        all_keyframes_info.extend(keyframes_info)
                    segment_success += 1
                    logging.info(f"Segment processed successfully: {segment_name}")
                else:
                    segment_error += 1
                    logging.error(f"Segment processing failed: {segment_name}")
            except Exception as e:
                segment_error += 1
                logging.error(f"Exception occurred while processing segment {segment_name}: {e}")
    
    return segment_success, segment_error, all_keyframes_info

def process_single_segment(segment_path, preprocessor, output_dir, memory_monitor):
    """Process a single video segment, including memory management and error handling"""
    segment_name = os.path.basename(segment_path)
    logging.info(f"Processing video segment: {segment_name}")
    
    # Wait for enough GPU memory
    if not memory_monitor.wait_for_memory():
        logging.error(f"Unable to allocate enough GPU memory for segment {segment_name}, skipping processing")
        return None, None
    
    try:
        # Save keyframes directly to the imp folder
        result, segment_keyframes_info = extract_keyframes(
            segment_path, 
            preprocessor_name=preprocessor, 
            output_dir=output_dir
        )
        return result, segment_keyframes_info
    except Exception as e:
        logging.error(f"Error processing segment {segment_name}: {e}")
        logging.error(traceback.format_exc())
        return None, None

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"cut_key_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def convert_to_h264(video_path):
    """Use ffmpeg to convert video to H.264 encoding, save in the same directory, and delete the original file after success."""
    try:
        video_dir = os.path.dirname(video_path)
        base_filename, ext = os.path.splitext(os.path.basename(video_path))

        # Always output mp4 to avoid inconsistent extensions after transcoding
        out_filename = f"{base_filename}_h264.mp4"
        out_path = os.path.join(video_dir, out_filename)

        if os.path.abspath(video_path) == os.path.abspath(out_path):
            logging.error(f"  Error: transcoding input and output path are the same, skipping transcoding: {video_path}")
            return None
        
        if os.path.exists(out_path):
            logging.info(f"  Target H.264 file already exists, skipping conversion: {out_filename}")
            # Consider whether to delete the original file? If the target already exists, maybe the original file should also be deleted
            try:
                if os.path.exists(video_path): # Confirm again that the original file exists
                    os.remove(video_path)
                    logging.info(f"    Deleted redundant original file: {os.path.basename(video_path)}")
            except OSError as del_e:
                logging.warning(f"    Failed to delete redundant original file: {del_e}")
            return out_path


        cmd = [
            "ffmpeg", "-y", # Overwrite output if somehow exists but check failed
            "-i", video_path,
            "-c:v", "libx264", # H.264 codec
            "-preset", "fast", # Encoding speed vs size trade-off
            "-crf", "23", # Constant Rate Factor (quality)
            "-c:a", "aac", # AAC audio codec
            "-b:a", "128k", # Audio bitrate
            "-strict", "-2", # Needed for AAC sometimes
            "-loglevel", "error", # Only show error messages
            out_path
        ]
        logging.info(f"  Transcoding to H.264: {os.path.basename(video_path)} -> {out_filename}")

        result = subprocess.run(cmd, check=False, capture_output=True, text=True) # check=False, capture output

        if result.returncode == 0:
            logging.info(f"  Transcoding successful: {out_filename}")
            # Delete the original file after successful transcoding
            try:
                os.remove(video_path)
                logging.info(f"    Deleted original file: {os.path.basename(video_path)}")
            except OSError as del_e:
                logging.warning(f"    Failed to delete original file: {del_e}")
            return out_path
        else:
            logging.error(f"  Transcoding failed: {os.path.basename(video_path)}")
            logging.error(f"    ffmpeg error: {result.stderr.strip()}")
            # Try to delete possibly incomplete output file
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return None

    except Exception as e:
        logging.error(f"  Unknown error occurred during transcoding: {e}")
        logging.error(traceback.format_exc())
        return None

def segment_video(video_path, output_dir, segment_length=60):
    """Split the video into segments of the specified length (seconds)
    
    Args:
        video_path: input video path
        output_dir: output directory
        segment_length: length of each segment in seconds, default 60 seconds
        
    Returns:
        List of paths of split segments, empty list if failed
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get safe base filename
        safe_basename = re.sub(r'[\\/*?:"<>|]', "_", os.path.splitext(os.path.basename(video_path))[0])
        
        # Get video info (duration)
        probe_cmd = [
            "ffprobe", "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            video_path
        ]
        
        try:
            duration = float(subprocess.check_output(probe_cmd).decode('utf-8').strip())
        except (subprocess.SubprocessError, ValueError) as e:
            logging.error(f"  Failed to get video duration: {e}")
            return []
        
        # If video duration is less than 1 minute, return the original video
        if (duration <= segment_length):
            logging.info(f"  Video duration is only {duration:.2f} seconds, less than segment length {segment_length} seconds, skipping segmentation")
            return [video_path]
            
        # Calculate the number of segments needed
        segment_count = int(duration / segment_length) + (1 if duration % segment_length > 0 else 0)
        logging.info(f"  Video duration {duration:.2f} seconds, will be split into {segment_count} segments")
        
        segment_paths = []
        
        # Split video
        for i in range(segment_count):
            start_time = i * segment_length
            # 构建输出文件名
            segment_filename = f"{safe_basename}_seg_{i+1:03d}.mp4"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # If file exists, skip
            if os.path.exists(segment_path):
                logging.info(f"    Segment {i+1}/{segment_count} already exists: {segment_filename}")
                segment_paths.append(segment_path)
                continue
                
            # Build ffmpeg command, use H.264 encoding
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-t", str(segment_length),
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-loglevel", "error",
                segment_path
            ]
            
            logging.info(f"    Splitting segment {i+1}/{segment_count}: {segment_filename}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"    Segment {i+1}/{segment_count} split successfully")
                segment_paths.append(segment_path)
            else:
                logging.error(f"    Segment {i+1}/{segment_count} split failed: {result.stderr.strip()}")
        
        logging.info(f"  Video segmentation complete, {len(segment_paths)} segments generated")
        return segment_paths
        
    except Exception as e:
        logging.error(f"  Unknown error occurred during video segmentation: {e}")
        logging.error(traceback.format_exc())
        return []

def extract_keyframes(video_path, preprocessor_name="VIP", output_dir=None, batch_size=64):
    """Extract keyframes in batches to reduce memory usage, and return keyframe information"""
    try:
        # Validate video path
        if not os.path.exists(video_path):
            logging.error(f"Error: video file does not exist: {video_path}")
            return None, None
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Using preprocessor: {preprocessor_name}")

        frame_original_res = None
        total_frames = 0

        # Try to read video using decord
        try:
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            logging.info(f"Total frames in video (Decord): {total_frames}")

            if total_frames == 0:
                raise ValueError("Decord read 0 frames")

            # If video is too long, read in chunks to avoid integer overflow
            if total_frames > 1500:
                logging.info(f"Large number of frames, using chunked processing (Decord)")
                frame_chunks = []
                chunk_size = 1500

                for i in range(0, total_frames, chunk_size):
                    end_idx = min(i + chunk_size, total_frames)
                    logging.info(f"Reading frames {i} to {end_idx-1} (Decord)")
                    chunk = vr[i:end_idx].asnumpy()
                    frame_chunks.append(chunk)

                frame_original_res = np.concatenate(frame_chunks, axis=0)
            else:
                logging.info(f"Reading all frames at once (Decord)")
                frame_original_res = vr[:].asnumpy()

        except (decord.DECORDError, ValueError, RuntimeError, Exception) as decord_error:
            logging.warning(f"Decord reading failed, trying OpenCV: {str(decord_error)}")

            # Use OpenCV as fallback
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logging.error(f"OpenCV cannot open video file: {video_path}")
                    raise Exception("Cannot read video file (OpenCV)")

                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()

                if not frames:
                    logging.error("No frames read from video (OpenCV)")
                    raise Exception("Failed to read video frames (OpenCV)")

                frame_original_res = np.array(frames)
                total_frames = len(frame_original_res)
                logging.info(f"Successfully read {total_frames} frames with OpenCV")
            except Exception as opencv_error:
                logging.error(f"OpenCV reading also failed: {opencv_error}")
                raise opencv_error

        if frame_original_res is None or total_frames == 0:
            logging.error("Failed to successfully read any video frames.")
            return None, None

        logging.info(f"Successfully read video frames, shape: {frame_original_res.shape}")

        # Get keyframe indices
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # Use model manager to get model
        model_manager = UVDModelManager()
        preprocessor_key = model_manager.get_model(preprocessor_name)
        if not preprocessor_key:
            logging.error(f"Failed to get preprocessor {preprocessor_name}")
            return None, None

        indices = None
        try:
            # Clean up GPU memory before call
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                
            indices_np = uvd.get_uvd_subgoals(
                video_path,
                preprocessor_key,
                device=device,
                return_indices=True,
            )

            if indices_np is not None:
                # Ensure return value is a 1D int array to avoid list<->int comparison errors
                indices_np = np.asarray(indices_np).astype(int).ravel()
                # Filter out-of-bounds indices
                indices_np = indices_np[indices_np < total_frames]
                indices = indices_np.tolist()
                logging.info(f"Obtained keyframe indices: {indices}")
        except Exception as e:
            logging.error(f"Error obtaining keyframe indices: {str(e)}")

        # If UVD fails or does not return indices, use uniform sampling
        if indices is None or not indices:
            logging.info("Using uniform sampling as fallback")
            num_frames_sample = min(10, total_frames)
            if total_frames > 0:
                indices = np.linspace(0, total_frames - 1, num_frames_sample, dtype=int).tolist()
                logging.info(f"Uniform sampling indices: {indices}")
            else:
                logging.error("Cannot perform uniform sampling, total frames is 0")
                return None, None

        if not indices:
            logging.warning(f"Warning: No keyframe indices found or generated")
            return None, None

        # Check again if indices are out of range
        valid_indices = [i for i in indices if 0 <= i < total_frames]
        if len(valid_indices) < len(indices):
            logging.warning(f"Warning: {len(indices) - len(valid_indices)} indices out of range, filtered")
            indices = valid_indices

        if not indices:
            logging.error("All indices out of range, unable to extract keyframes")
            return None, None

        # Extract keyframes
        subgoals = frame_original_res[indices]
        logging.info(f"Successfully extracted {len(subgoals)} keyframes")

        # Create keyframe index info, including frame index and time (estimated using frame index and total frames)
        frames_info = []
        fps = extract_video_fps(video_path) or 30  # Use 30fps by default
        
        for i, idx in enumerate(indices):
            time_in_seconds = idx / fps
            minutes = int(time_in_seconds // 60)
            seconds = time_in_seconds % 60
            frames_info.append({
                "frame_index": int(idx),
                "time_seconds": time_in_seconds,
                "time_formatted": f"{minutes:02d}:{seconds:06.3f}",
                "order": i + 1
            })

        # If need to save
        if output_dir:
            saved_paths, keyframe_details = save_frames(subgoals, video_path, output_dir)
            
            # Merge index info and saved file info
            for i in range(len(keyframe_details)):
                if i < len(frames_info):
                    keyframe_details[i].update(frames_info[i])
            
            return saved_paths, keyframe_details

        return subgoals, frames_info

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        logging.error("Detailed error info:")
        logging.error(traceback.format_exc())
        return None, None

def save_frames(frames, video_path, output_dir):
    """Save keyframes to the specified directory and return keyframe information"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use sanitized filename as base to avoid special character issues
    safe_basename = re.sub(r'[\\/*?:"<>|]', "_", os.path.splitext(os.path.basename(video_path))[0])
    saved_paths = []
    keyframe_info = []

    for i, frame in enumerate(frames):
        # OpenCV uses BGR format, but frames are RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_filename = f"{safe_basename}_keyframe_{timestamp}_{i+1}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)

        try:
            cv2.imwrite(frame_path, frame_bgr)
            saved_paths.append(frame_path)
            
            # Record related info for each keyframe
            frame_info = {
                "index": i + 1,
                "filename": frame_filename,
                "path": frame_path,
                "timestamp": timestamp,
                "base_video": os.path.basename(video_path),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            keyframe_info.append(frame_info)
        except Exception as e:
            logging.error(f"Error saving frame: {str(e)}")

    logging.info(f"Saved {len(saved_paths)} keyframes to {output_dir}")
    return saved_paths, keyframe_info

def save_keyframes_metadata(directory, keyframes_data):
    """Save keyframe metadata to JSON file"""
    try:
        metadata_file = os.path.join(directory, "keyframes_metadata.json")
        
        # Add timestamp for tracking
        metadata = {
            "directory": os.path.basename(directory),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "keyframes": keyframes_data,
            "total_keyframes": len(keyframes_data)
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved keyframe metadata to {metadata_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to save keyframe metadata: {e}")
        logging.error(traceback.format_exc())
        return False

def has_keyframes(video_path, output_dir):
    """Check whether the specified video already has keyframes"""
    safe_basename = re.sub(r'[\\/*?:"<>|]', "_", os.path.splitext(os.path.basename(video_path))[0])
    keyframe_pattern = os.path.join(output_dir, f"{safe_basename}_keyframe_*.jpg")
    return len(glob.glob(keyframe_pattern)) > 0

def check_video_duration(video_path):
    """Check the duration (seconds) of the video file
    
    Args:
        video_path: video file path
        
    Returns:
        Video duration in seconds, or None if failed
    """
    try:
        probe_cmd = [
            "ffprobe", "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            video_path
        ]
        
        result = subprocess.run(probe_cmd, check=False, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True)
        
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            logging.error(f"Failed to get duration for video {os.path.basename(video_path)}: {result.stderr.strip()}")
            return None
    except Exception as e:
        logging.error(f"Error checking video duration: {e}")
        return None

def process_directory(root_dir, preprocessor="VIP", shard_index=0, shard_count=1, 
                     start_index=None, end_index=None, sequential=False, max_parallel=3):
    """Process videos in all first-level subdirectories of the specified directory
    
    Args:
        root_dir: root directory path
        preprocessor: preprocessor name to use
        shard_index: current shard index
        shard_count: total number of shards
        start_index: manually specify start index (optional)
        end_index: manually specify end index (optional)
        sequential: use sequential allocation (True) or even/odd allocation (False)
        max_parallel: maximum number of concurrent processes
    """
    processed_count = 0
    skipped_count = 0
    error_count = 0
    converted_count = 0
    segmented_count = 0
    skipped_dirs = 0
    invalid_dirs = 0
    removed_dirs = 0

    # Get first-level subdirectories
    try:
        all_subdirs = [d.path for d in os.scandir(root_dir) if d.is_dir()]
    except FileNotFoundError:
        logging.error(f"Error: root directory not found: {root_dir}")
        return

    # Sort directories to ensure consistent order across runs
    all_subdirs.sort()
    
    # Pre-filter unprocessed directories
    unprocessed_subdirs = []
    for subdir in all_subdirs:
        imp_dir = os.path.join(subdir, "imp")
        metadata_file = os.path.join(imp_dir, "keyframes_metadata.json")
        
        # Check if already processed (imp directory exists and has metadata file)
        if os.path.exists(metadata_file):
            logging.info(f"Directory already processed, skipping: {subdir}")
            skipped_dirs += 1
            continue
                
        # Check if directory contains any MP4 file longer than 600 seconds
        mp4_files = glob.glob(os.path.join(subdir, '*.mp4'))
        has_long_video = False
        
        for mp4_file in mp4_files:
            duration = check_video_duration(mp4_file)
            if duration is not None and duration > 600:
                logging.warning(f"Found invalid data: {os.path.basename(mp4_file)} duration is {duration:.2f} seconds (>300 seconds)")
                has_long_video = True
                invalid_dirs += 1
                break
                
        if has_long_video:
            logging.info(f"Deleting invalid directory: {subdir}")
            try:
                shutil.rmtree(subdir)
                removed_dirs += 1
                logging.info(f"Directory deleted successfully: {subdir}")
            except Exception as e:
                logging.error(f"Failed to delete directory: {subdir}, error: {e}")
            continue
            
        unprocessed_subdirs.append(subdir)
    
    # Assign subdirectories according to specified method
    if start_index is not None or end_index is not None:
        # Use manually specified index range
        start = 0 if start_index is None else start_index
        end = len(unprocessed_subdirs) if end_index is None else end_index
        subdirs = unprocessed_subdirs[start:end]
        logging.info(f"Using manually specified index range: {start}:{end}")
    elif sequential:
        # Use sequential allocation
        total_dirs = len(unprocessed_subdirs)
        dirs_per_shard = total_dirs // shard_count
        remainder = total_dirs % shard_count
        
        start_idx = shard_index * dirs_per_shard + min(shard_index, remainder)
        end_idx = start_idx + dirs_per_shard + (1 if shard_index < remainder else 0)
        
        subdirs = unprocessed_subdirs[start_idx:end_idx]
        logging.info(f"Using sequential allocation: GPU {shard_index} processes directories {start_idx+1} to {end_idx}")
    else:
        # Use original even/odd allocation
        subdirs = []
        for i, subdir in enumerate(unprocessed_subdirs):
            if i % shard_count == shard_index:
                subdirs.append(subdir)
        logging.info(f"Using even/odd allocation: GPU {shard_index} processes {len(subdirs)}/{len(unprocessed_subdirs)} directories")

    # Preload model
    model_manager = UVDModelManager()
    preprocessor_key = model_manager.get_model(preprocessor)
    if not preprocessor_key:
        logging.error(f"Failed to initialize preprocessor {preprocessor}, cannot continue processing")
        return
        
    # Initialize GPU memory monitor
    memory_monitor = GPUMemoryMonitor()

    # Process each subdirectory
    for subdir_path in subdirs:
        logging.info(f"\n=== Processing directory: {subdir_path} ===")

        # Create output directory
        output_dir = os.path.join(subdir_path, "imp")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directory for segmented videos
        segments_dir = os.path.join(subdir_path, "segments")
        os.makedirs(segments_dir, exist_ok=True)

        video_files_to_process = []
        all_keyframes_data = []  # For collecting all keyframe info

        # 1. Find existing H.264 files
        h264_files = glob.glob(os.path.join(subdir_path, '*_h264.mp4'))

        if h264_files:
            logging.info(f"Found {len(h264_files)} existing H.264 video files.")
            video_files_to_process = h264_files
        else:
            logging.info(f"No H.264 video files found, searching for original MP4 files to convert...")
            # 2. Find original MP4 files (exclude _h264 suffix)
            original_mp4_files = [
                f for f in glob.glob(os.path.join(subdir_path, '*.mp4'))
                if '_h264.mp4' not in f
            ]

            if not original_mp4_files:
                logging.info(f"No original MP4 files found for conversion in {subdir_path}, skipping")
                continue

            logging.info(f"Found {len(original_mp4_files)} original MP4 files, starting conversion...")
            converted_files = []
            for original_path in original_mp4_files:
                converted_path = convert_to_h264(original_path)
                if converted_path:
                    converted_files.append(converted_path)
                    converted_count += 1

            if not converted_files:
                logging.warning(f"Failed to convert any original MP4 files, skipping directory {subdir_path}")
                continue

            video_files_to_process = converted_files

        if not video_files_to_process:
            logging.info(f"No video files to process in {subdir_path}, skipping")
            continue

        logging.info(f"Will process {len(video_files_to_process)} video files...")
        
        # For each video, segment first, then process
        for video_path in video_files_to_process:
            # Check if metadata file exists
            metadata_file = os.path.join(output_dir, "keyframes_metadata.json")
            if os.path.exists(metadata_file):
                logging.info(f"Directory already has keyframe metadata, skipping: {subdir_path}")
                skipped_count += 1
                continue
                
            # Segment video into 1-minute segments first
            logging.info(f"Starting segmentation of video: {os.path.basename(video_path)}")
            segment_files = segment_video(video_path, segments_dir)
            
            if not segment_files:
                logging.warning(f"Video segmentation failed, trying to process original video directly: {os.path.basename(video_path)}")
                # If segmentation fails, process original video directly
                result, keyframes_info = extract_keyframes(
                    video_path, 
                    preprocessor_name=preprocessor, 
                    output_dir=output_dir
                )
                if result:
                    all_keyframes_data.extend(keyframes_info or [])
                    processed_count += 1
                else:
                    error_count += 1
                continue
                
            # If only one segment (original video less than 1 minute) and it's the original video, process directly
            if len(segment_files) == 1 and segment_files[0] == video_path:
                result, keyframes_info = extract_keyframes(video_path, preprocessor_name=preprocessor, output_dir=output_dir)
                if result:
                    all_keyframes_data.extend(keyframes_info or [])
                    processed_count += 1
                else:
                    error_count += 1
                continue
                
            # Process each video segment - use parallel processing here
            logging.info(f"Video {os.path.basename(video_path)} segmented into {len(segment_files)} segments, starting parallel processing of segments")
            segmented_count += 1
            
            # Use new parallel processing function
            segment_success, segment_error, segment_keyframes_info = process_video_segments(
                segment_files, preprocessor, output_dir, max_workers=max_parallel
            )
            
            # Add all successfully processed segment keyframe info
            all_keyframes_data.extend(segment_keyframes_info)
            
            logging.info(f"Segments of video {os.path.basename(video_path)} processed: success {segment_success}/{len(segment_files)}")
            
            if segment_success > 0:
                processed_count += 1
            else:
                error_count += 1
        
        # Save metadata for all keyframes in this directory to JSON file
        if all_keyframes_data:
            save_keyframes_metadata(output_dir, all_keyframes_data)
            
        # Clean up memory after processing each directory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # After processing, release all model resources
    model_manager.release_model()
    
    logging.info("\n=== Processing complete ===")
    logging.info(f"Total videos converted: {converted_count}")
    logging.info(f"Total videos segmented: {segmented_count}")
    logging.info(f"Total successfully processed (keyframes extracted): {processed_count} videos")
    logging.info(f"Skipped already processed: {skipped_count} videos")
    logging.info(f"Failed to process (keyframe extraction): {error_count} videos")

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="Process all videos in a directory and extract keyframes, automatically convert non-H.264 videos")

    parser.add_argument("--root", type=str,
                       default="",
                       help="Root directory path containing video subdirectories")

    parser.add_argument("--preprocessor", type=str, default="CLIP", # Default changed to CLIP
                       choices=["VIP", "R3M", "LIV", "CLIP", "DINO-v2", "VC-1", "ResNet"],
                       help="Select UVD preprocessor")

    parser.add_argument("--shard_index", type=int, default=0, help="Current shard index (starting from 0)")
    parser.add_argument("--shard_count", type=int, default=1, help="Total number of shards")
    parser.add_argument("--sequential", action="store_true", help="Use sequential allocation mode instead of even/odd allocation")
    parser.add_argument("--start_index", type=int, default=None, help="Manually specify start index")
    parser.add_argument("--end_index", type=int, default=None, help="Manually specify end index")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--max_parallel", type=int, default=8, help="Maximum number of concurrent processes")
    parser.add_argument("--memory_threshold", type=int, default=4000, help="GPU memory threshold (MB), pause processing if below this value")

    args = parser.parse_args()

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("ffmpeg is available, starting processing...")
        
        # Set GPU memory threshold
        GPUMemoryMonitor.threshold_mb = args.memory_threshold
        
        process_directory(
            args.root, 
            args.preprocessor, 
            args.shard_index, 
            args.shard_count,
            args.start_index,
            args.end_index,
            args.sequential,
            args.max_parallel
        )
    except FileNotFoundError:
        logging.error("Error: ffmpeg command not found. Please ensure ffmpeg is installed and added to system PATH.")
    except subprocess.CalledProcessError:
        logging.error("Error: failed to execute ffmpeg -version, please check ffmpeg installation.")
