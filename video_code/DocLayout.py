import os
import cv2
import torch
import argparse
from doclayout_yolo import YOLOv10
import glob
import shutil
import concurrent.futures
import multiprocessing
import time
from tqdm import tqdm
import threading
import logging

# Added: Function to check if subdirectory has been processed
def is_dir_processed(subdir_path):
    """
    Check if the subdirectory has already been processed
    
    Args:
        subdir_path: Path to the subdirectory
    
    Returns:
        bool: True means processed, False means not processed
    """
    tag1_dir = os.path.join(subdir_path, "tag1")
    tag_dir = os.path.join(subdir_path, "tag")
    
    # Check if tag1 directory exists and is not empty
    if not os.path.exists(tag1_dir):
        return False
    
    # Check if there are images in tag1 directory
    image_files = glob.glob(os.path.join(tag1_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(tag1_dir, "*.jpeg")))
    image_files.extend(glob.glob(os.path.join(tag1_dir, "*.png")))
    
    if not image_files:
        return False
    
    # Check if there are corresponding result images in tag directory
    if os.path.exists(tag_dir):
        result_files = glob.glob(os.path.join(tag_dir, "*_res.jpg"))
        if result_files:
            return True
    
    return len(image_files) > 0

# Function for parallel processing
def process_directory(args):
    """
    Function to process a single directory, used for parallel execution
    
    Args:
        args: Dictionary containing all necessary parameters
    
    Returns:
        Dictionary containing processing result information
    """
    subdir_path = args['subdir_path']
    model = args['model']
    imgsz = args['imgsz']
    conf = args['conf']
    line_width = args['line_width']
    font_size = args['font_size']
    device = args['device']
    clear_tag1 = args['clear_tag1']
    
    subdir = os.path.basename(subdir_path)
    pic_dir = os.path.join(subdir_path, "pic")
    tag_dir = os.path.join(subdir_path, "tag")
    tag1_dir = os.path.join(subdir_path, "tag1")
    
    result = {
        'subdir': subdir,
        'success': False,
        'processed_images': 0,
        'error_images': 0,
        'error_message': None
    }
    
    try:
        # Check if pic directory exists and is not empty
        if not os.path.exists(pic_dir):
            result['error_message'] = f"{pic_dir} directory does not exist"
            return result
            
        image_paths = glob.glob(os.path.join(pic_dir, "*.jpg"))
        if not image_paths:
            result['error_message'] = f"No jpg images in {pic_dir}"
            return result

        # 1. Clear tag1 folder if needed
        if clear_tag1:
            if os.path.exists(tag1_dir):
                shutil.rmtree(tag1_dir)
            if os.path.exists(tag_dir):
                shutil.rmtree(tag_dir)
                
        os.makedirs(tag1_dir, exist_ok=True)
        os.makedirs(tag_dir, exist_ok=True)
            
        image_paths = sorted(image_paths)
        
        for image_path in image_paths:
            try:
                det_res = model.predict(
                    image_path,
                    imgsz=imgsz,
                    conf=conf,
                    device=device,
                )

                # Save visualization result
                annotated_frame = det_res[0].plot(pil=True, line_width=line_width, font_size=font_size)
                output_path = os.path.join(tag_dir, os.path.basename(image_path).replace(".jpg", "_res.jpg"))
                cv2.imwrite(output_path, annotated_frame)

                img = cv2.imread(image_path)
                img_h, img_w = img.shape[:2]
                boxes = det_res[0].boxes

                # Only keep boxes with class "figure" (ignore abandon etc.)
                max_area = 0
                max_crop = None
                max_name = None

                for idx, cls in enumerate(boxes.cls):
                    # Check if class name is "figure"
                    if hasattr(boxes, 'names'):
                        name = boxes.names[int(cls)]
                        if name != "figure":
                            continue
                    else:
                        # If no class names, assume figure class ID is 2 (modify according to actual situation)
                        if int(cls) != 3:
                            continue
                    xyxy = boxes.xyxy[idx].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        max_crop = img[y1:y2, x1:x2]
                        max_name = os.path.basename(image_path).replace(".jpg", "_figure.jpg")

                # Only output the largest figure, and area must be greater than 30% of full image, otherwise output original image
                if max_crop is not None and max_area > 0.3 * img_h * img_w:
                    crop_path = os.path.join(tag1_dir, max_name)
                    cv2.imwrite(crop_path, max_crop)
                else:
                    # No qualified figure, directly output original image
                    crop_path = os.path.join(tag1_dir, os.path.basename(image_path))
                    cv2.imwrite(crop_path, img)

                result['processed_images'] += 1

            except Exception as e:
                result['error_images'] += 1
                result['error_message'] = f"Failed to process image {os.path.basename(image_path)}: {str(e)}"
        
        result['success'] = True
        return result

    except Exception as e:
        result['error_message'] = f"Directory processing exception: {str(e)}"
        return result

# Define callback function at outer level
def process_result(future, results_dict, pbar):
    """Task completion callback function
    
    Args:
        future: Future object
        results_dict: Dictionary containing result statistics
        pbar: tqdm progress bar object
    """
    result = future.result()
    with results_dict['lock']:
        if result['success']:
            results_dict['processed_dirs'] += 1
            results_dict['total_processed_images'] += result['processed_images']
            if result['error_images'] > 0:
                logging.warning(f"⚠️ {result['subdir']} - Processed successfully but {result['error_images']} images failed")
            else:
                logging.debug(f"✅ {result['subdir']} - Successfully processed {result['processed_images']} images")
        else:
            results_dict['failed_dirs'] += 1
            logging.error(f"❌ {result['subdir']} - Processing failed: {result['error_message']}")

        results_dict['total_error_images'] += result['error_images']
        pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, required=True, type=str,
                        help='Model file path (.pt)')
    parser.add_argument('--root-dir', default=None, required=True, type=str,
                        help='Root directory, each subdirectory should have pic/ and tag/ folders')
    parser.add_argument('--imgsz', default=1024, required=False, type=int)
    parser.add_argument('--line-width', default=5, required=False, type=int)
    parser.add_argument('--font-size', default=20, required=False, type=int)
    parser.add_argument('--conf', default=0.2, required=False, type=float)
    parser.add_argument('--clear-tag1', action='store_true', help='Clear all tag1 folders before processing')
    parser.add_argument('--no-skip', action='store_true', help='Do not skip already processed subdirectories (default is to skip)')
    # Added parallel processing related parameters
    parser.add_argument('--workers', type=int, default=0, 
                        help='Number of worker processes for parallel processing, 0 means auto select')
    parser.add_argument('--quiet', action='store_true', help='Reduce output information')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    model_path = args.model
    if os.path.isdir(model_path):
        pt_files = glob.glob(os.path.join(model_path, "*.pt"))
        if pt_files:
            model_path = pt_files[0]
            logging.info(f"Model file found in directory: {model_path}")
        else:
            logging.error(f"No .pt model file found in directory {model_path}")
            raise FileNotFoundError(f"No .pt model file found in directory {model_path}")
    
    if not os.path.exists(model_path):
        logging.error(f"Model file does not exist: {model_path}")
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    logging.info(f"Loading local model: {model_path}")
    try:
        model = YOLOv10(model_path)
        logging.info("Model loaded successfully")
        
        # Get all subdirectories
        subdirs = [d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))]
        subdirs.sort()  # Sort alphabetically
        
        total_dirs = len(subdirs)
        logging.info(f"Found {total_dirs} subdirectories")
        
        # Filter directories to process
        dirs_to_process = []
        for subdir in subdirs:
            subdir_path = os.path.join(args.root_dir, subdir)
            # Check if already processed
            if not args.no_skip and is_dir_processed(subdir_path):
                continue
            dirs_to_process.append(subdir_path)
        
        skipped_dirs = total_dirs - len(dirs_to_process)
        logging.info(f"Will process {len(dirs_to_process)} directories, skipped {skipped_dirs} already processed directories")
        
        if not dirs_to_process:
            logging.info("No directories need processing, program ends.")
            exit(0)
        
        # Determine number of worker processes
        num_workers = args.workers if args.workers > 0 else min(max(1, multiprocessing.cpu_count() - 1), len(dirs_to_process))
        logging.info(f"Number of parallel processes: {num_workers}")
        
        # Result statistics dictionary
        results_dict = {
            'processed_dirs': 0,
            'failed_dirs': 0,
            'total_processed_images': 0,
            'total_error_images': 0,
            'lock': threading.Lock()
        }
        
        # Progress display related
        verbose = not args.quiet
        pbar = tqdm(total=len(dirs_to_process), disable=not verbose, desc="Processing progress")
        
        # Prepare task parameters
        task_args = []
        for subdir_path in dirs_to_process:
            task_args.append({
                'subdir_path': subdir_path,
                'model': model,
                'imgsz': args.imgsz,
                'conf': args.conf,
                'line_width': args.line_width,
                'font_size': args.font_size,
                'device': device,
                'clear_tag1': args.clear_tag1
            })

        # Use thread pool instead of process pool, because model is loaded in main process memory, using process pool would reload model repeatedly
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for task_arg in task_args:
                future = executor.submit(process_directory, task_arg)
                future.add_done_callback(lambda f: process_result(f, results_dict, pbar))
                futures.append(future)

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
            
        # Close progress bar
        pbar.close()
        
        # Output processing statistics
        elapsed_time = time.time() - start_time
        logging.info("====== Processing completed ======")
        logging.info(f"Total time: {elapsed_time:.2f} seconds")
        logging.info(f"Successfully processed: {results_dict['processed_dirs']}/{total_dirs} directories, {results_dict['total_processed_images']} images")
        
        if results_dict['failed_dirs'] > 0 or results_dict['total_error_images'] > 0:
            logging.warning(f"Processing failed: {results_dict['failed_dirs']} directories, {results_dict['total_error_images']} images")

    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error("Hint: Please make sure a valid YOLOv10 model file (.pt format) is provided")