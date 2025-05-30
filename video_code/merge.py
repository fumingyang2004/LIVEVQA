import os
import shutil
import imagehash
from PIL import Image
import argparse
from pathlib import Path
import glob
import numpy as np
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
import threading

def calculate_image_clarity(img_path):
    """
    Calculate image clarity score
    Use Laplacian operator to evaluate edge sharpness of the image
    
    Args:
        img_path: Image path
    
    Returns:
        clarity_score: Clarity score, higher means clearer image
    """
    try:
        # Read image and convert to grayscale
        img = cv2.imread(img_path)
        if img is None:
            return 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use Laplacian operator to calculate image gradient
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance of gradient, higher variance means clearer image
        clarity_score = np.var(laplacian)
        
        return clarity_score
    except Exception as e:
        print(f"❌ Failed to calculate image clarity {img_path}: {e}")
        return 0

def process_folder(folder_path, threshold=5, copy_mode=True, verbose=True, clean_existing=True):
    """
    Deduplicate images in the 'imp' subdirectory of the specified folder using perceptual hash,
    prioritizing images with higher clarity.
    
    Args:
        folder_path: Parent folder path
        threshold: Hamming distance threshold for hash, below which images are considered duplicates
        copy_mode: True to copy files, False to move files
        verbose: Whether to print detailed information
        clean_existing: Whether to clear existing images in the 'pic' directory
    
    Returns:
        Number of images kept
    """
    # Check if 'imp' directory exists
    imp_folder = os.path.join(folder_path, "imp")
    if not os.path.isdir(imp_folder):
        if verbose:
            print(f"Skipped: 'imp' directory not found in {folder_path}")
        return 0
    
    # Create output folder
    output_folder = os.path.join(folder_path, "pic")
    
    # If specified to clean existing images and 'pic' directory exists, delete all images inside
    if clean_existing and os.path.exists(output_folder):
        if verbose:
            print(f"Cleaning existing 'pic' directory: {output_folder}")
        
        # Delete all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files_to_delete = []
        for ext in image_extensions:
            files_to_delete.extend(glob.glob(os.path.join(output_folder, ext)))
            files_to_delete.extend(glob.glob(os.path.join(output_folder, ext.upper())))
        
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                if verbose:
                    print(f"Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Failed to delete file {file_path}: {e}")
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in 'imp' folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(imp_folder, ext)))
        image_files.extend(glob.glob(os.path.join(imp_folder, ext.upper())))
    
    if not image_files:
        if verbose:
            print(f"No images found in 'imp' directory: {imp_folder}")
        return 0  # Ensure integer return
    
    # Calculate perceptual hash and clarity for each image
    image_hashes = []
    image_clarity_scores = []
    valid_images = []
    
    if verbose:
        print("Calculating image hashes and clarity...")
    
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            # Use perceptual hash
            img_hash = imagehash.phash(img)
            image_hashes.append(img_hash)
            
            # Calculate image clarity score
            clarity_score = calculate_image_clarity(img_path)
            image_clarity_scores.append(clarity_score)
            
            valid_images.append(img_path)
        except Exception as e:
            print(f"❌ Unable to process image {img_path}: {e}")
    
    if not valid_images:
        if verbose:
            print("No valid images to process")
        return 0  # Ensure integer return
    
    # Improved deduplication algorithm, prioritize images with higher clarity
    kept_indices = []
    duplicate_groups = []  # Store groups of duplicate images
    
    # Treat all images as ungrouped initially
    ungrouped = list(range(len(valid_images)))
    
    # Generate groups of duplicate images
    while ungrouped:
        current_idx = ungrouped[0]
        ungrouped.remove(current_idx)
        
        # Current duplicate group
        current_group = [current_idx]
        
        # Find similar images in ungrouped list
        i = 0
        while i < len(ungrouped):
            idx = ungrouped[i]
            if image_hashes[current_idx] - image_hashes[idx] < threshold:
                current_group.append(idx)
                ungrouped.remove(idx)
            else:
                i += 1
        
        # If only one image, keep directly; otherwise add to duplicate groups
        if len(current_group) == 1:
            kept_indices.append(current_group[0])
        else:
            duplicate_groups.append(current_group)
    
    # From each duplicate group, select the image with the highest clarity
    for group in duplicate_groups:
        # Find image with highest clarity in the group
        best_idx = group[0]
        best_clarity = image_clarity_scores[best_idx]
        
        for idx in group[1:]:
            if image_clarity_scores[idx] > best_clarity:
                best_idx = idx
                best_clarity = image_clarity_scores[idx]
        
        kept_indices.append(best_idx)
        
        if verbose:
            group_files = [os.path.basename(valid_images[i]) for i in group]
            best_file = os.path.basename(valid_images[best_idx])
            print(f"In duplicate group {group_files}, selected clearest image: {best_file} (Clarity: {best_clarity:.2f})")
    
    # Copy or move kept images to output directory
    for i in kept_indices:
        src_path = valid_images[i]
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_folder, filename)
        
        try:
            if copy_mode:
                shutil.copy2(src_path, dst_path)
                if verbose:
                    print(f"Copied: {filename} (Clarity: {image_clarity_scores[i]:.2f})")
            else:
                shutil.move(src_path, dst_path)
                if verbose:
                    print(f"Moved: {filename} (Clarity: {image_clarity_scores[i]:.2f})")
        except Exception as e:
            print(f"❌ Failed to process file {src_path}: {e}")
    
    if verbose:
        print(f"Kept {len(kept_indices)} non-duplicate images (prioritizing clarity) out of {len(valid_images)} images")
    
    return len(kept_indices)  # Ensure integer return

def check_folder_processed(folder_path):
    """
    Check if the specified folder has been processed (has 'pic' directory with images)
    
    Args:
        folder_path: Folder path
        
    Returns:
        bool: True if processed, False otherwise
    """
    pic_folder = os.path.join(folder_path, "pic")
    
    # Check if 'pic' folder exists
    if not os.path.isdir(pic_folder):
        return False
    
    # Check if 'pic' folder contains images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(pic_folder, ext)))
        image_files.extend(glob.glob(os.path.join(pic_folder, ext.upper())))
    
    # Consider processed if images exist
    return len(image_files) > 0

def process_folder_wrapper(args):
    """
    Wrapper for process_folder function, used in process pool
    
    Args:
        args: Tuple containing all parameters
        
    Returns:
        Result tuple (folder_name, kept_images, success)
    """
    folder, threshold, copy_mode, verbose, clean_existing = args
    folder_name = os.path.basename(folder)
    
    try:
        kept_images = process_folder(folder, threshold, copy_mode, verbose, clean_existing)
        return (folder_name, kept_images, True)
    except Exception as e:
        if verbose:
            print(f"❌ Error processing folder {folder_name}: {e}")
        return (folder_name, 0, False)

def main():
    parser = argparse.ArgumentParser(description="Deduplicate images in the 'imp' directory of each subfolder under the specified root directory")
    parser.add_argument("--root_dir", type=str, 
                        default="",
                        help="Root directory path")
    parser.add_argument("--threshold", type=int, default=10, 
                        help="Hamming distance threshold for hash, below which images are considered duplicates")
    parser.add_argument("--move", action="store_true", 
                        help="Move files instead of copying")
    parser.add_argument("--clean", action="store_true",
                        help="Clear existing images in 'pic' directory before deduplication")
    parser.add_argument("--no_skip", action="store_true",
                        help="Do not skip already processed subdirectories (default is to skip)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel worker processes, 0 means use CPU core count")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    args = parser.parse_args()
    
    # Default is to skip processed directories, disable with --no_skip
    skip_processed = not args.no_skip
    verbose = not args.quiet
    
    if verbose:
        print(f"Starting to process directory: {args.root_dir}")
        print(f"Processing images in 'imp' folder of each subdirectory, results saved in 'pic' folder")
        print(f"Similarity threshold: {args.threshold} (lower means stricter duplicate detection)")
        print(f"File handling mode: {'Move' if args.move else 'Copy'}")
        print(f"Clear existing 'pic' directory: {'Yes' if args.clean else 'No'}")
        print(f"Skip already processed subdirectories: {'No' if args.no_skip else 'Yes'}")
    
    # Determine number of worker processes
    num_workers = args.workers if args.workers > 0 else max(1, multiprocessing.cpu_count() - 1)
    if verbose:
        print(f"Number of parallel worker processes: {num_workers}")
    
    # Get all subfolders under root directory
    subfolders = [f.path for f in os.scandir(args.root_dir) if f.is_dir()]
    total_folders = len(subfolders)
    
    if total_folders == 0:
        print(f"No subdirectories found under root directory {args.root_dir}, exiting.")
        return
    
    # Filter folders to process (exclude already processed)
    folders_to_process = []
    skipped_folders = 0
    
    for folder in subfolders:
        # Check if already processed and skipping enabled
        if skip_processed and check_folder_processed(folder):
            skipped_folders += 1
            if verbose:
                print(f"Skipped already processed folder: {os.path.basename(folder)}")
            continue
        
        folders_to_process.append(folder)
    
    if not folders_to_process:
        print(f"All {total_folders} subdirectories have been processed, no folders to process.")
        return
        
    if verbose:
        print(f"\nAbout to process {len(folders_to_process)}/{total_folders} folders...")
        
    # Prepare process pool arguments
    process_args = [
        (folder, args.threshold, not args.move, False, args.clean) 
        for folder in folders_to_process
    ]
    
    # Create shared counters and lock for real-time statistics
    manager = multiprocessing.Manager()
    results_dict = manager.dict()
    results_dict['processed_folders'] = 0
    results_dict['total_kept_images'] = 0
    results_dict['error_folders'] = 0
    
    # Create progress bar
    pbar = tqdm(total=len(folders_to_process), disable=not verbose,
                desc="Processing Progress", unit="folders")
    
    # Define callback function
    def update_progress(future):
        folder_name, kept_images, success = future.result()
        
        with results_lock:
            if success:
                if kept_images > 0:
                    results_dict['processed_folders'] += 1
                    results_dict['total_kept_images'] += kept_images
                    if verbose:
                        tqdm.write(f"✓ Completed {folder_name}: Kept {kept_images} images")
                else:
                    tqdm.write(f"! Processed {folder_name}: No valid images")
            else:
                results_dict['error_folders'] += 1
                tqdm.write(f"❌ Failed to process: {folder_name}")
            pbar.update(1)
    
    # Use process pool for parallel processing
    start_time = time.time()
    results_lock = threading.Lock()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = []
        for args_set in process_args:
            future = executor.submit(process_folder_wrapper, args_set)
            future.add_done_callback(update_progress)
            futures.append(future)
        
        # Wait for all tasks to complete
        for future in futures:
            try:
                future.result()
            except Exception:
                pass
    
    # Close progress bar
    pbar.close()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Aggregate results
    processed_folders = results_dict['processed_folders']
    total_kept_images = results_dict['total_kept_images']
    error_folders = results_dict['error_folders']
    
    print(f"\nFinished processing {len(folders_to_process)}/{total_folders} subdirectories in {elapsed_time:.2f} seconds")
    print(f"Skipped already processed: {skipped_folders} subdirectories")
    print(f"Successfully processed: {processed_folders} subdirectories")
    print(f"Failed to process: {error_folders} subdirectories")
    print(f"Total kept {total_kept_images} non-duplicate images saved in respective 'pic' folders")
    
    if num_workers > 1:
        print(f"Parallel efficiency: {num_workers} processes, average time per folder: {elapsed_time/max(1, len(folders_to_process)):.2f} seconds")

if __name__ == "__main__":
    # Prevent multiprocessing recursion issues on Windows
    multiprocessing.freeze_support()
    main()