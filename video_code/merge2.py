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
    Calculate the clarity score of an image using the Laplacian operator to evaluate edge sharpness.
    
    Args:
        img_path: image path
    
    Returns:
        clarity_score: clarity score, the higher the clearer the image
    """
    try:
        # Read the image and convert to grayscale
        img = cv2.imread(img_path)
        if img is None:
            return 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use Laplacian operator to calculate image gradient
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate the variance of the gradient; larger variance means clearer image
        clarity_score = np.var(laplacian)
        
        return clarity_score
    except Exception as e:
        print(f"❌ Failed to calculate image clarity {img_path}: {e}")
        return 0

def process_folder(folder_path, threshold=5, copy_mode=True, verbose=True, clean_existing=True):
    """
    Perform perceptual hash-based deduplication on images in the 'tag1' subfolder, preferring the clearest image in each duplicate group.
    
    Args:
        folder_path: parent folder path
        threshold: hamming distance threshold for hash values, less than this is considered duplicate
        copy_mode: True to copy files, False to move files
        verbose: whether to print detailed info
        clean_existing: whether to clear existing images in the pic directory
    
    Returns:
        number of kept images
    """
    # Check if tag1 directory exists
    imp_folder = os.path.join(folder_path, "tag1")
    if not os.path.isdir(imp_folder):
        if verbose:
            print(f"Skipped: tag1 directory not found in {folder_path}")
        return 0
    
    # Create output folder
    output_folder = os.path.join(folder_path, "tag2")
    
    # If specified to clean existing images and pic directory exists, delete all images in it
    if clean_existing and os.path.exists(output_folder):
        if verbose:
            print(f"Cleaning existing pic directory: {output_folder}")
        
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
    
    # Get all image files in tag1 folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(imp_folder, ext)))
        image_files.extend(glob.glob(os.path.join(imp_folder, ext.upper())))
    
    if not image_files:
        if verbose:
            print(f"No images found in imp directory: {imp_folder}")
        return 0  # Ensure an integer is returned
    
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
        return 0  # Ensure an integer is returned
    
    # Improved deduplication algorithm, prefer to keep clearer images
    kept_indices = []
    duplicate_groups = []  # Store duplicate image groups
    
    # Initially treat all images as ungrouped
    ungrouped = list(range(len(valid_images)))
    
    # Generate duplicate image groups
    while ungrouped:
        current_idx = ungrouped[0]
        ungrouped.remove(current_idx)
        
        # Current duplicate group
        current_group = [current_idx]
        
        # Find similar images in ungrouped images
        i = 0
        while i < len(ungrouped):
            idx = ungrouped[i]
            if image_hashes[current_idx] - image_hashes[idx] < threshold:
                current_group.append(idx)
                ungrouped.remove(idx)
            else:
                i += 1
        
        # If only one image, keep directly; else add to duplicate group
        if len(current_group) == 1:
            kept_indices.append(current_group[0])
        else:
            duplicate_groups.append(current_group)
    
    # From each duplicate group, select the clearest image
    for group in duplicate_groups:
        # Find the image with highest clarity in the group
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
            print(f"In duplicate group {group_files} selected the clearest image: {best_file} (clarity: {best_clarity:.2f})")
    
    # Copy or move kept images to output directory
    for i in kept_indices:
        src_path = valid_images[i]
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_folder, filename)
        
        try:
            if copy_mode:
                shutil.copy2(src_path, dst_path)
                if verbose:
                    print(f"Copied: {filename} (clarity: {image_clarity_scores[i]:.2f})")
            else:
                shutil.move(src_path, dst_path)
                if verbose:
                    print(f"Moved: {filename} (clarity: {image_clarity_scores[i]:.2f})")
        except Exception as e:
            print(f"❌ Failed to process file {src_path}: {e}")
    
    if verbose:
        print(f"Kept {len(kept_indices)} non-duplicate images out of {len(valid_images)} (kept clearer images)")
    
    return len(kept_indices)  # Ensure an integer is returned

def check_folder_processed(folder_path):
    """
    Check whether the folder has been processed (i.e., has a 'tag2' folder containing images).
    
    Args:
        folder_path: folder path
        
    Returns:
        bool: True if processed, False otherwise
    """
    pic_folder = os.path.join(folder_path, "tag2")
    
    # Check if pic folder exists
    if not os.path.isdir(pic_folder):
        return False
    
    # Check if there are images in the pic folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(pic_folder, ext)))
        image_files.extend(glob.glob(os.path.join(pic_folder, ext.upper())))
    
    # Consider processed if images exist
    return len(image_files) > 0

def process_folder_wrapper(args):
    """
    Wrapper for process_folder function for multiprocessing use.
    
    Args:
        args: tuple containing all arguments
        
    Returns:
        result tuple (folder_name, kept_images, success)
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
    parser = argparse.ArgumentParser(description="Deduplicate images in the imp (tag1) folder of each subdirectory under the specified directory.")
    parser.add_argument("--root_dir", type=str, 
                        default="",
                        help="Root directory path")
    parser.add_argument("--threshold", type=int, default=25, 
                        help="Hamming distance threshold for hash values; less than this is considered duplicate")
    parser.add_argument("--move", action="store_true", 
                        help="Move files instead of copying")
    parser.add_argument("--clean", action="store_true",
                        help="Clear existing images in the pic (tag2) directory before deduplication")
    parser.add_argument("--no_skip", action="store_true",
                        help="Do not skip already processed subdirectories (by default, processed subdirectories are skipped)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel worker processes; 0 means use CPU core count")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    args = parser.parse_args()
    
    # By default, skip processed directories; use --no_skip to disable this
    skip_processed = not args.no_skip
    verbose = not args.quiet
    
    if verbose:
        print(f"Start processing directory: {args.root_dir}")
        print(f"Will process images in the tag1 folder of each subdirectory, results saved to tag2 folder")
        print(f"Similarity threshold: {args.threshold} (lower value means stricter duplicate detection)")
        print(f"File operation mode: {'Move' if args.move else 'Copy'}")
        print(f"Clean existing pic directory: {'Yes' if args.clean else 'No'}")
        print(f"Skip already processed subdirectories: {'No' if args.no_skip else 'Yes'}")
    
    # Determine number of worker processes
    num_workers = args.workers if args.workers > 0 else max(1, multiprocessing.cpu_count() - 1)
    if verbose:
        print(f"Number of parallel worker processes: {num_workers}")
    
    # Get all subfolders in root directory
    subfolders = [f.path for f in os.scandir(args.root_dir) if f.is_dir()]
    total_folders = len(subfolders)
    
    if total_folders == 0:
        print(f"No subfolders found under root directory {args.root_dir}, nothing to process.")
        return
    
    # Filter folders to process (exclude already processed)
    folders_to_process = []
    skipped_folders = 0
    
    for folder in subfolders:
        # Check if already processed and skip if enabled
        if skip_processed and check_folder_processed(folder):
            skipped_folders += 1
            if verbose:
                print(f"Skipped already processed folder: {os.path.basename(folder)}")
            continue
        
        folders_to_process.append(folder)
    
    if not folders_to_process:
        print(f"All {total_folders} subdirectories have been processed, nothing to do.")
        return
        
    if verbose:
        print(f"\nAbout to process {len(folders_to_process)}/{total_folders} folders...")
        
    # Prepare process pool arguments
    process_args = [
        (folder, args.threshold, not args.move, False, args.clean) 
        for folder in folders_to_process
    ]
    
    # Create shared counter and lock for real-time stats
    manager = multiprocessing.Manager()
    results_dict = manager.dict()
    results_dict['processed_folders'] = 0
    results_dict['total_kept_images'] = 0
    results_dict['error_folders'] = 0
    
    # Progress bar
    pbar = tqdm(total=len(folders_to_process), disable=not verbose,
                desc="Processing", unit="folder")
    
    # Callback function
    def update_progress(future):
        folder_name, kept_images, success = future.result()
        
        with results_lock:
            if success:
                if kept_images > 0:
                    results_dict['processed_folders'] += 1
                    results_dict['total_kept_images'] += kept_images
                    if verbose:
                        tqdm.write(f"✓ Finished {folder_name}: kept {kept_images} images")
                else:
                    tqdm.write(f"! Processed {folder_name}: no valid images")
            else:
                results_dict['error_folders'] += 1
                tqdm.write(f"❌ Failed to process: {folder_name}")
            pbar.update(1)
    
    # Parallel processing with process pool
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
    
    # Gather results
    processed_folders = results_dict['processed_folders']
    total_kept_images = results_dict['total_kept_images']
    error_folders = results_dict['error_folders']
    
    print(f"\nFinished processing {len(folders_to_process)}/{total_folders} subdirectories in {elapsed_time:.2f} seconds")
    print(f"Skipped already processed: {skipped_folders} subdirectories")
    print(f"Successfully processed: {processed_folders} subdirectories")
    print(f"Failed to process: {error_folders} subdirectories")
    print(f"Kept a total of {total_kept_images} non-duplicate images, saved to each folder's tag2 directory")
    
    if num_workers > 1:
        print(f"Parallel efficiency: {num_workers} processes, average time per folder: {elapsed_time/max(1, len(folders_to_process)):.2f} seconds")

if __name__ == "__main__":
    # Prevent multiprocessing recursion issues on Windows
    multiprocessing.freeze_support()
    main()