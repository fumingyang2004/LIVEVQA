#!/usr/bin/env python3
import argparse
import json
import os
import sys
import logging
import threading
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Get project root directory (parent of Evaluation)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_qa_handler import get_qa_answer_local, judge_answer_local
from unified_client import is_model_available
from local_config import add_local_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

output_file_lock = threading.Lock()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Enhanced VQA Benchmark - Supports API and Local Models")
    
    # Basic arguments
    parser.add_argument("--benchmark", "-b", required=True, choices=["News", "Paper", "Video"])
    parser.add_argument("--qa-model", "-q", required=True, help="QA model name")
    parser.add_argument("--judge-model", "-j", default="gpt-4o-mini", help="Judge model name")
    
    # Provider configuration
    parser.add_argument("--qa-provider", choices=["openai", "openrouter", "local"], default="openai")
    parser.add_argument("--judge-provider", choices=["openai", "openrouter", "local"], default="openai")
    parser.add_argument("--local-base-url", help="Base URL for local models")
    parser.add_argument("--judge-base-url", help="Base URL for judge model if different")
    
    # Performance settings
    parser.add_argument("--workers", "-w", type=int, default=5, help="Max worker threads")
    parser.add_argument("--api-delay", type=float, default=1.0, help="Delay between API calls")
    
    # API configuration
    parser.add_argument("--api-key", help="API key for OpenAI/OpenRouter")
    parser.add_argument("--http-referer", default="http://localhost/vqa-benchmark")
    parser.add_argument("--site-title", default="VQA Benchmark Tool")
    
    # Path configuration
    parser.add_argument("--project-root", help="Project root directory (auto-detected if not specified)")
    
    return parser.parse_args()

def setup_environment(args):
    """Setup environment variables based on arguments"""
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Add local models if specified
    if args.qa_provider == "local" and args.local_base_url:
        add_local_model(args.qa_model, args.local_base_url)
    
    if args.judge_provider == "local" and args.judge_base_url:
        add_local_model(args.judge_model, args.judge_base_url)
    elif args.judge_provider == "local" and args.local_base_url:
        add_local_model(args.judge_model, args.local_base_url)

def get_project_paths(args):
    """Get project paths with proper structure validation"""
    if args.project_root:
        base_dir = Path(args.project_root)
    else:
        # Auto-detect: parent directory of Evaluation folder
        base_dir = Path(__file__).parent.parent
    
    # Validate project structure
    if not base_dir.exists():
        raise FileNotFoundError(f"Project root directory not found: {base_dir}")
    
    # Check if benchmark directories exist
    benchmark_dir = base_dir / args.benchmark
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    
    benchmark_file = benchmark_dir / "benchmark.json"
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
    
    # Image directory mapping
    image_dirs = {
        "News": "News_imgs",
        "Paper": "Paper_images", 
        "Video": "Video_images"
    }
    
    image_base_dir = benchmark_dir / image_dirs[args.benchmark]
    if not image_base_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_base_dir}")
    
    return {
        "base_dir": base_dir,
        "benchmark_dir": benchmark_dir,
        "benchmark_file": benchmark_file,
        "image_base_dir": image_base_dir
    }

def process_qa_item(item, args, paths):
    """Process single QA item with enhanced local model support"""
    item_id = item.get("id", f"unknown_{hash(str(item))}")
    base_img = item.get("img_path")
    
    result = {
        "id": item_id,
        "original_img_filename": base_img,
        "source": item.get("source"),
        "time": item.get("time"),
        "level1": None,
        "level2": None
    }
    
    for level_key in ["level1_qa", "level2_qa"]:
        if level_key not in item or not item[level_key]:
            continue
            
        qa_data = item[level_key]
        question = qa_data.get("question")
        ground_truth = qa_data.get("Ground_Truth_List")
        question_type = qa_data.get("question_type", "unknown")
        
        img_filename = qa_data.get("img_path", base_img)
        if not img_filename:
            continue
            
        full_image_path = paths["image_base_dir"] / img_filename
        
        level_result = {
            "question": question,
            "question_type": question_type,
            "img_filename_from_json": img_filename,
            "img_path_used": str(full_image_path),
            "ground_truth_list": ground_truth,
            "model_qa_output": None,
            "judge_evaluation": None
        }
        
        if not all([question, ground_truth, full_image_path.exists()]):
            level_result["model_qa_output"] = {
                "answer": None, 
                "confidence_score": 0, 
                "error": "Incomplete data or missing image"
            }
            level_result["judge_evaluation"] = {
                "grade": None,
                "reasoning": "Incomplete input data",
                "status_for_stats": "Error"
            }
        else:
            # Get QA answer
            qa_base_url = args.local_base_url if args.qa_provider == "local" else None
            qa_response = get_qa_answer_local(
                str(full_image_path), question, args.qa_model,
                args.qa_provider, qa_base_url, args.api_delay
            )
            level_result["model_qa_output"] = qa_response
            
            # Judge answer
            judge_base_url = args.judge_base_url or qa_base_url if args.judge_provider == "local" else None
            judge_response = judge_answer_local(
                question, qa_response.get("answer"), ground_truth,
                args.judge_model, args.judge_provider, judge_base_url, args.api_delay
            )
            level_result["judge_evaluation"] = judge_response
        
        result[level_key.replace("_qa", "")] = level_result
    
    return result

def save_result_incrementally(result, filepath):
    """Save result incrementally with thread safety"""
    with output_file_lock:
        current_results = []
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        current_results = json.loads(content)
                        if not isinstance(current_results, list):
                            current_results = []
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error reading existing results: {e}")
                current_results = []
        
        current_results.append(result)
        
        try:
            temp_file = filepath + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, filepath)
        except Exception as e:
            logger.error(f"Error saving result: {e}")

def load_existing_results(filepath):
    """Load existing results and return processed IDs"""
    processed_ids = set()
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    results = json.loads(content)
                    if isinstance(results, list):
                        for item in results:
                            if "id" in item:
                                processed_ids.add(item["id"])
        except Exception as e:
            logger.warning(f"Error loading existing results: {e}")
    
    return processed_ids

def generate_statistics(results):
    """Generate evaluation statistics"""
    stats = {
        'level1': defaultdict(int),
        'level2': defaultdict(int), 
        'by_type': defaultdict(lambda: defaultdict(int))
    }
    
    for result in results:
        for level_key in ['level1', 'level2']:
            level_data = result.get(level_key)
            if not level_data or not level_data.get("judge_evaluation"):
                continue
                
            question_type = level_data.get("question_type", "unknown")
            judge_eval = level_data["judge_evaluation"]
            status = judge_eval.get("status_for_stats", "Error")
            
            stats[level_key]['total_questions'] += 1
            stats['by_type'][question_type]['total_questions'] += 1
            
            if status == "CORRECT":
                stats[level_key]['graded_A_correct'] += 1
                stats['by_type'][question_type]['graded_A_correct'] += 1
            elif status == "INCORRECT":
                stats[level_key]['graded_B_incorrect'] += 1
                stats['by_type'][question_type]['graded_B_incorrect'] += 1
            elif status == "NOT_ATTEMPTED":
                stats[level_key]['graded_C_not_attempted'] += 1
                stats['by_type'][question_type]['graded_C_not_attempted'] += 1
            else:
                stats[level_key]['errors_judge_processing'] += 1
                stats['by_type'][question_type]['errors_judge_processing'] += 1
    
    return stats

def write_summary(stats, output_file, args):
    """Write summary statistics to file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            import time
            f.write(f"--- {args.qa_model} @ {args.benchmark} Benchmark ---\n")
            f.write(f"QA Provider: {args.qa_provider}\n")
            f.write(f"Judge Model: {args.judge_model} ({args.judge_provider})\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Level statistics
            for level in ['level1', 'level2']:
                s = stats[level]
                total = s['total_questions']
                correct = s['graded_A_correct']
                incorrect = s['graded_B_incorrect']
                not_attempted = s['graded_C_not_attempted']
                errors = s['errors_judge_processing']
                
                f.write(f"{level.capitalize()}:\n")
                if total > 0:
                    f.write(f"  Total: {total}\n")
                    f.write(f"  Correct (A): {correct}\n")
                    f.write(f"  Incorrect (B): {incorrect}\n")
                    f.write(f"  Not Attempted (C): {not_attempted}\n")
                    f.write(f"  Errors: {errors}\n")
                    
                    valid_attempts = correct + incorrect
                    if valid_attempts > 0:
                        accuracy = (correct / valid_attempts) * 100
                        f.write(f"  Accuracy (A/(A+B)): {accuracy:.2f}%\n")
                    
                    total_valid = correct + incorrect + not_attempted
                    if total_valid > 0:
                        overall_accuracy = (correct / total_valid) * 100
                        f.write(f"  Overall Accuracy (A/(A+B+C)): {overall_accuracy:.2f}%\n")
                f.write("\n")
            
            # Overall statistics
            overall_total = stats['level1']['total_questions'] + stats['level2']['total_questions']
            overall_correct = stats['level1']['graded_A_correct'] + stats['level2']['graded_A_correct']
            overall_incorrect = stats['level1']['graded_B_incorrect'] + stats['level2']['graded_B_incorrect']
            overall_not_attempted = stats['level1']['graded_C_not_attempted'] + stats['level2']['graded_C_not_attempted']
            
            f.write("Overall:\n")
            if overall_total > 0:
                f.write(f"  Total Questions: {overall_total}\n")
                f.write(f"  Correct: {overall_correct}\n")
                f.write(f"  Incorrect: {overall_incorrect}\n")
                f.write(f"  Not Attempted: {overall_not_attempted}\n")
                
                valid_total = overall_correct + overall_incorrect
                if valid_total > 0:
                    accuracy = (overall_correct / valid_total) * 100
                    f.write(f"  Accuracy: {accuracy:.2f}%\n")
            
    except Exception as e:
        logger.error(f"Error writing summary: {e}")

def main():
    args = parse_arguments()
    setup_environment(args)
    
    try:
        paths = get_project_paths(args)
    except FileNotFoundError as e:
        logger.error(f"Path configuration error: {e}")
        logger.error("Please ensure you're running from the correct directory or specify --project-root")
        return
    
    # Output files
    qa_safe_name = args.qa_model.replace('/', '_').replace(':', '_')
    provider_suffix = f"_{args.qa_provider}" if args.qa_provider != "openai" else ""
    
    output_json = paths["benchmark_dir"] / f"{qa_safe_name}_results{provider_suffix}.json"
    output_summary = paths["benchmark_dir"] / f"{qa_safe_name}_summary{provider_suffix}.txt"
    
    # Check model availability
    if not is_model_available(args.qa_model, args.qa_provider, args.local_base_url):
        logger.error(f"QA model {args.qa_model} ({args.qa_provider}) not available")
        return
    
    judge_url = args.judge_base_url or args.local_base_url
    if not is_model_available(args.judge_model, args.judge_provider, judge_url):
        logger.error(f"Judge model {args.judge_model} ({args.judge_provider}) not available")
        return
    
    # Load benchmark data
    try:
        with open(paths["benchmark_file"], 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading benchmark data: {e}")
        return
    
    # Check existing results
    processed_ids = load_existing_results(output_json)
    unprocessed_items = [item for item in benchmark_data 
                        if item.get("id", f"unknown_{hash(str(item))}") not in processed_ids]
    
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} processed, {len(unprocessed_items)} remaining")
    
    if not unprocessed_items:
        logger.info("All items processed, generating final statistics")
        with open(output_json, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        # Process unprocessed items
        logger.info(f"Processing {len(unprocessed_items)} items with {args.workers} workers")
        logger.info(f"Project root: {paths['base_dir']}")
        logger.info(f"Image directory: {paths['image_base_dir']}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_qa_item, item, args, paths)
                for item in unprocessed_items
            ]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(unprocessed_items), desc="Processing"):
                try:
                    result = future.result()
                    if result:
                        save_result_incrementally(result, output_json)
                except Exception as e:
                    logger.error(f"Processing error: {e}")
        
        # Load all results for statistics
        with open(output_json, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    
    # Generate and save statistics
    stats = generate_statistics(all_results)
    write_summary(stats, output_summary, args)
    
    logger.info(f"Evaluation completed.")
    logger.info(f"Results: {output_json}")
    logger.info(f"Summary: {output_summary}")

if __name__ == "__main__":
    main()
