import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import openai
from tqdm import tqdm

from config import CONFIG, LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'text_evaluator.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

client = openai.OpenAI(api_key=CONFIG["api_key"])

def evaluate_text_meaningfulness(text: str, topic: str = None) -> Tuple[bool, str]:
    topic_context = f"\nRelated topic: {topic}" if topic else ""
    
    prompt = f"""Please determine whether the following text is meaningful (provides useful information, not just meaningless dialogue).
    
Text content:
"{text}"{topic_context}

Please carefully analyze if this text:
1. Contains substantive information
2. Describes specific events, people, or situations
3. Can serve as a basis for news reporting or information source
4. Is not just fragmentary, contextless dialogue
5. Is relevant to the topic (if provided)

Please respond only with "Meaningful" or "Not meaningful", followed by a brief explanation (no more than 20 words).
"""

    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Use low temperature for more consistent answers
            max_tokens=100
        )
        result = response.choices[0].message.content.strip()
        
        # Parse the response
        is_meaningful = "Meaningful" in result.split('\n')[0]
        reason = ' '.join(result.split('\n')[1:]) if len(result.split('\n')) > 1 else "No detailed reason"
        
        return is_meaningful, reason
    except Exception as e:
        logger.error(f"API call error: {e}")
        return True, f"Evaluation failed: {str(e)}"  # Default to meaningful to avoid false negatives

def process_item(item: Dict) -> Dict:
    text = item.get("text", "")
    topic = item.get("topic", "")  # Get topic information
    
    if not text:
        item["is_meaningful"] = False
        item["evaluation_reason"] = "Text is empty"
        return item
    
    is_meaningful, reason = evaluate_text_meaningfulness(text, topic)
    
    # Add evaluation results to the item
    item["is_meaningful"] = is_meaningful
    item["evaluation_reason"] = reason
    
    return item

def evaluate_json_file(
    json_file_path: str, 
    output_file_path: Optional[str] = None,
    max_workers: int = None
) -> List[Dict]:
    """
    Evaluate whether texts in a JSON file are meaningful.
    
    Args:
        json_file_path: Path to the JSON file
        output_file_path: Path to save the output file; if None, do not save
        max_workers: Maximum number of threads; defaults to config setting
        
    Returns:
        List[Dict]: List of evaluated items
    """
    if max_workers is None:
        max_workers = CONFIG.get("max_workers", 4)
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON file: {e}")
        return []
    
    logger.info(f"Starting evaluation of file: {json_file_path}, number of items: {len(data)}")
    
    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_data = list(tqdm(
            executor.map(process_item, data),
            total=len(data),
            desc="Evaluating text"
        ))
    
    # Save results (if output path specified)
    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Evaluation results saved to: {output_file_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}")
    
    # Statistics
    meaningful_count = sum(1 for item in processed_data if item.get("is_meaningful", True))
    logger.info(f"Evaluation complete: Total {len(processed_data)} items, meaningful {meaningful_count}, not meaningful {len(processed_data)-meaningful_count}")
    
    return processed_data

def filter_meaningful_items(items: List[Dict]) -> List[Dict]:
    """
    Filter out only meaningful items.
    
    Args:
        items: List of evaluated items
        
    Returns:
        List[Dict]: List containing only meaningful items
    """
    return [item for item in items if item.get("is_meaningful", True)]

def evaluate_text_data(
    input_file_path: str = "",
    output_file_path: Optional[str] = None,
    save_filtered: bool = False,
    filtered_output_path: Optional[str] = None,
    max_workers: int = None,
    keep_all: bool = True  # New parameter, default to keep all items including non-meaningful ones
) -> Tuple[List[Dict], List[Dict]]:
    """
    Public interface function: evaluate whether text data is meaningful.
    
    Args:
        input_file_path: Input JSON file path, default to specified path
        output_file_path: Output file path; if None, do not save
        save_filtered: Whether to save the filtered data file
        filtered_output_path: Path to save filtered file; if provided, use this path
        max_workers: Maximum number of threads; default to config setting
        keep_all: Whether to keep all items (including non-meaningful); default True
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (all_items, filtered_items)
                                      all_items: all evaluated items
                                      filtered_items: filtered result based on keep_all
    """
    # Handle default output file path
    if output_file_path is None and input_file_path:
        dir_name = os.path.dirname(input_file_path)
        base_name = os.path.basename(input_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file_path = os.path.join(dir_name, f"{name_without_ext}_evaluated.json")
    
    # Evaluate all items
    all_items = evaluate_json_file(
        json_file_path=input_file_path,
        output_file_path=output_file_path,
        max_workers=max_workers
    )
    
    # Decide whether to filter items based on keep_all
    filtered_items = all_items if keep_all else filter_meaningful_items(all_items)
    
    # If need to save filtered file
    if save_filtered:
        # Use specific filtered file path if provided
        if filtered_output_path:
            filtered_output = filtered_output_path
        elif output_file_path:
            suffix = "_all" if keep_all else "_filtered"
            filtered_output = os.path.join(
                os.path.dirname(output_file_path),
                f"{os.path.splitext(os.path.basename(output_file_path))[0]}{suffix}.json"
            )
        else:
            # If no output path provided, generate path based on input file
            dir_name = os.path.dirname(input_file_path)
            base_name = os.path.basename(input_file_path)
            name_without_ext = os.path.splitext(base_name)[0]
            suffix = "_all" if keep_all else "_filtered"
            filtered_output = os.path.join(dir_name, f"{name_without_ext}{suffix}.json")
        
        try:
            with open(filtered_output, 'w', encoding='utf-8') as f:
                json.dump(filtered_items, f, ensure_ascii=False, indent=2)
            logger.info(f"{'All items' if keep_all else 'Filtered meaningful items'} saved to: {filtered_output}")
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
    
    return all_items, filtered_items

if __name__ == "__main__":
    # Example usage
    input_path = ""
    all_items, filtered_items = evaluate_text_data(
        input_file_path=input_path,
        save_filtered=True,
        keep_all=True  # Keep all items, including non-meaningful ones
    )
    print(f"Total number of items: {len(all_items)}")
    print(f"Number of retained items: {len(filtered_items)}")