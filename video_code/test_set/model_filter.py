"""
L1 Topics Automatic Filter

This script uses GPT-4.1 model to evaluate and filter question entries that don't meet requirements.
"""

import os
import sys
import json
import time
import base64
import argparse
import logging
import glob
import re
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
import numpy as np

# Base directories
BASE_DIR = "/mnt/nvme1/fmy/LiveVQApro"
DATA_DIR = os.path.join(BASE_DIR, "data/raw_data")
STATS_DIR = os.path.join(BASE_DIR, "data/test_set")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for dir_path in [DATA_DIR, STATS_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Configure logging
log_file = os.path.join(LOG_DIR, "model_filter.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API configuration
API_KEY = ""
API_MODEL = ""
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Thread lock
save_lock = threading.Lock()
stats_lock = threading.Lock()

# Get latest L1 topics file
def get_latest_l1_topics_file():
    """Get the latest l1_topics_{timestamp}.json file in data/raw_data directory"""
    files = glob.glob(os.path.join(DATA_DIR, "l1_topics_*.json"))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

# Extract timestamp from filename
def get_timestamp_from_filename(filename):
    """Extract timestamp from filename"""
    match = re.search(r'l1_topics_(\d+)\.json', filename)
    if match:
        return match.group(1)
    return None

# Determine input and output file paths
def determine_io_files(timestamp=None):
    """Determine input and output file paths based on timestamp"""
    if timestamp:
        # Use specified timestamp
        input_file = os.path.join(DATA_DIR, f"l1_topics_{timestamp}.json")
        if not os.path.exists(input_file):
            logger.error(f"Specified input file does not exist: {input_file}")
            return None, None, None, None
    else:
        # Use latest file
        input_file = get_latest_l1_topics_file()
        if not input_file:
            logger.error("No l1_topics file found")
            return None, None, None, None
        timestamp = get_timestamp_from_filename(input_file)
        if not timestamp:
            logger.error(f"Could not extract timestamp from filename: {input_file}")
            return None, None, None, None
    
    # Set output file paths
    output_file = os.path.join(DATA_DIR, f"l1_filtered_topics_{timestamp}.json")
    stats_file = os.path.join(STATS_DIR, f"statistics_{timestamp}.json")
    discard_log_file = os.path.join(STATS_DIR, f"discarded_items_{timestamp}.txt")
    
    return input_file, output_file, stats_file, discard_log_file, timestamp

# Encode image to base64
def encode_image_to_base64(image_path):
    """Encode image as base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error: {image_path} - {e}")
        return None

# GPT-4.1 API call function
def call_gpt_model(messages, max_retries=3, retry_delay=2):
    """
    Call GPT-4.1 API for content evaluation
    
    Args:
        messages: API request messages
        max_retries: Maximum retry attempts
        retry_delay: Retry wait time (seconds)
        
    Returns:
        API response content or None (if call fails)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": API_MODEL,
        "messages": messages,
        "temperature": 0.1,  # Low temperature for more consistent results
        "max_tokens": 1500
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {response.status_code} {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        except Exception as e:
            logger.warning(f"API call exception (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    logger.error(f"API call failed, maximum retry attempts reached")
    return None

# Create evaluation prompt
def create_evaluation_prompt(topic_item):
    """
    Create prompt to evaluate topic entry
    
    Args:
        topic_item: Topic entry data
        
    Returns:
        List of messages with system and user prompts
    """
    system_prompt = """You are a specialized AI assistant tasked with evaluating and filtering news-related visual questions. Your expertise is in identifying high-quality questions that require temporal and social context knowledge. Please carefully review each news article and its associated questions to determine if they meet our rigorous quality standards.

YOUR TASK:
Evaluate each question associated with the news topic against these strict quality criteria, and identify questions that should be removed.

EVALUATION CRITERIA - A question MUST BE DISCARDED if ANY of these conditions apply:

1. TEMPORAL CONTEXT: The image shows no clear temporal or contemporary information. Modern context is ESSENTIAL.
   - DISCARD if: The image is timeless, could have been taken anytime in the last decade, or lacks clear contemporary elements
   - DISCARD if: The image is just a close-up of an object with no time-specific context (especially common in food-related news)

2. FAMOUS FIGURE RECOGNITION: Questions about extremely well-known figures that are too obvious.
   - DISCARD if: The image shows Donald Trump, and the question asks who he is

3. AMBIGUOUS ANSWERS: Ground_Truth values that are too general or ambiguous to be definitive.
   - DISCARD if: Answers like "Designer sneakers", "high-end sneakers" that could refer to many different items
   - GOOD EXAMPLE: "Nike Air Force 1" or "Louis Vuitton Trainers" (specific identifiable items)
   
4. SIMPLE COUNTING: Questions that just require counting visible objects.
   - DISCARD if: Questions asking "how many X are in the image" with a numerical answer

5. BOOK COVERS: Questions about book covers that merely ask about the cover itself.
   - DISCARD if: The image shows a book cover and the answer is simply "book cover", "memoir cover", "book jacket", etc.

6. VISIBLE TEXT ANSWERS: Questions where the answer is directly visible as text in the image.
   - DISCARD if: The answer is "Google" and the image shows the Google logo with text
   - DISCARD if: Any brand name, sign, or text in the image that directly shows the answer

7. GENERIC LOCATION/ESTABLISHMENT TYPES: Questions with answers that only identify a generic type of place.
   - DISCARD if: Answers like "textile factory", "garment factory", "shopping mall", "clothing store" without specific identification
   - GOOD EXAMPLE: "Nike Factory in Vietnam" or "Galeries Lafayette department store" (specific identifiable places)

8. GENERIC EVENT DESCRIPTIONS: Events without specific identification.
   - DISCARD if: Answers like "stunt performance", "protest", "fashion show" without specific event naming
   - GOOD EXAMPLE: "2023 Paris Fashion Week" or "Black Lives Matter protest in Portland" (specific identifiable events)

9. CHART DATA: Questions about data obviously present in charts/graphs.
   - DISCARD if: The image is a scientific chart and the question just asks about data clearly presented in it

10. INCOMPLETE CONTENT: Missing required elements.
    - DISCARD if: The topic has no questions or no images

11. GENERIC PERSON DESCRIPTIONS: Questions about people with vague descriptive answers.
    - DISCARD if: Answers like "police officer", "protestor", "doctor" without specific person identification
    - GOOD EXAMPLE: "Emmanuel Macron" or "Taylor Swift" (specific identifiable people)

INSTRUCTIONS:
1. Analyze the provided news article, its image(s), and each associated question
2. For each question, determine if it violates ANY of the above criteria
3. Return a JSON response indicating:
   - Which questions should be removed and why
   - Which questions are acceptable to keep
4. Be extremely strict in your evaluation - when in doubt, discard the question

REMEMBER: Time-specific context is our most critical requirement. ANY image lacking clear contemporary elements must be discarded.
"""

    # Prepare topic data
    topic = topic_item.get('topic', 'No title')
    text = topic_item.get('text', 'No text')
    img_paths = topic_item.get('img_paths', [])
    img_urls = topic_item.get('img_urls', [])
    captions = topic_item.get('captions', [])
    qa_items = topic_item.get('level1_qas', [])
    
    # Create text representation of question data
    questions_text = ""
    
    # Ensure qa_items is a list type and not empty
    if not isinstance(qa_items, list):
        qa_items = []
    
    # Fix: Properly handle multiple questions in level1_qas array
    for i, qa in enumerate(qa_items):
        if not isinstance(qa, dict):
            continue
        
        question = qa.get('question', 'No question')
        question_type = qa.get('question_type', 'Unknown type')
        options = qa.get('options', [])
        correct_answer = qa.get('Ground_Truth', '')
        correct_answer_list = qa.get('Ground_Truth_List', [])
        
        questions_text += f"\nQUESTION {i+1}:\n"
        questions_text += f"- Question: {question}\n"
        questions_text += f"- Question Type: {question_type}\n"
        
        if options:
            questions_text += "- Options:\n"
            for opt in options:
                questions_text += f"  * {opt}\n"
        
        if correct_answer:
            questions_text += f"- Correct Answer: {correct_answer}\n"
        
        if correct_answer_list:
            answers_str = ", ".join([f'"{ans}"' for ans in correct_answer_list])
            questions_text += f"- Acceptable Answers: [{answers_str}]\n"
    
    # Create user prompt
    user_prompt = f"""Please evaluate this news topic and its associated questions:

TOPIC ID: {topic_item.get('id', 'Unknown')}

ARTICLE TITLE: {topic}

ARTICLE TEXT: {text}

IMAGES ({len(img_paths)} total):
"""

    # Add image descriptions
    for i, (img_path, caption) in enumerate(zip(img_paths, captions) if len(img_paths) == len(captions) else []):
        user_prompt += f"IMAGE {i+1}: {img_path}\n"
        if caption and caption != "null":
            user_prompt += f"CAPTION: {caption}\n"
    
    # Add questions
    if questions_text:
        user_prompt += f"\nQUESTIONS TO EVALUATE:{questions_text}"
    else:
        user_prompt += "\nQUESTIONS TO EVALUATE: None provided"
    
    user_prompt += """

Please analyze each question against our strict quality criteria and provide your evaluation in this JSON format:

```json
{
  "topic_id": "ID of the topic",
  "evaluation": {
    "questions_to_discard": [
      {
        "question_index": 0, // 0-based index of the question to discard
        "reason": "Clear explanation of why this question should be discarded, citing specific criteria violated"
      },
      // Additional questions to discard...
    ],
    "questions_to_keep": [0, 1, 2] // 0-based indices of questions that meet all quality criteria
  },
  "discard_entire_topic": false, // Set to true ONLY if all questions should be discarded or the topic has no valid questions
  "reasoning": "Brief explanation of your overall assessment, especially if discarding the entire topic"
}
```

Be extremely strict in your evaluation. Remember that images MUST show clear temporal context (recent/contemporary elements), and answers must be specific and uniquely identifiable rather than generic descriptions.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Add image attachments (if available)
    for img_path in img_paths:
        if os.path.exists(img_path):
            base64_image = encode_image_to_base64(img_path)
            if base64_image:
                messages.append({
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                })
    
    return messages

# Parse model response
def parse_model_response(response, topic):
    """
    Parse model response and extract evaluation results
    
    Args:
        response: Model API response
        topic: Original topic entry
        
    Returns:
        tuple: (Updated topic data, discarded question indices list, discard entire topic flag, discard reason)
    """
    if not response or 'choices' not in response or not response['choices']:
        logger.error("Invalid API response")
        return topic, [], False, "Invalid API response"
    
    try:
        content = response['choices'][0]['message']['content']
        
        # Extract JSON from response
        json_match = re.search(r'```json\n([\s\S]*?)\n```|(\{[\s\S]*\})', content)
        if not json_match:
            logger.error(f"Could not extract JSON from response: {content[:200]}...")
            return topic, [], False, "Failed to extract JSON from response"
        
        json_str = json_match.group(1) or json_match.group(2)
        
        # Try to clean up some common JSON formatting issues
        json_str = json_str.replace('```', '').strip()
        
        try:
            evaluation = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Problematic JSON: {json_str[:500]}...")
            
            # Try to fix common JSON errors
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            
            try:
                evaluation = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(f"Could not fix JSON format, original content: {content[:500]}...")
                return topic, [], False, "JSON parse error"
        
        # Extract evaluation results
        questions_to_discard = evaluation.get('evaluation', {}).get('questions_to_discard', [])
        discard_indices = []
        
        # Ensure questions_to_discard is a list
        if isinstance(questions_to_discard, list):
            for item in questions_to_discard:
                if isinstance(item, dict) and 'question_index' in item:
                    try:
                        # Ensure index is an integer
                        idx = int(item['question_index'])
                        discard_indices.append(idx)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid question index: {item['question_index']}")
        
        discard_entire_topic = evaluation.get('discard_entire_topic', False)
        reasoning = evaluation.get('reasoning', "No reasoning provided")
        
        # If we need to discard the entire topic
        if discard_entire_topic:
            # Keep ID, set other fields to null
            topic_id = topic.get('id')
            updated_topic = {k: None if k != 'id' else topic_id for k in topic}
            updated_topic['discarded'] = True
            updated_topic['discard_reason'] = reasoning
            return updated_topic, [], True, reasoning
        
        # Process questions to be discarded
        if discard_indices:
            qa_items = topic.get('level1_qas', [])
            
            # Ensure qa_items is a list type
            if not isinstance(qa_items, list):
                qa_items = []
            elif len(qa_items) == 0:
                # If qa_items is an empty list, mark entire topic for discard
                topic_id = topic.get('id')
                updated_topic = {k: None if k != 'id' else topic_id for k in topic}
                updated_topic['discarded'] = True
                updated_topic['discard_reason'] = "Empty level1_qas"
                return updated_topic, [], True, "Empty level1_qas"
            
            # Create new list with only non-discarded questions
            updated_qa_items = []
            
            # Ensure indices in discard_indices are within valid range
            valid_discard_indices = [idx for idx in discard_indices if 0 <= idx < len(qa_items)]
            
            # If number of indices to discard equals total questions, discard entire topic
            if len(valid_discard_indices) >= len(qa_items):
                topic_id = topic.get('id')
                updated_topic = {k: None if k != 'id' else topic_id for k in topic}
                updated_topic['discarded'] = True
                updated_topic['discard_reason'] = "All questions were discarded"
                return updated_topic, valid_discard_indices, True, "All questions were discarded"
            
            # Keep only non-discarded questions
            for i, qa in enumerate(qa_items):
                if i in valid_discard_indices:
                    # Skip discarded questions
                    logger.info(f"Discarding question {i}: {qa.get('question', '')[:50]}...")
                else:
                    updated_qa_items.append(qa)
            
            # Update topic with non-discarded questions
            topic['level1_qas'] = updated_qa_items
            
            # If all questions were discarded (considering invalid indices)
            if not updated_qa_items:
                topic_id = topic.get('id')
                updated_topic = {k: None if k != 'id' else topic_id for k in topic}
                updated_topic['discarded'] = True
                updated_topic['discard_reason'] = "All questions were discarded"
                return updated_topic, valid_discard_indices, True, "All questions were discarded"
        
        # Update image-related fields
        kept_image_indices = [i for i in range(len(topic.get('level1_qas', []))) if i not in discard_indices]
        if 'img_urls' in topic and isinstance(topic['img_urls'], list):
            topic['img_urls'] = [topic['img_urls'][i] for i in kept_image_indices if i < len(topic['img_urls'])]
        if 'img_paths' in topic and isinstance(topic['img_paths'], list):
            topic['img_paths'] = [topic['img_paths'][i] for i in kept_image_indices if i < len(topic['img_paths'])]
        if 'captions' in topic and isinstance(topic['captions'], list):
            topic['captions'] = [topic['captions'][i] for i in kept_image_indices if i < len(topic['captions'])]
        
        return topic, discard_indices, discard_entire_topic, reasoning
    
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        logger.error(traceback.format_exc())
        return topic, [], False, f"Error: {str(e)}"

# Process single topic
def process_topic(topic, discarded_items, progress=None):
    """
    Process individual topic entry
    
    Args:
        topic: Topic entry data
        discarded_items: Shared list of discarded entries
        progress: tqdm progress bar object
        
    Returns:
        Processed topic entry
    """
    if progress:
        progress.set_description(f"Processing ID: {topic.get('id', 'Unknown')}")
    
    # Check if topic is empty or already discarded
    if topic.get('discarded') or not topic.get('topic') or not topic.get('level1_qas'):
        # If topic is already discarded or has no questions, log and skip
        topic_id = topic.get('id')
        if topic_id is not None:
            with stats_lock:
                discarded_items.append({
                    'id': topic_id,
                    'reason': "Topic was already discarded or has no questions"
                })
        
        # Keep ID, set other fields to null
        updated_topic = {k: None if k != 'id' else topic_id for k in topic}
        updated_topic['discarded'] = True
        updated_topic['discard_reason'] = "Topic was already discarded or has no questions"
        
        if progress:
            progress.update(1)
        return updated_topic
    
    # Ensure level1_qas is a list type
    qa_items = topic.get('level1_qas', [])
    if not isinstance(qa_items, list):
        logger.warning(f"level1_qas for topic ID:{topic.get('id')} is not a list type, will be initialized as empty list")
        topic['level1_qas'] = []
        # Mark as discarded
        updated_topic = {k: None if k != 'id' else topic.get('id') for k in topic}
        updated_topic['discarded'] = True
        updated_topic['discard_reason'] = "Invalid level1_qas format - not a list"
        
        with stats_lock:
            discarded_items.append({
                'id': topic.get('id'),
                'reason': "Invalid level1_qas format"
            })
            
        if progress:
            progress.update(1)
        return updated_topic
        
    # Check if level1_qas is an empty list
    if len(qa_items) == 0:
        logger.warning(f"level1_qas for topic ID:{topic.get('id')} is an empty list, will be skipped")
        updated_topic = {k: None if k != 'id' else topic.get('id') for k in topic}
        updated_topic['discarded'] = True
        updated_topic['discard_reason'] = "Empty level1_qas"
        
        with stats_lock:
            discarded_items.append({
                'id': topic.get('id'),
                'reason': "Empty level1_qas list"
            })
            
        if progress:
            progress.update(1)
        return updated_topic
    
    # Create evaluation prompt
    messages = create_evaluation_prompt(topic)
    
    # Call model
    response = call_gpt_model(messages)
    
    # Parse response
    updated_topic, discard_indices, discard_entire_topic, reasoning = parse_model_response(response, topic)
    
    # Log discarded entries
    if discard_entire_topic and topic.get('id') is not None:
        with stats_lock:
            discarded_items.append({
                'id': topic.get('id'),
                'reason': reasoning
            })
    
    if progress:
        progress.update(1)
    
    return updated_topic

# Safely save JSON
def safe_save_json(file_path, data):
    """
    Safely save JSON data to file
    
    Args:
        file_path: File path
        data: Data to save
        
    Returns:
        bool: Whether save was successful
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create temporary file
        tmp_file = f"{file_path}.tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Use atomic operation to replace original file
        os.replace(tmp_file, file_path)
        logger.info(f"Successfully saved to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        # Clean up temporary file
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except:
                pass
        return False

# Process topic list
def process_topics_with_model(topics, output_file, stats_file, discard_log_file, max_workers=8):
    """
    Process topic list using model
    
    Args:
        topics: Topic list
        output_file: Output file path
        stats_file: Statistics file path
        discard_log_file: Discard log file path
        max_workers: Maximum parallel worker threads
        
    Returns:
        bool: Whether processing was successful
    """
    # For recording discarded entries
    discarded_items = []
    
    # List for processed results
    processed_topics = []
    
    # Total processed count
    processed_count = 0
    
    try:
        # Use thread pool to process topics
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for topic in topics:
                future = executor.submit(process_topic, topic, discarded_items)
                futures.append(future)
            
            # Process results
            with tqdm(total=len(futures), desc="Evaluating topics") as progress:
                for i, future in enumerate(as_completed(futures)):
                    try:
                        processed_topic = future.result()
                        processed_topics.append(processed_topic)
                        processed_count += 1
                        
                        # Save results every 10 items
                        if processed_count % 10 == 0:
                            with save_lock:
                                safe_save_json(output_file, processed_topics)
                                
                    except Exception as e:
                        logger.error(f"Error processing topic: {e}")
                    
                    # Update progress bar description
                    discard_count = len([t for t in processed_topics if t.get('discarded')])
                    progress.set_description(f"Processed: {processed_count}/{len(topics)} Discarded: {discard_count}")
                    progress.update(1)
        
        # Final save of results
        final_save_success = safe_save_json(output_file, processed_topics)
        
        # Save statistics data
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_topics": len(topics),
            "processed_topics": processed_count,
            "discarded_topics": len([t for t in processed_topics if t.get('discarded')]),
            "discarded_items": discarded_items
        }
        
        stats_save_success = safe_save_json(stats_file, stats)
        
        # Save discard log
        with open(discard_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Discarded items ({len(discarded_items)}):\n")
            for item in discarded_items:
                f.write(f"ID: {item['id']} - Reason: {item['reason']}\n")
        
        logger.info(f"Processing complete, total processed {processed_count} topics, "
                   f"discarded {len([t for t in processed_topics if t.get('discarded')])} topics")
        
        return final_save_success and stats_save_success
    
    except Exception as e:
        logger.error(f"Error processing topic list: {e}")
        return False

# Main function
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Filter L1 topics using GPT-4.1 model')
    parser.add_argument('--timestamp', '-t', type=str, help='File timestamp to process')
    parser.add_argument('--workers', type=int, default=8, help='Parallel worker threads')
    args = parser.parse_args()
    
    # Determine input and output files
    result = determine_io_files(args.timestamp)
    if not result:
        logger.error("Could not determine input/output files, program exiting")
        return 1
    
    input_file, output_file, stats_file, discard_log_file, timestamp = result
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Statistics file: {stats_file}")
    logger.info(f"Discard log: {discard_log_file}")
    
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        
        logger.info(f"Successfully loaded {len(topics)} topics")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return 1
    
    # Process topics
    success = process_topics_with_model(topics, output_file, stats_file, discard_log_file, args.workers)
    
    if success:
        logger.info("Processing complete")
        return 0
    else:
        logger.error("Errors occurred during processing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
