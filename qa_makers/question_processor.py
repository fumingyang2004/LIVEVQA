import re
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional

from ranking.client import setup_client, get_thread_client, call_gpt_with_retry
from qa_makers.image_utils import encode_image_to_base64
from qa_makers.prompt_generator import create_prompt_for_topic
from qa_makers.file_utils import save_results

logger = logging.getLogger(__name__)

# Synchronization locks
output_lock = threading.Lock()
save_lock = threading.Lock()

def generate_question_for_image(client, prompt_data):
    """Generates a question for a single image."""
    system_prompt = prompt_data["system"]
    user_prompt = prompt_data["user"]
    img_path = prompt_data["img_path"]
    
    # Create message list
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Add image
    base64_image = encode_image_to_base64(img_path)
    if base64_image:
        messages.append({
            "role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    
    # Call API
    try:
        response = call_gpt_with_retry(
            client, 
            messages, 
            max_retries=3, 
            retry_delay=2
        )
        
        if not response:
            logger.error("API call failed")
            return None
            
        content = response.choices[0].message.content
        
        # Extract JSON part
        json_match = re.search(r'```json\n([\s\S]*?)\n```|(\{[\s\S]*\})', content)
        
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            try:
                result = json.loads(json_str)
                
                # Check for errors
                if "error" in result:
                    logger.warning(f"Failed to generate question: {result['error']}")
                    return None
                    
                return result
            except json.JSONDecodeError:
                logger.error(f"JSON parsing error: {content[:200]}")
        else:
            logger.error(f"Could not extract JSON from response: {content[:200]}")
            
        return None
    except Exception as e:
        logger.error(f"Error generating question: {str(e)}")
        return None

def process_topic(topic, output_data, config, last_save_time):
    """Processes all images for a single topic."""
    # Check if it's a placeholder topic
    if topic.get('id') is not None and topic.get('topic') is None:
        logger.info(f"Skipping placeholder topic ID: {topic.get('id')}")
        # Add to output, maintaining consistency but not generating questions
        topic_copy = {k: topic.get(k) for k in topic.keys()}
        topic_copy["level1_qas"] = []
        
        with output_lock:
            output_data.append(topic_copy)
        return True
    
    # Create a copy of the topic
    topic_copy = {k: topic.get(k) for k in topic.keys()}
    topic_copy["level1_qas"] = []
    
    # Get client
    client = get_thread_client()
    
    # Get image paths
    img_paths = topic.get('img_paths', [])
    img_urls = topic.get('img_urls', [])
    
    if not img_paths:
        logger.info(f"No images for topic: {topic.get('topic', '')[:50]}")
        with output_lock:
            output_data.append(topic_copy)
        return True
    
    # Track used question types and question content to avoid duplicates
    used_question_types = set()
    used_questions = []
    
    # Process each image
    for img_index, img_path in enumerate(img_paths):
        if not img_path:
            continue
            
        logger.info(f"Processing image {img_index+1}/{len(img_paths)} for topic '{topic.get('topic', '')[:30]}'")
        
        # Get URL for the current image
        img_url = img_urls[img_index] if img_index < len(img_urls) else ""
        
        # Create prompt, passing in used question types and question content
        prompt_data = create_prompt_for_topic(topic, img_index, used_question_types, used_questions)
        if not prompt_data:
            continue
            
        # Generate question
        question_data = generate_question_for_image(client, prompt_data)
        
        if question_data:
            # Check for duplicate questions
            is_duplicate = False
            new_question = question_data.get('question', '').lower()
            new_type = question_data.get('question_type', '').lower()
            
            # Check question similarity
            for existing_question in used_questions:
                # Simple similarity check: if key parts of the question are similar, consider it a duplicate
                existing_processed = existing_question.lower().replace("based on the provided image, ", "")
                new_processed = new_question.lower().replace("based on the provided image, ", "")
                
                # Extract the main part of the question (usually content after wh-words)
                existing_focus = re.sub(r'^(what|who|where|when|which|how|why)\s+', '', existing_processed)
                new_focus = re.sub(r'^(what|who|where|when|which|how|why)\s+', '', new_processed)
                
                if (new_focus in existing_focus or existing_focus in new_focus) and len(new_focus) > 10:
                    is_duplicate = True
                    logger.warning(f"Duplicate question detected: '{new_question}' with '{existing_question}'")
                    break
            
            # If the question is not a duplicate, add it to the results
            if not is_duplicate:
                # Update used question types and question content
                used_question_types.add(new_type)
                used_questions.append(new_question)
                
                # Add image path and URL fields to question data
                question_data["img_path"] = img_path
                question_data["img_url"] = img_url
                
                # Add to question list
                topic_copy["level1_qas"].append(question_data)
                logger.info(f"Successfully generated question for topic '{topic.get('topic', '')[:30]}': {question_data.get('question', '')[:50]}")
            else:
                logger.info(f"Skipping duplicate question: {new_question[:50]}")
    
    # Add to output
    with output_lock:
        output_data.append(topic_copy)
        
    # Save results
    with save_lock:
        last_save_time = save_results(config["output_file"], output_data, last_save_time)
        
    return True

def process_topic_thread(topic, output_data, config, last_save_time):
    """Thread function: Processes a single topic."""
    try:
        success = process_topic(topic, output_data, config, last_save_time)
        return success
    except Exception as e:
        logger.error(f"Error processing topic in thread: {str(e)}")
        return False