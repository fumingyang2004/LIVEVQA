import re
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI
from config import CONFIG

#from ranking.client import setup_client, get_thread_client, call_gpt_with_retry
from qa_makers.image_utils import encode_image_to_base64
from qa_makers.prompt_generator import create_prompt_for_topic
from qa_makers.file_utils import save_results

logger = logging.getLogger(__name__)


# Synchronization locks
output_lock = threading.Lock()
save_lock = threading.Lock()

def setup_client():
    """Set up OpenAI client"""
    return OpenAI(api_key=CONFIG["api_key"])

_thread_local = threading.local()

def get_thread_client():
    """Get thread-local OpenAI client instance
    
    Ensure each thread has its own independent client instance to avoid conflicts between threads
    """
    if not hasattr(_thread_local, "client"):
        _thread_local.client = setup_client()
    return _thread_local.client

def call_gpt(client, messages):
    """Make a call to OpenAI GPT API
    
    Args:
        client: OpenAI client instance
        messages: List of message objects
        
    Returns:
        Response from OpenAI API
    """
    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            messages=messages,
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
        )
        return response
    except Exception as e:
        logger.error(f"API call error: {str(e)}")
        return None
    
def call_gpt_with_retry(client, messages, max_retries=3, retry_delay=2):
    """Call OpenAI API with retry mechanism
    
    Args:
        client: OpenAI client instance
        messages: List of messages
        max_retries: Maximum number of retries
        retry_delay: Retry interval (seconds)
        
    Returns:
        API response or None
    """
    attempt = 0
    while attempt < max_retries:
        try:
            return call_gpt(client, messages)
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"API call failed after {max_retries} attempts: {str(e)}")
                return None
            
            # Exponential backoff
            sleep_time = retry_delay * (2 ** (attempt - 1))
            logger.warning(f"API call error (attempt {attempt}/{max_retries}): {str(e)}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)


def generate_question_for_image(client, prompt_data):
    """Generate questions for a single image"""
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
                    logger.warning(f"Unable to generate question: {result['error']}")
                    return None
                    
                return result
            except json.JSONDecodeError:
                logger.error(f"JSON parsing error: {content[:200]}")
        else:
            logger.error(f"Unable to extract JSON from response: {content[:200]}")
            
        return None
    except Exception as e:
        logger.error(f"Error generating question: {str(e)}")
        return None

def process_topic(topic, output_data, config, last_save_time):
    """Process all images for a single topic"""
    # Check if it is a placeholder topic
    if topic.get('id') is not None and topic.get('topic') is None:
        logger.info(f"Skipping placeholder topic ID: {topic.get('id')}")
        # Add to output to maintain consistency but do not generate questions
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
        logger.info(f"Topic has no images: {topic.get('topic', '')[:50]}")
        with output_lock:
            output_data.append(topic_copy)
        return True
    
    # Track used question types and questions to avoid duplicates
    used_question_types = set()
    used_questions = []
    
    # Process each image
    for img_index, img_path in enumerate(img_paths):
        if not img_path:
            continue
            
        logger.info(f"Processing image {img_index+1}/{len(img_paths)} for topic '{topic.get('topic', '')[:30]}'")
        
        # Get current image URL
        img_url = img_urls[img_index] if img_index < len(img_urls) else ""
        
        # Create prompt, passing used question types and questions
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
                # Simple similarity check: consider duplicate if key parts are similar
                existing_processed = existing_question.lower().replace("based on the provided image, ", "")
                new_processed = new_question.lower().replace("based on the provided image, ", "")
                
                # Extract main part of the question (usually the part after wh-words)
                existing_focus = re.sub(r'^(what|who|where|when|which|how|why)\s+', '', existing_processed)
                new_focus = re.sub(r'^(what|who|where|when|which|how|why)\s+', '', new_processed)
                
                if (new_focus in existing_focus or existing_focus in new_focus) and len(new_focus) > 10:
                    is_duplicate = True
                    logger.warning(f"Duplicate question detected: '{new_question}' vs '{existing_question}'")
                    break
            
            # If question is not duplicate, add to results
            if not is_duplicate:
                # Update used question types and questions
                used_question_types.add(new_type)
                used_questions.append(new_question)
                
                # Add image path and URL fields to question data
                question_data["img_path"] = img_path
                question_data["img_url"] = img_url
                
                # Add to question list
                topic_copy["level1_qas"].append(question_data)
                logger.info(f"Successfully generated question for topic '{topic.get('topic', '')[:30]}': {question_data.get('question', '')[:50]}")
            else:
                logger.info(f"Skipped duplicate question: {new_question[:50]}")
    
    # Add to output
    with output_lock:
        output_data.append(topic_copy)
        
    # Save results
    with save_lock:
        last_save_time = save_results(config["output_file"], output_data, last_save_time)
        
    return True

def process_topic_thread(topic, output_data, config, last_save_time):
    """Thread function: process a single topic"""
    try:
        success = process_topic(topic, output_data, config, last_save_time)
        return success
    except Exception as e:
        logger.error(f"Error in topic processing thread: {str(e)}")
        return False
