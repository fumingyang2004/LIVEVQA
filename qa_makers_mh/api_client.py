import os
import re
import json
import time
import logging
from openai import OpenAI
from qa_makers_mh.config import API_KEY, MODEL_NAME, MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger(__name__)

# OpenAI client initialization
def create_client():
    """Creates and returns an OpenAI client."""
    return OpenAI(api_key=API_KEY)

def generate_multihop_questions(client, prompt_data):
    """
    Generates multi-hop questions using GPT-4.1.
    
    Args:
        client: The OpenAI client.
        prompt_data: A dictionary containing system and user prompts.
        
    Returns:
        dict: A dictionary containing 'level2_qas'.
    """
    system_prompt = prompt_data["system"]
    user_prompt = prompt_data["user"]
    img_path = prompt_data["img_path"]
    
    # Create message list
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Add image if it exists
    if img_path and os.path.exists(img_path):
        from qa_makers_mh.utils import encode_image_to_base64
        base64_image = encode_image_to_base64(img_path)
        if base64_image:
            messages.append({
                "role": "user", 
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }]
            })
    
    # Retry parameters
    max_retries = MAX_RETRIES
    retry_delay = RETRY_DELAY
    last_exception = None
    
    # Retry API calls
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON part
            json_match = re.search(r'```json\n([\s\S]*?)\n```|(\{[\s\S]*\})', content)
            
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                try:
                    result = json.loads(json_str)
                    
                    # Ensure the result contains the necessary field
                    if 'level2_qas' not in result:
                        result = {'level2_qas': []}
                    
                    return result
                except json.JSONDecodeError:
                    last_exception = f"JSON parsing error: {content[:200]}..."
                    logger.error(last_exception)
            else:
                last_exception = f"Could not extract JSON from response: {content[:200]}..."
                logger.error(last_exception)
                
            # If we reach here, the attempt failed, so wait and retry
            time.sleep(retry_delay)
            
        except Exception as e:
            last_exception = str(e)
            logger.error(f"API call error (Attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(retry_delay)
    
    # If all attempts fail, return an empty result
    logger.error(f"All API call attempts failed: {last_exception}")
    return {'level2_qas': []}