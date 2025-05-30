"""
Image processing module for handling images in topics
"""

import base64
import re
import json
import logging
import os
from ranking.config import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from ranking.client import call_gpt

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path):
    """Encode an image to base64 format"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        return None

def create_messages_for_topic(topic_data):
    """Create messages for image analysis of a news article"""
    # Create system message
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Create user message with article content
    title = topic_data.get('topic', 'No title')
    text = topic_data.get('text', 'No text')
    img_paths = topic_data.get('img_paths', [])
    
    user_content = USER_PROMPT_TEMPLATE.format(
        title=title,
        text=text,
        image_count=len(img_paths)
    )
    
    messages.append({"role": "user", "content": user_content})
    
    # Add images as attachments in a separate message
    image_content = create_image_content(topic_data)
    if image_content:
        messages.append({
            "role": "user", 
            "content": image_content
        })
    else:
        messages.append({
            "role": "user", 
            "content": "This article has no available images. Please evaluate if the article should be kept."
        })
    
    return messages

def create_image_content(topic_data):
    """Create content list with images for GPT-4o"""
    img_paths = topic_data.get('img_paths', [])
    img_urls = topic_data.get('img_urls', [])
    captions = topic_data.get('captions', [])
    
    # If no images, return None
    if not img_paths:
        return None
    
    # Create content list
    content = [{"type": "text", "text": "Here are the images from this article:"}]
    
    for i, img_path in enumerate(img_paths):
        if not img_path:
            continue

        caption = captions[i] if i < len(captions) else "null"
        url = img_urls[i] if i < len(img_urls) else "N/A"
        
        # Add image info
        content.append({
            "type": "text",
            "text": f"Image {i+1}\nCaption: {caption}\nPath: {img_path}\nURL: {url}"
        })
        
        # Add image itself
        base64_image = encode_image_to_base64(img_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
    
    # If we couldn't add any images, return None
    if len(content) <= 1:
        return None
        
    return content

def process_topic_images(client, topic_data):
    """Process a news topic with GPT-4o for image analysis"""
    messages = create_messages_for_topic(topic_data)
    
    # Skip processing if no messages were created
    if not messages:
        logger.warning(f"No messages created for topic: {topic_data.get('topic', '')[:50]}")
        return None
    
    try:
        logger.info(f"Sending request to OpenAI API for topic: {topic_data.get('topic', '')[:50]}")
        
        response = call_gpt(client, messages)
        if not response:
            return None
            
        reply_content = response.choices[0].message.content
        logger.debug(f"API response received: {reply_content[:100]}...")
        
        # Extract JSON content
        try:
            # Find JSON pattern in the response
            json_match = re.search(r'(\{[\s\S]*\})', reply_content)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result
            else:
                logger.error(f"Could not extract JSON from response: {reply_content[:200]}...")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Original response: {reply_content[:200]}...")
            return None
            
    except Exception as e:
        logger.error(f"Error processing topic: {str(e)}")
        return None