"""
Topic manager module for handling and updating topics
"""

import logging
from ranking.image_processor import process_topic_images

logger = logging.getLogger(__name__)

def is_discarded_topic(topic):
    """Checks if a topic was previously discarded (has id but topic field is None)"""
    return topic.get('id') is not None and topic.get('topic') is None and topic.get('discarded', False)

def update_topic_data(topic_data, processed_result):
    """Updates topic data with processed results"""
    # Return a placeholder if the topic should be removed
    if not processed_result or not processed_result.get('keep_topic', True):
        topic_id = topic_data.get('id')
        logger.info(f"Topic ID {topic_id} marked for removal: {topic_data.get('topic', '')[:50]}...")
        # Create placeholder with only ID preserved, all other fields set to null
        return {
            'id': topic_id,
            'topic': None,
            'text': None,
            'img_urls': [],
            'img_paths': [],
            'captions': [],
            'source': None,
            'url': None,
            'discarded': True  # Flag to indicate this topic was intentionally discarded
        }
    
    processed_data = processed_result.get('processed_data', {})
    
    if processed_data:
        # Update image-related fields
        new_img_urls = processed_data.get('img_urls', [])
        new_img_paths = processed_data.get('img_paths', [])
        new_captions = processed_data.get('captions', [])
        
        # Ensure all three lists have the same length
        max_len = max(len(new_img_urls), len(new_img_paths), len(new_captions))
        if max_len > 0:
            # Fill missing values
            new_img_urls = new_img_urls + [''] * (max_len - len(new_img_urls))
            new_img_paths = new_img_paths + [''] * (max_len - len(new_img_paths))
            new_captions = new_captions + ['null'] * (max_len - len(new_captions))
            
            # Update original data
            topic_data['img_urls'] = new_img_urls
            topic_data['img_paths'] = new_img_paths
            topic_data['captions'] = new_captions
            
            # Update image tags in text if needed
            update_image_tags_in_text(topic_data, new_img_paths)
    
    return topic_data

def update_image_tags_in_text(topic_data, new_img_paths):
    """Updates image tags in the article text to match the number of images"""
    if "text" not in topic_data:
        return
        
    text = topic_data["text"]
    
    # Find existing image tags
    import re
    img_tags = re.findall(r'<img\d+>', text)
    
    # If text has no image tags but we have images
    if not img_tags and new_img_paths:
        # Add image tags at the beginning
        img_tags_text = ' '.join([f"<img{i+1}>" for i in range(len(new_img_paths))])
        topic_data["text"] = f"{img_tags_text}\n\n{text}"
    
    # If number of tags doesn't match number of images
    elif len(img_tags) != len(new_img_paths):
        # Remove excess tags or add missing tags as needed
        current_text = text
        
        # First, remove all existing image tags
        for tag in img_tags:
            current_text = current_text.replace(tag, "")
        
        # If we have images, add the correct number of tags at appropriate positions
        if new_img_paths:
            # Simple approach: add tags at the beginning
            img_tags_text = ' '.join([f"<img{i+1}>" for i in range(len(new_img_paths))])
            topic_data["text"] = f"{img_tags_text}\n\n{current_text}"
        else:
            topic_data["text"] = current_text

def process_and_update_realtime(client, topic, output_data, processed_ids):
    """Processes a single topic and updates the output data in real time"""
    try:
        # Skip processing if topic has no ID
        if 'id' not in topic:
            logger.warning(f"Topic missing ID, skipping: {topic.get('topic', '')[:50]}...")
            return False
            
        topic_id = topic['id']
        
        # Check if topic was already processed
        if topic_id in processed_ids:
            existing_entry = processed_ids[topic_id]
            
            # If the topic was previously discarded, just include the placeholder
            if is_discarded_topic(existing_entry):
                logger.info(f"Topic ID {topic_id} was previously discarded, skipping processing")
                
                # Ensure the discarded placeholder is in output_data
                if not any(t.get('id') == topic_id for t in output_data):
                    output_data.append(existing_entry)
                    
                return True
                
            # If already processed normally, use that version
            logger.info(f"Topic ID {topic_id} already processed, using existing version")
            if not any(t.get('id') == topic_id for t in output_data):
                output_data.append(existing_entry)
                
            return True
            
        # Skip topics without images
        if not topic.get('img_paths') or len(topic.get('img_paths', [])) == 0:
            logger.debug(f"Topic has no images, keeping as is: {topic.get('topic', '')[:50]}...")
            output_data.append(topic)
            return True

        # Process with GPT-4o
        logger.info(f"Processing topic ID {topic_id}: {topic.get('topic', '')[:50]}...")
        processed_result = process_topic_images(client, topic)
        
        if processed_result:
            # Update topic data - note that this can return a placeholder if topic should not be kept
            updated_topic = update_topic_data(topic.copy(), processed_result)
            
            # Add to output data regardless of whether it's a full topic or placeholder
            output_data.append(updated_topic)
            
            # Update processed IDs index
            if 'id' in updated_topic:
                processed_ids[updated_topic['id']] = updated_topic
                
            if updated_topic.get('discarded'):
                logger.info(f"Topic ID {topic_id} removed but placeholder preserved")
            else:
                logger.info(f"Successfully processed topic ID {topic_id} with {len(updated_topic.get('img_paths', []))} images")
                
            return True
        else:
            # On processing failure, keep original data
            logger.warning(f"Processing failed for topic ID {topic_id}, keeping original topic: {topic.get('topic', '')[:50]}...")
            output_data.append(topic)
            return True
    
    except Exception as e:
        logger.error(f"Error processing topic ID {topic.get('id', 'unknown')}: {e}")
        # Keep original data on error
        output_data.append(topic)
        return False