"""Data management module, responsible for data processing, saving, and deduplication"""
import os
import re
import json
import shutil
import logging
from datetime import datetime

from collectors.utils import download_image
from collectors.utils_display import print_info, print_warning, print_error, print_success
from collectors.json_utils import safe_read_json, safe_write_json, verify_json_integrity, repair_json_file, append_to_json_array

# Set up logging
logger = logging.getLogger(__name__)

class DataManager:
    """Data manager, handles storage and processing of topic collection results"""
    
    def __init__(self, base_manager):
        """
        Initializes the data manager
        
        Args:
            base_manager: Base manager instance, providing configuration and shared functionality
        """
        self.base_manager = base_manager
        self.data_dir = base_manager.data_dir
        self.img_dir = base_manager.img_dir
        self.verbose = base_manager.verbose
        self.quiet = base_manager.quiet
        self.current_max_id = -1  # Retain this variable for compatibility with existing code, but no longer used for ID generation
    
    def _get_next_id(self):
        """
        Gets the next available ID, using the format: 0_YYYYMMDDHHMMSSffffff
        
        Returns:
            str: A unique ID based on timestamp
        """
        # Use current time to generate unique ID
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        microseconds = f"{current_time.microsecond:06d}"
        
        # Generate ID in the format 0_YYYYMMDDHHMMSSffffff
        new_id = f"0_{timestamp}{microseconds}"
        
        return new_id
    
    def _load_current_max_id(self):
        """
        Loads the maximum ID from the existing hot_topics.json file
        This method is retained for compatibility with existing code, but no longer used for ID generation
        
        Returns:
            int: Current maximum ID, or -1 if file does not exist or is empty
        """
        # Retain this method for compatibility, but it's no longer actually needed
        self.current_max_id = 0
        return self.current_max_id
    
    def save_to_file_realtime(self, output_file, topic):
        """
        Real-time updates a single topic to a JSON file, adding a safe saving mechanism
        
        Args:
            output_file: Output file path
            topic: Topic data to be saved
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Record pre-operation status
            original_data_exists = os.path.exists(output_file)
            original_file_size = os.path.getsize(output_file) if original_data_exists else 0
            
            # If file exists but format is incorrect, try to repair
            if original_data_exists:
                if not verify_json_integrity(output_file):
                    logger.warning(f"JSON file format error, attempting to repair: {output_file}")
                    if not repair_json_file(output_file):
                        print_warning(f"JSON file repair failed, a new file will be created")
                        # Create backup
                        backup_file = output_file + f".corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        try:
                            shutil.copy2(output_file, backup_file)
                        except Exception as e:
                            print_warning(f"Failed to create backup of corrupted file: {e}")
            
            # If file exists, load existing data and create backup
            existing_data = []
            if original_data_exists:
                # Create backup file
                backup_file = output_file + ".bak"
                try:
                    shutil.copy2(output_file, backup_file)
                except Exception as e:
                    print_warning(f"Failed to create backup file: {e}")
                    logger.warning(f"Failed to create backup file: {e}")
                    
                # Read existing data
                existing_data = safe_read_json(output_file, [])
                
                # Verify read success
                if not existing_data and original_file_size > 10:  # Ensure original file is not empty and read result is empty
                    print_warning(f"Failed to read existing JSON data, attempting to restore from backup")
                    if os.path.exists(backup_file):
                        existing_data = safe_read_json(backup_file, [])
                        if existing_data:
                            print_info(f"Successfully restored data from backup, containing {len(existing_data)} records")
                        else:
                            print_error(f"Failed to restore data from backup")
                            return False
                    else:
                        print_error(f"No backup file available")
                        return False
            
            # Ensure topic data is valid
            if not topic or not isinstance(topic, dict):
                print_warning("Attempting to save invalid topic data, skipping")
                return False
                
            # Clean topic data from values that might cause issues
            cleaned_topic = self._clean_json_object(topic)
            
            # Process into new format
            cleaned_topic = self._convert_to_new_format(cleaned_topic)
            
            # Record data length for verification
            original_count = len(existing_data)
            
            # Check if an entry with the same URL already exists
            found_existing = False
            for idx, item in enumerate(existing_data):
                if item.get('url') == cleaned_topic.get('url'):
                    # Retain original ID
                    if 'id' in item:
                        cleaned_topic['id'] = item['id']
                    # Update existing entry
                    existing_data[idx] = cleaned_topic
                    found_existing = True
                    break
            
            # If no entry with the same URL was found, add a new entry and assign a new ID
            if not found_existing:
                # Ensure topic has an ID field
                if 'id' not in cleaned_topic:
                    cleaned_topic['id'] = self._get_next_id()
                existing_data.append(cleaned_topic)
            
            # Verify data integrity
            if len(existing_data) < original_count:
                print_error(f"Data loss risk detected! Original data: {original_count} records, new data: {len(existing_data)} records")
                logger.error(f"Data loss risk! Original: {original_count}, New: {len(existing_data)}")
                return False
                
            # Use safe write method, instead of direct append
            write_success = safe_write_json(output_file, existing_data)
            
            if not write_success:
                print_error(f"Failed to write data")
                return False
                
            # Perform post-write verification
            if verify_json_integrity(output_file, min_expected_items=original_count):
                if self.verbose:
                    print_info(f"Data updated in real-time to {output_file}")
                return True
            else:
                print_error(f"Write verification failed, attempting to restore backup")
                # Attempt to restore from backup
                backup_file = output_file + ".bak"
                if os.path.exists(backup_file):
                    try:
                        shutil.copy2(backup_file, output_file)
                        print_info("Successfully restored data from backup")
                    except Exception as restore_error:
                        print_error(f"Failed to restore from backup: {restore_error}")
                return False
                
        except Exception as e:
            print_error(f"Failed to save data in real-time: {e}")
            logger.exception(f"Real-time data saving exception: {e}")
            # Attempt to restore from backup
            backup_file = output_file + ".bak"
            if os.path.exists(backup_file):
                try:
                    shutil.copy2(backup_file, output_file)
                    print_info("Successfully restored data from backup")
                except Exception as restore_error:
                    print_error(f"Failed to restore from backup: {restore_error}")
            return False

    def save_to_file(self, output_file, topics):
        """
        Saves all hot topics to a single JSON file, adding a safe write mechanism
        and ensuring existing data is retained
        
        Args:
            output_file: Output file path
            topics: List of topics to be saved
            
        Returns:
            str: Path to the successfully saved file, None if failed
        """
        print_info(f"Saving to {output_file}...")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Record pre-operation status
        original_data_exists = os.path.exists(output_file)
        original_file_size = os.path.getsize(output_file) if original_data_exists else 0

        # Read existing data (if any)
        existing_data = safe_read_json(output_file, [])
        
        # Verify read success
        if original_data_exists and not existing_data and original_file_size > 10:
            print_warning(f"Failed to read existing JSON data, attempting to restore from backup")
            backup_file = output_file + ".bak"
            if os.path.exists(backup_file):
                existing_data = safe_read_json(backup_file, [])
                if existing_data:
                    print_info(f"Successfully restored data from backup, containing {len(existing_data)} records")
                else:
                    print_error(f"Failed to restore data from backup")
                    return None
            else:
                print_error(f"No backup file available, unable to load existing data")
        
        # Record original data size
        original_count = len(existing_data)
        
        # Create backup file
        if os.path.exists(output_file) and len(existing_data) > 0:
            backup_file = output_file + ".bak"
            try:
                shutil.copy2(output_file, backup_file)
            except Exception as e:
                print_warning(f"Failed to create backup file: {e}")
                logger.warning(f"Failed to create backup file: {e}")
        
        # Clean new data and convert to new format
        cleaned_topics = []
        for topic in topics:
            if topic:
                cleaned_topic = self._clean_json_object(topic)
                cleaned_topic = self._convert_to_new_format(cleaned_topic)
                cleaned_topics.append(cleaned_topic)
        
        # Merge data, avoid duplicates, and handle IDs
        merged_data = self._merge_topics_without_duplicates(existing_data, cleaned_topics)
        
        # Verify data integrity, ensuring no data loss
        if len(merged_data) < original_count:
            print_error(f"Potential data loss detected! Original {original_count} records, merged only {len(merged_data)} records")
            logger.error(f"Data loss risk! Original: {original_count}, Merged: {len(merged_data)}")
            # Do not proceed, return failure
            return None
        
        # Use safe write method
        if safe_write_json(output_file, merged_data):
            # Post-write verification
            if verify_json_integrity(output_file, min_expected_items=original_count):
                print_success(f"Saved {len(merged_data)} records to {output_file} (added {len(cleaned_topics)} new records)")
                return output_file
            else:
                print_error(f"Post-save verification failed, attempting to restore from backup")
                # Attempt to restore
                backup_file = output_file + ".bak"
                if os.path.exists(backup_file):
                    try:
                        shutil.copy2(backup_file, output_file)
                        print_info("Successfully restored data from backup")
                    except Exception as restore_error:
                        print_error(f"Failed to restore from backup: {restore_error}")
                return None
        else:
            print_error(f"Failed to save data")
            # Attempt to restore from backup
            backup_file = output_file + ".bak"
            if os.path.exists(backup_file):
                try:
                    shutil.copy2(backup_file, output_file)
                    print_info("Successfully restored data from backup")
                except Exception as restore_error:
                    print_error(f"Failed to restore from backup: {restore_error}")
            return None

    def _convert_to_new_format(self, topic):
        """Converts old format topics to new format
        
        Args:
            topic: Original topic data
            
        Returns:
            dict: Converted topic data
        """
        # If already in new format, return directly
        if 'img_urls' in topic and isinstance(topic['img_urls'], list):
            # Ensure captions and img_paths are synchronized
            if 'captions' not in topic or not isinstance(topic['captions'], list):
                topic['captions'] = ["null"] * len(topic['img_urls'])
                
            if 'img_paths' not in topic:
                topic['img_paths'] = []
                
            # Ensure fields exist
            if 'text' not in topic:
                topic['text'] = ""
                
            return topic
        
        # Create new format topic
        new_topic = {
            'topic': topic.get('topic', ''),
            'text': topic.get('text', ''),
            'img_urls': [],
            'img_paths': [],
            'captions': [],
            'source': topic.get('source', ''),
            'url': topic.get('url', ''),
            'category': topic.get('category', '')
        }
        
        # Handle single image case
        img_url = topic.get('img_url', '')
        if img_url:
            new_topic['img_urls'] = [img_url]
            new_topic['captions'] = ["null"]  # Use "null" as placeholder for no caption
            
            # Handle image path
            img_path = topic.get('img_path', '')
            if img_path:
                new_topic['img_paths'] = [img_path]
        
        return new_topic

    def _clean_json_object(self, obj):
        """
        Cleans values in a JSON object that might cause serialization issues
        
        Args:
            obj: Object to clean
            
        Returns:
            Cleaned object
        """
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k is None:
                    continue
                    
                # Special handling for topic field, ensure correct format
                if k == 'topic' and isinstance(v, str):
                    is_cnn = False
                    if 'source' in obj and isinstance(obj['source'], str):
                        is_cnn = 'CNN' in obj['source']
                    elif 'url' in obj and isinstance(obj['url'], str):
                        is_cnn = 'cnn.com' in obj['url'].lower()
                    
                    if is_cnn:
                        # CNN article special handling - strictly retain only the first line
                        if '\n' in v:
                            result[k] = v.split('\n')[0].strip()
                        else:
                            result[k] = v.strip()
                    else:
                        # Non-CNN articles also clean extra whitespace
                        result[k] = re.sub(r'\s+', ' ', v).strip()
                else:
                    result[k] = self._clean_json_object(v)
            return result
        elif isinstance(obj, list):
            return [self._clean_json_object(item) for item in obj if item is not None]
        elif obj is None:
            return ""
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # Convert other types to string
            return str(obj)

    def _merge_topics_without_duplicates(self, existing_topics, new_topics):
        """
        Merges old and new topic lists, avoiding duplicates and handling IDs
        
        Args:
            existing_topics: Existing topic list
            new_topics: Newly crawled topic list
            
        Returns:
            list: Merged list, without duplicates
        """
        # Use URL and title as unique identifiers
        seen_urls = {topic.get('url', ''): i for i, topic in enumerate(existing_topics) if topic.get('url')}
        seen_titles = {topic.get('topic', ''): i for i, topic in enumerate(existing_topics) if topic.get('topic')}
        
        merged_list = existing_topics.copy()
        added_count = 0
        updated_count = 0
        
        # No longer need to record and synchronize max ID
        
        for new_topic in new_topics:
            url = new_topic.get('url', '')
            title = new_topic.get('topic', '')
            
            if url and url in seen_urls:
                # Update existing entry (replace old data with new data), retain original ID
                index = seen_urls[url]
                if 'id' in merged_list[index]:
                    new_topic['id'] = merged_list[index]['id']
                merged_list[index] = new_topic
                updated_count += 1
            elif title and title in seen_titles and not url:
                # If no URL but title is the same, also update, retain original ID
                index = seen_titles[title]
                if 'id' in merged_list[index]:
                    new_topic['id'] = merged_list[index]['id']
                merged_list[index] = new_topic
                updated_count += 1
            else:
                # Add new entry, assign a new timestamp-based ID
                if 'id' not in new_topic:
                    new_topic['id'] = self._get_next_id()
                merged_list.append(new_topic)
                # Update index
                if url:
                    seen_urls[url] = len(merged_list) - 1
                if title:
                    seen_titles[title] = len(merged_list) - 1
                added_count += 1
        
        if updated_count > 0 and self.verbose:
            print_info(f"Updated {updated_count} existing records")
        if added_count > 0 and self.verbose:
            print_info(f"Added {added_count} new records")
            
        return merged_list
    
    def process_topic(self, topic):
        """
        Processes a single topic, downloads images, etc.
        
        Args:
            topic: Topic data to be processed
            
        Returns:
            dict: Processed topic data
        """
        # Clean topic
        title = topic['topic'].strip()
        url = topic.get('url', '')
        
        # Get source prefix
        source_name = topic.get('source', '').split()[0].lower()  # Extract the first word of the source as prefix
        
        # Check if it's new format or old format
        if 'img_urls' in topic and isinstance(topic['img_urls'], list):
            # Multi-image processing
            from collectors.utils_images import download_multiple_images
            
            img_urls = topic.get('img_urls', [])
            img_captions = topic.get('captions', [])
            
            # Ensure captions length matches img_urls
            if len(img_captions) < len(img_urls):
                img_captions.extend(["null"] * (len(img_urls) - len(img_captions)))
            
            # Limit to processing at most 4 images
            if len(img_urls) > 4:
                img_urls = img_urls[:4]
                img_captions = img_captions[:4]
                
            # Download multiple images
            if img_urls:
                img_paths = download_multiple_images(img_urls, self.img_dir, source_name)
            else:
                img_paths = []
            
            # Ensure text contains image tags
            text = topic.get('text', '')
            if img_urls and not any(f"<img{i+1}>" in text for i in range(len(img_urls))):
                from collectors.utils_images import insert_image_tags
                text = insert_image_tags(text, len(img_urls))
            
            # Update topic structure, using new format
            updated_topic = {
                'topic': title,
                'text': text,
                'img_urls': img_urls,
                'img_paths': img_paths,
                'captions': img_captions,
                'source': topic.get('source', ''),
                'url': url
            }
            
        else:
            # Single image processing - convert to new format
            img_url = topic.get('img_url', '')
            img_path = None
            
            if img_url:
                from collectors.utils import download_image
                img_path = download_image(img_url, self.img_dir, source_name)
                
                # Insert image tag in text
                text = topic.get('text', '')
                if not "<img1>" in text:
                    text = f"<img1>\n\n{text}"
                
                # Update topic structure to new format
                updated_topic = {
                    'topic': title,
                    'text': text,
                    'img_urls': [img_url] if img_url else [],
                    'img_paths': [img_path] if img_path else [],
                    'captions': ["null"] if img_url else [],  # Use "null" as placeholder
                    'source': topic.get('source', ''),
                    'url': url
                }
            else:
                # No image case
                updated_topic = {
                    'topic': title,
                    'text': topic.get('text', ''),
                    'img_urls': [],
                    'img_paths': [],
                    'captions': [],
                    'source': topic.get('source', ''),
                    'url': url
                }
        
        # Ensure the processed topic has an ID
        if 'id' not in updated_topic:
            updated_topic['id'] = self._get_next_id()
        
        return updated_topic
