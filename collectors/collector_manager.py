"""Topic collection manager module"""
import os
import time
import glob
import json
import shutil
from datetime import datetime

from collectors.config import DATA_DIR
from collectors.duplicate_checker import DuplicateChecker
from collectors.utils_display import Spinner, print_warning, print_info, print_subheader, print_error
from collectors.directory_creator import create_project_directories
from collectors.utils_date import get_mmddhhmi_timestamp

from collectors.managers.base_manager import BaseManager
from collectors.managers.data_manager import DataManager
from collectors.managers.task_manager import TaskManager
from collectors.managers.logging_manager import LoggingManager

class TopicCollectorManager:
    """Hot topic collection manager class, responsible for coordinating crawlers and processing results"""
    
    def __init__(self):
        """Initializes the hot topic collection manager"""
        # Initialize results list
        self.all_topics = []
        self.start_time = time.time()
        
        # Initialize various manager components
        self.base_manager = BaseManager()
        self.data_manager = DataManager(self.base_manager)
        self.task_manager = TaskManager(self.base_manager)
        self.logging_manager = LoggingManager(self.base_manager)
        
        # Deduplication checker
        self.duplicate_checker = None
        
        # Current running output file path
        self.current_output_file = None
    
    def setup(self, verbose=True, quiet=False):
        """Sets up the collector, loads historical data"""
        # Ensure directories exist
        create_project_directories()
        
        # Set output verbosity
        self.base_manager.setup(verbose=verbose, quiet=quiet)
        
        # Create new output file
        timestamp = get_mmddhhmi_timestamp()
        self.current_output_file = os.path.join(DATA_DIR, f"hot_topics_{timestamp}.json")
        
        # Initialize deduplication checker
        spinner = Spinner("Loading historical topic data...").start()
        self.duplicate_checker = DuplicateChecker()
        total_topics = self.duplicate_checker.load_existing_topics()
        
        spinner.stop(f"Loading complete, {len(total_topics)} historical topics")
        
        # Create an empty output file
        with open(self.current_output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)
    
    def add_topic_realtime(self, topic):
        """Adds non-duplicate topics to the collection results"""
        if not topic or 'topic' not in topic:
            return False
            
        title = topic['topic'].strip()
        url = topic.get('url', '')
        
        # Use deduplication checker to check for duplicates
        if self.duplicate_checker.is_duplicate_realtime(title, url):
            if self.base_manager.verbose:
                print_info(f"[Real-time Deduplication] Skipping duplicate: {title[:50]}...")
            return False
            
        # Record title to deduplication checker
        self.duplicate_checker.add_seen_title(title)
        
        # Process topic data
        updated_topic = self.data_manager.process_topic(topic)
        
        # ID is already generated in process_topic via _get_next_id method
        
        # Add to results list
        self.all_topics.append(updated_topic)
        
        # Save to timestamped file in real-time
        self._save_to_timestamped_file_realtime(updated_topic)
        
        return True
    
    def _save_to_timestamped_file_realtime(self, topic):
        """Saves to a timestamped file in real-time"""
        if not self.current_output_file:
            timestamp = get_mmddhhmi_timestamp()
            self.current_output_file = os.path.join(DATA_DIR, f"hot_topics_{timestamp}.json")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.current_output_file), exist_ok=True)
            
            # File locking strategy - use temporary file and atomic operation
            temp_file = f"{self.current_output_file}.temp.{os.getpid()}"
            
            # Create primary backup file - backup the entire file before each update
            primary_backup = f"{self.current_output_file}.primary_bak"
            
            # Read current file content
            current_data = []
            original_count = 0
            if os.path.exists(self.current_output_file) and os.path.getsize(self.current_output_file) > 0:
                # Create primary backup before reading data
                try:
                    shutil.copy2(self.current_output_file, primary_backup)
                except Exception as e:
                    print_warning(f"Failed to create primary backup file: {e}")
                
                # Retry reading multiple times to prevent file being written by other processes
                retry_count = 3
                for attempt in range(retry_count):
                    try:
                        with open(self.current_output_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content:
                                current_data = json.loads(content)
                                if not isinstance(current_data, list):
                                    current_data = []
                        break  # Successfully read, exit loop
                    except json.JSONDecodeError:
                        if attempt < retry_count - 1:
                            # Retry after a short delay
                            import time
                            time.sleep(0.1)
                        else:
                            # Last attempt failed, try to restore from backup
                            print_warning(f"Failed to parse JSON file, attempting to restore from backup")
                            if os.path.exists(primary_backup):
                                try:
                                    with open(primary_backup, 'r', encoding='utf-8') as f:
                                        backup_content = f.read()
                                        current_data = json.loads(backup_content) if backup_content else []
                                    print_info(f"Successfully restored data from primary backup")
                                except Exception as backup_error:
                                    print_warning(f"Failed to restore from backup: {backup_error}")
                                    current_data = []
                            else:
                                print_warning(f"Failed to parse JSON file, using empty list and recreating: {self.current_output_file}")
                                current_data = []
                    except Exception as e:
                        print_warning(f"Failed to read file: {e}")
                        current_data = []
                        break
                
                # Record original data count
                original_count = len(current_data)
            
            # Check if an entry with the same URL or ID already exists
            topic_url = topic.get('url', '')
            topic_id = topic.get('id', '')
            is_duplicate = False
            
            # Ensure current_data is a list
            if not isinstance(current_data, list):
                current_data = []
            
            for idx, item in enumerate(current_data):
                if not isinstance(item, dict):
                    continue
                    
                # Check if URL already exists
                if topic_url and item.get('url') == topic_url:
                    # Update existing entry, retain ID
                    if 'id' in item:
                        topic['id'] = item['id']
                    current_data[idx] = topic
                    is_duplicate = True
                    break
                
                # Check if ID already exists
                if topic_id and item.get('id') == topic_id:
                    current_data[idx] = topic
                    is_duplicate = True
                    break
            
            # If not a duplicate, add to the list
            if not is_duplicate:
                current_data.append(topic)
            
            # Ensure data has not decreased
            if len(current_data) < original_count:
                print_warning(f"Data decrease detected: from {original_count} records to {len(current_data)} records, attempting to restore from backup")
                if os.path.exists(primary_backup):
                    try:
                        # Read data from primary backup
                        with open(primary_backup, 'r', encoding='utf-8') as f:
                            backup_data = json.loads(f.read())
                        
                        if len(backup_data) > len(current_data):
                            # Add new topic to backup data
                            found = False
                            for idx, item in enumerate(backup_data):
                                if (topic_url and item.get('url') == topic_url) or \
                                   (topic_id and item.get('id') == topic_id):
                                    backup_data[idx] = topic
                                    found = True
                                    break
                            
                            if not found:
                                backup_data.append(topic)
                            
                            # Replace with restored data
                            current_data = backup_data
                            print_info(f"Successfully restored data from backup and added new entry, total {len(current_data)} records")
                    except Exception as restore_error:
                        print_error(f"Failed to restore from backup: {restore_error}")
            
            # Save to temporary file
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)
            
            # Atomic replacement of original file
            import platform
            if platform.system() == 'Windows':
                # Windows does not support atomic rename to existing file, delete original first
                if os.path.exists(self.current_output_file):
                    os.unlink(self.current_output_file)
            
            os.rename(temp_file, self.current_output_file)
            
            # Verify data count after writing
            if os.path.exists(self.current_output_file):
                try:
                    with open(self.current_output_file, 'r', encoding='utf-8') as f:
                        final_data = json.loads(f.read())
                        if len(final_data) < original_count:
                            print_error(f"Verification failed: Final data ({len(final_data)} records) is less than original data ({original_count} records), restoring backup")
                            if os.path.exists(primary_backup):
                                shutil.copy2(primary_backup, self.current_output_file)
                                print_info(f"Restored original data from backup")
                            return False
                except Exception as verify_err:
                    print_warning(f"Failed to verify written data: {verify_err}")
            
            return True
            
        except Exception as e:
            print_warning(f"Failed to save to file in real-time: {e}")
            
            # Attempt to restore from primary backup
            primary_backup = f"{self.current_output_file}.primary_bak"
            if os.path.exists(primary_backup) and os.path.exists(self.current_output_file):
                try:
                    shutil.copy2(primary_backup, self.current_output_file)
                except Exception as restore_err:
                    print_error(f"Failed to restore from backup: {restore_err}")
            
            # Clean up temporary file
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
            return False
    
    def process_articles_batch(self, articles):
        """
        Processes a batch of articles, checks for duplicates, and adds to results
        
        Args:
            articles: List of articles
            
        Returns:
            int: Number of articles added
        """
        added_count = 0
        for article in articles:
            if 'topic' in article:
                updated_article = self.data_manager.process_topic(article)
                
                if not self.duplicate_checker.is_duplicate_realtime(
                    updated_article.get('topic', ''), 
                    updated_article.get('url', '')
                ):
                    self.duplicate_checker.add_seen_title(updated_article.get('topic', ''))
                    
                    # ID is already generated in process_topic
                    
                    self.all_topics.append(updated_article)
                    self._save_to_timestamped_file_realtime(updated_article)
                    added_count += 1
        
        return added_count
    
    def collect_all_topics(self):
        """Collects hot topics from all sources"""
        # Initialize results list
        self.all_topics = []
        
        # Log start of collection
        self.logging_manager.log_section_header("Starting hot topic collection")
        
        # Get all crawler tasks
        tasks = self.task_manager.get_collection_tasks(self)
        
        if not tasks:
            print_warning("No crawlers enabled, please check ENABLE_* settings in config file")
            return []
        
        # Execute all tasks
        self.task_manager.execute_tasks(tasks, self)
        
        return self.all_topics
    
    def save_to_file(self, output_file, topics):
        """
        Saves all hot topics to a single JSON file
        
        Args:
            output_file: Output file path
            topics: List of topics to be saved
            
        Returns:
            str: Path to the successfully saved file, None if failed
        """
        # If real-time saving has already saved all content, no need to save again here
        # Just update the file path reference
        self.current_output_file = output_file
        
        # Still call the original save method to ensure integrity
        return self.data_manager.save_to_file(output_file, topics)
    
    def log_collection_result(self, output_file, start_time):
        """
        Logs the results of hot topic collection
        
        Args:
            output_file: Output file path
            start_time: Start time
        """
        self.logging_manager.log_collection_result(self.all_topics, output_file, start_time)
