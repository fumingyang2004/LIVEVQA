"""Duplicate article checking module"""
import os
import re
import glob
import json
from collectors.config import WORKSPACE, STOPWORDS, DATA_DIR
from collectors.utils import get_title_keywords, calculate_title_similarity
from collectors.utils_display import print_info, print_warning, print_progress

class DuplicateChecker:
    """Article deduplication class, responsible for checking if newly collected articles duplicate existing ones"""
    
    def __init__(self):
        self.output_dir = os.path.join(WORKSPACE, "data", "raw_data")
        self.existing_topics = []
        self.seen_titles = set()  # Titles seen in the current session
        self.seen_urls = set()    # URLs seen in the current session
    
    def load_existing_topics(self):
        """Loads topics from all existing hot_topics files"""
        try:
            # Find all JSON files starting with hot_topics
            hot_files = glob.glob(os.path.join(self.output_dir, "hot_topics*.json"))
            total_files = len(hot_files)
            
            if not hot_files:
                print_warning("No topic files found, starting collection from scratch")
                return []
                
            print_info(f"Found {total_files} topic files")
            
            # Set of URLs for deduplication
            loaded_urls = set()
            
            # Display progress
            for i, file_path in enumerate(hot_files):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            topics = json.load(f)
                            if topics and isinstance(topics, list):
                                # Remove entries with duplicate URLs
                                for topic in topics:
                                    url = topic.get('url', '')
                                    if url and url not in loaded_urls:
                                        loaded_urls.add(url)
                                        self.existing_topics.append(topic)
                                        # Also add title to seen set
                                        if 'topic' in topic and topic['topic']:
                                            self.seen_titles.add(topic['topic'])
                        except json.JSONDecodeError as e:
                            print_warning(f"JSON parsing error {file_path}: {e}")
                            continue
                    
                    # Update progress
                    print_progress(i+1, total_files, prefix="Loading historical data:", 
                                   suffix=f"({i+1}/{total_files}) - {os.path.basename(file_path)}")
                
                except Exception as e:
                    print_warning(f"Error reading file {file_path}: {e}")
            
            print_info(f"Successfully loaded {len(self.existing_topics)} historical topics")
            
            # Pre-populate URL set to speed up subsequent deduplication
            self.seen_urls = loaded_urls
            
        except Exception as e:
            print_warning(f"Error loading historical topics: {e}")
            
        return self.existing_topics
    
    def is_duplicate_topic(self, new_topic):
        """Checks if a new topic duplicates an existing topic"""
        if not new_topic or 'topic' not in new_topic or not new_topic['topic']:
            return False, None
        
        new_title = new_topic['topic']
        new_url = new_topic.get('url', '')
        
        # URL quick check
        if new_url and new_url in self.seen_urls:
            # Find corresponding topic to return
            for existing in self.existing_topics:
                if existing.get('url') == new_url:
                    return True, existing
            # If no specific corresponding topic found, but URL exists, it's still a duplicate
            return True, None
        
        # Title quick check
        if new_title in self.seen_titles:
            # Find corresponding topic
            for existing in self.existing_topics:
                if existing.get('topic') == new_title:
                    return True, existing
            # If no specific corresponding topic found, but title exists, it's still a duplicate
            return True, None
            
        # Extract title information
        new_title_lower = new_title.lower()
        new_title_words = new_title_lower.split()
        new_title_start = ' '.join(new_title_words[:5]) if len(new_title_words) >= 3 else ""
        new_keywords = get_title_keywords(new_title, STOPWORDS)
        
        if len(new_keywords) < 3 and len(new_title_words) < 5:  # Too few keywords, unreliable
            if not new_url:  # If no URL, cannot determine if duplicate
                return False, None
        
        # Optimized loop to check for duplicates
        for existing in self.existing_topics:
            if 'topic' not in existing:
                continue
                
            existing_title = existing['topic']
            existing_url = existing.get('url', '')
            
            # Check method 1: Exact URL match
            if new_url and existing_url and new_url == existing_url:
                self.seen_urls.add(new_url)  # Add to seen URL set
                return True, existing
                
            # Check method 2: Exact match of first 5 words
            if new_title_start:
                existing_words = existing_title.lower().split()
                existing_start = ' '.join(existing_words[:5]) if len(existing_words) >= 3 else ""
                if existing_start and new_title_start == existing_start:
                    self.seen_titles.add(new_title)  # Add to seen title set
                    return True, existing
            
            # Check method 3: Keyword overlap
            if len(new_keywords) >= 3:
                existing_keywords = get_title_keywords(existing_title, STOPWORDS)
                if len(existing_keywords) >= 3:
                    overlap = new_keywords.intersection(existing_keywords)
                    min_keywords = min(len(new_keywords), len(existing_keywords))
                    if len(overlap) >= min_keywords * 0.7:
                        self.seen_titles.add(new_title)  # Add to seen title set
                        return True, existing
            
            # Check method 4: Title similarity (only calculate if previous checks missed)
            similarity = calculate_title_similarity(new_title, existing_title)
            if similarity > 0.85:  # Increase threshold to reduce false positives
                self.seen_titles.add(new_title)  # Add to seen title set
                return True, existing
                
        # If all checks passed, add to seen sets
        if new_url:
            self.seen_urls.add(new_url)
        self.seen_titles.add(new_title)
        
        return False, None
    
    def is_duplicate_realtime(self, title, url):
        """Real-time checks if an article is a duplicate (simplified deduplication for crawler internal use)"""
        if not title:
            return True
        
        # URL quick check - most efficient check
        if url and url in self.seen_urls:
            return True
            
        # Title quick check - exact match
        if title in self.seen_titles:
            return True
            
        # Clean extra whitespace from title
        clean_title = re.sub(r'\s{2,}', ' ', title.strip())
        clean_title_lower = clean_title.lower()
        
        # Check for match of first 5 words
        title_words = clean_title_lower.split()
        title_start = ' '.join(title_words[:5]) if len(title_words) >= 3 else ""
        
        # Extract keywords
        keywords = get_title_keywords(clean_title, STOPWORDS)
        keyword_count = len(keywords)
        
        # If too few keywords, only check URL and exact title match
        if keyword_count < 3 and len(title_words) < 5:
            # URL and exact title already checked, no extra action needed here
            # In this case, add to seen sets
            self.seen_titles.add(title)
            if url:
                self.seen_urls.add(url)
            return False
        
        # Quick check against existing topics
        for existing in self.existing_topics:
            if 'topic' not in existing or not existing['topic']:
                continue
                
            existing_title = existing['topic']
            existing_url = existing.get('url', '')
            
            # URL match
            if url and existing_url and url == existing_url:
                return True
            
            # Title prefix exact match
            if title_start:
                existing_words = existing_title.lower().split()
                existing_start = ' '.join(existing_words[:5]) if len(existing_words) >= 3 else ""
                if existing_start and title_start == existing_start:
                    return True
            
            # Title keyword match - only calculate if enough keywords
            if keyword_count >= 3:
                existing_keywords = get_title_keywords(existing_title, STOPWORDS)
                if len(existing_keywords) >= 3:
                    overlap = keywords.intersection(existing_keywords)
                    if len(overlap) >= min(keyword_count, len(existing_keywords)) * 0.7:
                        return True
            
            # Title similarity - most resource-intensive, check last
            if calculate_title_similarity(clean_title, existing_title) > 0.85:
                return True
        
        # If all checks passed, add to seen sets and return non-duplicate
        self.seen_titles.add(title)
        if url:
            self.seen_urls.add(url)
        return False
    
    def add_seen_title(self, title, url=None):
        """Adds title and URL to the seen sets"""
        if title:
            self.seen_titles.add(title)
        if url:
            self.seen_urls.add(url)
