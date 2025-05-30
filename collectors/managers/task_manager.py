"""Task management module, responsible for creating and executing crawler tasks"""
import inspect
import concurrent.futures
import time
import random

from collectors.config import (
    ENABLE_CNN, ENABLE_VARIETY, ENABLE_BBC, 
    ENABLE_FORBES, ENABLE_APNEWS
)
from collectors.utils_display import (
    print_info, print_warning, print_error,
    print_subheader, Spinner
)
from collectors.variety_collector import scrape_variety_entertainment
from collectors.cnn_collector import scrape_cnn_news
from collectors.forbes_collector import scrape_forbes_news
from collectors.bbc_collector import scrape_bbc_news
from collectors.apnews_collector import scrape_apnews_news

class TaskManager:
    """Task manager, responsible for creating and executing crawler tasks"""
    
    def __init__(self, base_manager):
        """
        Initializes the task manager
        
        Args:
            base_manager: Base manager instance, providing configuration and shared functionality
        """
        self.base_manager = base_manager
        self.user_agents = base_manager.user_agents
        self.verbose = base_manager.verbose
        self.quiet = base_manager.quiet
    
    def get_collection_tasks(self, collector=None):
        """
        Gets all enabled crawler tasks
        
        Args:
            collector: Collector instance, for real-time deduplication
            
        Returns:
            dict: Mapping of task names to task functions
        """
        tasks = {}
        
        # Attempt to add crawlers that support real-time deduplication
        try:
            # Variety crawler
            if ENABLE_VARIETY:
                if len(inspect.signature(scrape_variety_entertainment).parameters) >= 2:
                    tasks['variety'] = lambda: scrape_variety_entertainment(self.user_agents, collector)
                else:
                    tasks['variety'] = lambda: scrape_variety_entertainment(self.user_agents)
                    print_warning("Variety crawler does not support real-time deduplication")
            
            # CNN crawler
            if ENABLE_CNN:
                if len(inspect.signature(scrape_cnn_news).parameters) >= 2:
                    tasks['cnn'] = lambda: scrape_cnn_news(self.user_agents, collector)
                else:
                    tasks['cnn'] = lambda: scrape_cnn_news(self.user_agents)
                    print_warning("CNN crawler does not support real-time deduplication")
            
            # BBC crawler
            if ENABLE_BBC:
                if len(inspect.signature(scrape_bbc_news).parameters) >= 2:
                    tasks['bbc'] = lambda: scrape_bbc_news(self.user_agents, collector)
                else:
                    tasks['bbc'] = lambda: scrape_bbc_news(self.user_agents)
                    print_warning("BBC crawler does not support real-time deduplication")
                    
            # Forbes crawler
            if ENABLE_FORBES:
                tasks['forbes'] = lambda: scrape_forbes_news(self.user_agents, collector)
                
            # AP News crawler
            if ENABLE_APNEWS:
                tasks['apnews'] = lambda: scrape_apnews_news(self.user_agents, collector)
                
        except Exception as e:
            print_error(f"Error setting up crawler tasks: {e}")
            # Use fallback solution
            if ENABLE_VARIETY:
                tasks['variety'] = lambda: scrape_variety_entertainment(self.user_agents)
            if ENABLE_CNN:
                tasks['cnn'] = lambda: scrape_cnn_news(self.user_agents)
            if ENABLE_BBC:
                tasks['bbc'] = lambda: scrape_bbc_news(self.user_agents)
            if ENABLE_FORBES:
                tasks['forbes'] = lambda: scrape_forbes_news(self.user_agents)
            if ENABLE_APNEWS:
                tasks['apnews'] = lambda: scrape_apnews_news(self.user_agents)
        
        return tasks
    
    def execute_tasks(self, tasks, collector):
        """
        Executes all crawler tasks
        
        Args:
            tasks: Dictionary of tasks
            collector: Collector instance
            
        Returns:
            list: All collected topics
        """
        # Display enabled crawlers
        enabled_crawlers = list(tasks.keys())
        print_info(f"Enabled crawlers: {', '.join(enabled_crawlers)}")
            
        # Display crawler run progress
        print_subheader("Starting crawlers")
        pending_crawlers = set(enabled_crawlers)
        completed_crawlers = set()
        
        # Execute all crawler tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_source = {executor.submit(func): source for source, func in tasks.items()}
            
            # Progress monitoring
            spinner = None
            if not self.quiet:
                spinner = Spinner(f"Crawlers running (0/{len(tasks)})...").start()
            
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    # Process return result
                    result = future.result()
                    
                    # If crawler does not support real-time deduplication, manually process returned articles
                    if result and isinstance(result, list) and not any(t in collector.all_topics for t in result):
                        # Check if articles need to be manually added to all_topics
                        added_count = collector.process_articles_batch(result)
                        
                        if self.verbose:
                            print_info(f"{source.capitalize()} crawling completed, added {added_count} non-duplicate articles")
                    else:
                        if self.verbose:
                            print_info(f"{source.capitalize()} crawling completed")
                    
                    # Update progress
                    pending_crawlers.remove(source)
                    completed_crawlers.add(source)
                    
                    if spinner:
                        spinner.stop(f"Crawler progress: {len(completed_crawlers)}/{len(tasks)} completed")
                        if pending_crawlers:
                            spinner = Spinner(f"Crawlers running ({len(completed_crawlers)}/{len(tasks)}), waiting for: {', '.join(pending_crawlers)}").start()
                        
                except Exception as e:
                    print_error(f"{source.capitalize()} crawling failed: {e}")
                    pending_crawlers.remove(source)
        
        # Stop spinner
        if spinner:
            spinner.stop()
        
        # Remove unnecessary fields, retain fields required for new format
        for topic in collector.all_topics:
            if 'is_recent' in topic:
                del topic['is_recent']
            if 'has_image' in topic:
                del topic['has_image']
            # Ensure new format fields exist
            if 'img_urls' not in topic:
                topic['img_urls'] = []
            if 'img_paths' not in topic:
                topic['img_paths'] = []
            if 'captions' not in topic:
                topic['captions'] = []
                
        # Display collection summary
        if not self.quiet:
            print_subheader("Data collection completed")
            print_info(f"Total valid articles collected: {len(collector.all_topics)}")
        
        return collector.all_topics
