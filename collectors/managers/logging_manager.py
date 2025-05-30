"""Logging module, responsible for recording various logs and statistics"""
import os
from datetime import datetime, timedelta

from collectors.config import (
    ENABLE_CNN, ENABLE_VARIETY, ENABLE_BBC, 
    ENABLE_FORBES, ENABLE_APNEWS
)

class LoggingManager:
    """Logging manager, records logs of the collection process and results"""
    
    def __init__(self, base_manager):
        """
        Initializes the logging manager
        
        Args:
            base_manager: Base manager instance, providing configuration and shared functionality
        """
        self.base_manager = base_manager
        self.log_dir = base_manager.log_dir
        self.data_dir = base_manager.data_dir
        self.img_dir = base_manager.img_dir
        self.verbose = base_manager.verbose
        self.quiet = base_manager.quiet
    
    def log_section_header(self, message):
        """
        Logs the header of a running section
        
        Args:
            message: The message to log
        """
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, "hot_topics_log.txt")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
            f.write(f"{'='*60}\n")
    
    def log_collection_result(self, all_topics, output_file, start_time):
        """
        Logs the results of hot topic collection
        
        Args:
            all_topics: All collected topics
            output_file: Output file path
            start_time: Start time
        """
        # Calculate elapsed time
        elapsed_time = datetime.now() - start_time
        minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
        
        # Check current mode
        full_day_mode = getattr(self.base_manager, 'full_day_mode', False)
        target_date_desc = getattr(self.base_manager, 'target_date_desc', "")
        
        # Create run result summary
        result_summary = [
            f"Run Time: {int(minutes)}m {int(seconds)}s",
            f"Total Topics Collected: {len(all_topics)}",
            f"Crawl Mode: {'Full-day crawl for ' + target_date_desc if full_day_mode else 'Standard crawl'}",
            f"Output File: {output_file}",
            f"Image Save Location: {self.img_dir}",
            f"Data Source Statistics:",
        ]
        
        # Display statistics for each data source based on configuration
        source_stats = [
            (ENABLE_CNN, 'CNN', 'cnn.com'),
            (ENABLE_VARIETY, 'Variety', 'variety.com'),
            (ENABLE_BBC, 'BBC', 'bbc.'),
            (ENABLE_FORBES, 'Forbes', 'forbes.com'),
            (ENABLE_APNEWS, 'AP News', 'apnews.com')
        ]
        
        for enabled, name, domain in source_stats:
            if enabled:
                count = sum(1 for t in all_topics if domain in t.get('url', ''))
                result_summary.append(f"   - {name}: {count} articles")
            
        result_summary.append(f"Category Statistics:")
        
        # Add category statistics (no longer used)
        categories = {}
        for t in all_topics:
            cat = t.get('category', 'unknown') if 'category' in t else 'unknown'
            categories[cat] = categories.get(cat, 0) + 1
            
        for cat, count in categories.items():
            result_summary.append(f"   - {cat}: {count} articles")
        
        # Write summary to log file
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, "hot_topics_results.txt")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Hot Topic Collection Results\n")
            f.write(f"{'='*60}\n")
            for line in result_summary:
                f.write(f"{line}\n")
            f.write(f"{'='*60}\n")
        
        # If not in quiet mode, print result summary
        from collectors.utils_display import print_subheader, print_info
        if not self.quiet:
            if self.verbose:
                print_subheader("Collection results logged to file")
                print_info(f"Log file: {log_file}")
