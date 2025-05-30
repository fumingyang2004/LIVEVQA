"""Progress monitoring and logging utility functions"""
import threading
import time
import os
from datetime import datetime

class ProgressMonitor:
    """Progress monitoring class"""
    def __init__(self, collector, interval=10):
        """Initializes the progress monitor"""
        self.collector = collector
        self.interval = interval
        self.stop_flag = threading.Event()
        self.thread = None
        self.last_count = 0
        
    def _monitor(self):
        """Monitoring thread function"""
        while not self.stop_flag.is_set():
            if hasattr(self.collector, 'all_topics'):
                current_count = len(self.collector.all_topics)
                if current_count != self.last_count:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetched valid news: {current_count} articles")
                    self.last_count = current_count
            time.sleep(self.interval)
            
    def start(self):
        """Starts monitoring"""
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self
        
    def stop(self):
        """Stops monitoring"""
        self.stop_flag.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(1)  # Wait for at most 1 second

def get_random_headers(user_agents):
    """Gets request headers with a random user agent"""
    import random
    return {'User-Agent': random.choice(user_agents)}
