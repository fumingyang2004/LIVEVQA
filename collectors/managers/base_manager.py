"""Base manager, providing shared functionality and attributes"""
import time
import os
from collectors.config import WORKSPACE, DATA_DIR, IMG_DIR

class BaseManager:
    """Base manager class, providing functionality shared by all managers"""
    
    def __init__(self):
        """Initializes the base manager"""
        self.start_time = time.time()
        self.log_dir = os.path.join(WORKSPACE, "logs")
        self.data_dir = DATA_DIR
        self.img_dir = IMG_DIR
        self.verbose = True
        self.quiet = False
        
        # List of user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        ]
    
    def setup(self, verbose=True, quiet=False):
        """Sets up basic parameters"""
        self.verbose = verbose
        self.quiet = quiet
        return self
