import os

# Base path
BASE_DIR = "YOUR_BASE_PATH_HERE"  # Replace with your actual base path
DATA_DIR = os.path.join(BASE_DIR, "data/raw_data")

# Save frequency
SAVE_INTERVAL = 2  # Save results every X items processed

# GPT configuration
API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "YOUR_MODEL_NAME_HERE"
MAX_WORKERS = 8  # Maximum number of threads
MAX_RETRIES = 3  # Maximum API call retries
RETRY_DELAY = 2  # API call retry interval (seconds)