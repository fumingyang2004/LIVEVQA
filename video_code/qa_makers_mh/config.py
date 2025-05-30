import os

# Base path
BASE_DIR = "/mnt/nvme1/fmy/LiveVQApro"
DATA_DIR = os.path.join(BASE_DIR, "data/raw_data")

# Save frequency
SAVE_INTERVAL = 2  # Save results every X items processed

# GPT configuration
API_KEY = ""
MODEL_NAME = ""
MAX_WORKERS = 8  # Maximum number of threads
MAX_RETRIES = 3  # Maximum API call retries
RETRY_DELAY = 2  # API call retry interval (seconds)