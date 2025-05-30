import os
# API configuration
API_KEY = "YOUR_API_KEY"
API_MODEL = "YOUR_API_MODEL"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Base directories
BASE_DIR = "YOUR BASE PATH"  # Replace with your actual base path
DATA_DIR = os.path.join(BASE_DIR, "data/raw_data")
STATS_DIR = os.path.join(BASE_DIR, "data/test_set")
LOG_DIR = os.path.join(BASE_DIR, "logs")