
import os

BASE_DIR = "YOUR BASE DIRECTORY HERE"  # Replace with your actual base directory
DATA_DIR = os.path.join(BASE_DIR, "data/raw_data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

CONFIG = {
    "api_key": "YOUR API KEY HERE",
    "model": "YOUR MODEL HERE", 
    "max_workers": 8, 
    "temperature": 0.7,  
    "max_tokens": 2000
}

