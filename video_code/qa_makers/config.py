import os

BASE_DIR = ""
DATA_DIR = os.path.join(BASE_DIR, "data/raw_data")
print(f"config.py:DATA_DIR: {DATA_DIR}")
LOG_DIR = os.path.join(BASE_DIR, "logs")

CONFIG = {
    "api_key": "",
    "model": "gpt-4.1",
    "max_workers": 8,  
    "temperature": 0.7,  
    "max_tokens": 2000
}
