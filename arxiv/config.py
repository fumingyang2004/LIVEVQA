import os
from pathlib import Path

# Base directories
BASE_DIR = Path("/mnt/nvme1/fmy/LIVEVQA/arxiv")
DATA_DIR = BASE_DIR / "data"

# Raw data directories
RAW_DIR = DATA_DIR / "raw"
RAW_JSON_DIR = RAW_DIR / "json"
RAW_HTML_DIR = RAW_DIR / "html"
RAW_IMAGES_DIR = RAW_DIR / "images"

# Processed data directories
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_JSON_DIR = PROCESSED_DIR / "json"
PROCESSED_HTML_DIR = PROCESSED_DIR / "html"
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"

# Create directories if they don't exist
for directory in [
    RAW_JSON_DIR, RAW_HTML_DIR, RAW_IMAGES_DIR,
    PROCESSED_JSON_DIR, PROCESSED_HTML_DIR, PROCESSED_IMAGES_DIR
]:
    os.makedirs(directory, exist_ok=True)

# ArXiv API settings
MAX_RESULTS_PER_QUERY = 100
CATEGORIES = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", 
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", 
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", 
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", 
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]

# Download settings
REQUEST_TIMEOUT = 30  # seconds
RETRY_ATTEMPTS = 3
DELAY_BETWEEN_REQUESTS = 1  # seconds 
