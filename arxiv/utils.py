import json
import logging
import os
import re
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('arxiv_scraper.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_logger(name):
    """Return a logger with the given name."""
    return logging.getLogger(name)

def save_json(data, filepath):
    """Save data as JSON to the given filepath."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False

def load_json(filepath):
    """Load JSON data from the given filepath."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return None

def make_request(url, method="GET", headers=None, params=None, data=None, 
                timeout=30, max_retries=3, delay=1):
    """Make an HTTP request with retries."""
    headers = headers or {}
    params = params or {}
    
    for attempt in range(max_retries):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                timeout=timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Request failed after {max_retries} attempts: {url}")
                raise

def sanitize_filename(filename):
    """Sanitize a string to be used as a filename."""
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:197] + "..."
    return sanitized

def get_paper_id_from_url(url):
    """Extract paper ID from arXiv URL."""
    match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9v.]+)', url)
    if match:
        return match.group(1)
    return None

def format_date(date_str):
    """Format date string as YYYY-MM-DD."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d")
    except:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except:
            return date_str 