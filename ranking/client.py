"""
Client module for interacting with OpenAI API
"""

from openai import OpenAI
import logging
import time 
import threading
from ranking.config import CONFIG

logger = logging.getLogger(__name__)

# Add thread-local storage to ensure each thread has its own client instance
_thread_local = threading.local()

def setup_client():
    """Set up OpenAI client"""
    return OpenAI(api_key=CONFIG["api_key"])

def get_thread_client():
    """Retrieves the thread-local OpenAI client instance.
    
    Ensures each thread has its own independent client instance to avoid conflicts.
    """
    if not hasattr(_thread_local, "client"):
        _thread_local.client = setup_client()
    return _thread_local.client

def call_gpt(client, messages):
    """Make a call to OpenAI GPT API
    
    Args:
        client: OpenAI client instance
        messages: List of message objects
        
    Returns:
        Response from OpenAI API
    """
    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            messages=messages,
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
        )
        return response
    except Exception as e:
        logger.error(f"API call error: {str(e)}")
        return None
        
def call_gpt_with_retry(client, messages, max_retries=3, retry_delay=2):
    """Calls the OpenAI API with a retry mechanism.
    
    Args:
        client: OpenAI client instance.
        messages: List of message objects.
        max_retries: Maximum number of retries.
        retry_delay: Delay between retries (in seconds).
        
    Returns:
        API response or None.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            return call_gpt(client, messages)
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"API call failed after {max_retries} attempts: {str(e)}")
                return None
            
            # Exponential backoff
            sleep_time = retry_delay * (2 ** (attempt - 1))
            logger.warning(f"API call error (attempt {attempt}/{max_retries}): {str(e)}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)