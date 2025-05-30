import openai
import logging
import threading
from typing import Optional

from .local_config import get_local_config

logger = logging.getLogger(__name__)
local_thread_storage = threading.local()

def get_local_client(model_name: str, base_url: Optional[str] = None):
    """Get thread-local local model client"""
    if not hasattr(local_thread_storage, "local_clients"):
        local_thread_storage.local_clients = {}
    
    client_key = f"{model_name}_{base_url or 'default'}"
    
    if client_key not in local_thread_storage.local_clients:
        config = get_local_config(model_name, base_url)
        
        try:
            client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
            local_thread_storage.local_clients[client_key] = client
            logger.debug(f"Local client initialized for {model_name} at {config.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize local client for {model_name}: {e}")
            local_thread_storage.local_clients[client_key] = None
    
    return local_thread_storage.local_clients[client_key]

def test_local_model_connection(model_name: str, base_url: Optional[str] = None) -> bool:
    """Test if local model is accessible"""
    try:
        client = get_local_client(model_name, base_url)
        if not client:
            return False
        
        # Simple test call
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            temperature=0
        )
        return True
    except Exception as e:
        logger.warning(f"Local model {model_name} connection test failed: {e}")
        return False
