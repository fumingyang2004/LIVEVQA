import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_benchmark import get_api_client
from .local_client import get_local_client, test_local_model_connection

def get_unified_client(model_name: str, provider: str = "openai", local_base_url: str = None):
    """Get client based on provider type"""
    if provider == "local":
        return get_local_client(model_name, local_base_url)
    else:
        return get_api_client()

def is_model_available(model_name: str, provider: str = "openai", local_base_url: str = None) -> bool:
    """Check if model is available"""
    if provider == "local":
        return test_local_model_connection(model_name, local_base_url)
    else:
        try:
            client = get_api_client()
            return client is not None
        except:
            return False
