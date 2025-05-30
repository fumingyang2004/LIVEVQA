import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class LocalModelConfig:
    name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "sk-no-key-required"
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout: int = 120

LOCAL_MODELS = {
    "llava-v1.6-34b": LocalModelConfig(
        name="llava-v1.6-34b",
        base_url="http://localhost:8000/v1"
    ),
    "qwen2-vl-72b": LocalModelConfig(
        name="qwen2-vl-72b", 
        base_url="http://localhost:8001/v1"
    ),
    "qwen2-7b-instruct": LocalModelConfig(
        name="qwen2-7b-instruct",
        base_url="http://localhost:8002/v1"
    )
}

def get_local_config(model_name: str, base_url: Optional[str] = None) -> LocalModelConfig:
    if model_name in LOCAL_MODELS:
        config = LOCAL_MODELS[model_name]
        if base_url:
            config.base_url = base_url
        return config
    
    return LocalModelConfig(
        name=model_name,
        base_url=base_url or "http://localhost:8000/v1"
    )

def add_local_model(model_name: str, base_url: str, **kwargs) -> LocalModelConfig:
    config = LocalModelConfig(name=model_name, base_url=base_url, **kwargs)
    LOCAL_MODELS[model_name] = config
    return config
