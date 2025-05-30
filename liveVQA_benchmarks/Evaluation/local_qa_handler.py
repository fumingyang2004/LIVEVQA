import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path (parent directory of Evaluation)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unified_benchmark import (
    ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE,
    GRADER_TEMPLATE,
    encode_image_to_base64,
    get_image_mime_type
)
from .unified_client import get_unified_client
from .local_config import get_local_config

logger = logging.getLogger(__name__)

def get_qa_answer_local(image_path: str, question: str, model_name: str, 
                       provider: str = "openai", local_base_url: str = None,
                       api_delay: float = 1.0):
    """Get QA answer using local or API models"""
    default_response = {"answer": None, "confidence_score": 0, "error": "QA client error"}
    
    try:
        client = get_unified_client(model_name, provider, local_base_url)
        if not client:
            return {**default_response, "error": f"Client for {model_name} ({provider}) not available"}
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return {**default_response, "error": f"Failed to encode image: {image_path}"}
        
        mime_type = get_image_mime_type(image_path)
        qa_prompt = ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE.format(question=question)
        
        # Build message
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": qa_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        }]
        
        # Get model config for parameters
        if provider == "local":
            config = get_local_config(model_name, local_base_url)
            temperature = config.temperature
            max_tokens = config.max_tokens
        else:
            temperature = 0.2
            max_tokens = None
        
        # API call
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        response = client.chat.completions.create(**params)
        raw_content = response.choices[0].message.content.strip()
        
        # Parse response (reuse existing logic from unified_benchmark.py)
        import json
        import re
        
        try:
            parsed_response = json.loads(raw_content)
            
            if "answer" in parsed_response and "confidence_score" in parsed_response:
                try:
                    parsed_response["confidence_score"] = float(parsed_response["confidence_score"])
                except (ValueError, TypeError):
                    parsed_response["confidence_score"] = 0.0
                
                if api_delay > 0:
                    time.sleep(api_delay)
                return parsed_response
            
            # Handle alternative formats
            if "answer" not in parsed_response and "response" in parsed_response:
                parsed_response["answer"] = parsed_response["response"]
            
            if "confidence_score" not in parsed_response:
                for alt_key in ["confidence", "score", "probability"]:
                    if alt_key in parsed_response:
                        parsed_response["confidence_score"] = parsed_response[alt_key]
                        break
                else:
                    parsed_response["confidence_score"] = 100.0
            
            try:
                parsed_response["confidence_score"] = float(parsed_response["confidence_score"])
            except:
                parsed_response["confidence_score"] = 100.0
                
            if api_delay > 0:
                time.sleep(api_delay)
            return parsed_response
            
        except json.JSONDecodeError:
            # Fallback parsing
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', raw_content)
            confidence_match = re.search(r'"confidence_score"\s*:\s*(\d+\.?\d*)', raw_content)
            
            result = {
                "answer": answer_match.group(1) if answer_match else raw_content,
                "confidence_score": float(confidence_match.group(1)) if confidence_match else 100.0,
                "error": "Fallback JSON parsing"
            }
            
            if api_delay > 0:
                time.sleep(api_delay)
            return result
            
    except Exception as e:
        logger.error(f"QA API call error (Model: {model_name}, Provider: {provider}): {e}")
        return {**default_response, "error": f"QA API call failed: {str(e)}"}

def judge_answer_local(question: str, model_answer: str, ground_truth_list: list,
                      judge_model: str, provider: str = "openai", 
                      local_base_url: str = None, api_delay: float = 1.0):
    """Judge answer using local or API models"""
    default_error = {
        "grade": None,
        "reasoning": "Judge client error",
        "status_for_stats": "Error"
    }
    
    try:
        client = get_unified_client(judge_model, provider, local_base_url)
        if not client:
            return {**default_error, "reasoning": f"Judge client for {judge_model} ({provider}) not available"}
        
        if not model_answer or model_answer.strip() == "":
            return {
                "grade": "C",
                "reasoning": "No answer provided by QA model",
                "status_for_stats": "NOT_ATTEMPTED"
            }
        
        if not ground_truth_list:
            return {**default_error, "reasoning": "Invalid ground truth list"}
        
        # Format target answers
        formatted_target = " OR ".join([f'"{gt}"' for gt in ground_truth_list])
        
        judge_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=formatted_target,
            predicted_answer=model_answer
        )
        
        messages = [{"role": "user", "content": judge_prompt}]
        
        response = client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=0.0
        )
        
        judge_output = response.choices[0].message.content.strip().upper()
        
        status_map = {
            "A": "CORRECT",
            "B": "INCORRECT", 
            "C": "NOT_ATTEMPTED"
        }
        
        status = status_map.get(judge_output, "Error")
        
        if api_delay > 0:
            time.sleep(api_delay)
            
        return {
            "grade": judge_output if judge_output in ["A", "B", "C"] else None,
            "reasoning": f"AI Grader result: {judge_output}",
            "status_for_stats": status
        }
        
    except Exception as e:
        logger.error(f"Judge API call error (Model: {judge_model}, Provider: {provider}): {e}")
        return {**default_error, "reasoning": f"Judge API call failed: {str(e)}"}
