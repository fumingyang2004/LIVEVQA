import openai
import json
import base64 
import os
import time
import re
import sys
import argparse
from collections import defaultdict
import logging
import threading
import concurrent.futures
from tqdm import tqdm
from pathlib import Path

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Thread-local Storage and Lock ---
thread_local_storage = threading.local()
output_file_lock = threading.Lock()

# --- NEW PROMPT TEMPLATES ---
ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
Here is the question:
{question}
Please provide your best guess and a confidence score between
0% to 100% in the following JSON format:
{{
"answer": "Your answer here",
"confidence_score": number
}}
"""

GRADER_TEMPLATE ="""Your job is to look at a question, a gold target, and a predicted answer, and then assign a
grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to
double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha
Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born
on July 4, 1998, and Sasha was born on June 10, 2001.
'''
These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't
matter.
- Hedging and guessing are permissible, provided that the gold target is fully included
and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or
it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama
has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify
further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's
recommended to confirm the most accurate and updated information since this could change
over time. This model may not always reflect the most current information.
'''
These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target. Incorrect statements
that have some hedging (e.g., "it is possible that", "although i'm not sure, i think
") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
'''
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I
can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm
not sure about the other one.
'''
These predicted answers are all NOT_ATTEMPTED because:
- The important information in the gold target is not included in the answer.
- No statements in the answer contradict the gold target.

Also note the following things:
- For grading questions where the gold target is an number, the predicted answer needs to be
correct to the last significant figure in the gold answer. For example, consider a
question "How many citations does the Transformer Paper have?" with gold target "120k".
- Predicted answers "120k", "124k", and 115k" are all CORRECT.
- Predicted answers "100k" and "113k" are INCORRECT.
- Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED
because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the
predicted answer only needs to contain the information that is in the question.
- For example, consider the question "What episode did Derek and Meredith get legally
married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding".
Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer
.
- Do not punish predicted answers if they omit information that would be clearly inferred
from the question.
- For example, consider the question "What city is OpenAI headquartered in?" and the gold
target "San Francisco, California". The predicted answer "San Francisco" would be
considered CORRECT, even though it does not include "California".
- Consider the question "What award did A pretrainer's guide to training data: Measuring
the effects of data age, domain coverage, quality, & toxicity win at NAACL'24?", the
gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper"
would be considered CORRECT, because "award" is presumed in the question.
- For the question "What is the height of Jason Wei in meters?", the gold target is "1.73
m". The predicted answer "1.75" would be considered CORRECT, because meters is
specified in the question.
- For the question "What is the name of Barack Obama's wife?", the gold target is "
Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because
the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
- For example, if the gold target is "Hyung Won Chung", you can consider the following
predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won
Chung".

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT_ATTEMPTED. Don't
apologize or correct yourself if there was a mistake; we are just trying to grade the
answer.
'''
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
'''
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED
Just return the letters "A", "B", or "C", with no text around it.
"""

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description="Unified Benchmark Testing Tool - Supports News, Paper and Video benchmarks")
parser.add_argument("--benchmark", "-b", dest="benchmark_type", required=True, 
                  choices=["News", "Paper", "Video"],
                  help="Benchmark type to run (News, Paper, Video)")
parser.add_argument("--apikey", "-k", dest="api_key", 
                  help="Set API key (for either OpenAI or OpenRouter)")
parser.add_argument("--qamodel", "-q", dest="qa_model", 
                  help="Set QA model name")
parser.add_argument("--judgemodel", "-j", dest="judge_model", default="gpt-4.1-mini",
                  help="Set judge model name")
parser.add_argument("--workers", "-w", dest="max_workers", type=int, 
                  help="Maximum concurrent worker threads")
# New argument for API provider selection
parser.add_argument("--api-provider", "-p", dest="api_provider", choices=["openai", "openrouter"], default="openai",
                  help="API provider to use (openai or openrouter)")
# OpenRouter specific arguments
parser.add_argument("--http-referer", dest="http_referer", default="http://localhost/vqa-benchmark",
                  help="HTTP Referer header for OpenRouter (only used with --api-provider openrouter)")
parser.add_argument("--site-title", dest="site_title", default="VQA Benchmark Tool",
                  help="Site title header for OpenRouter (only used with --api-provider openrouter)")
args = parser.parse_args()

# Set API keys and models
API_KEY = args.api_key or os.environ.get("OPENAI_API_KEY", "YOUR API KEY")
QA_MODEL_NAME = args.qa_model or os.environ.get("QA_MODEL_NAME", "")
JUDGE_MODEL_NAME = args.judge_model or os.environ.get("JUDGE_MODEL_NAME", "gpt-4.1-mini")

# API Provider specific configuration
API_PROVIDER = args.api_provider.lower()
# OpenRouter configuration
HTTP_REFERER = args.http_referer
SITE_TITLE = args.site_title

# Configure based on benchmark type
BENCHMARK_TYPE = args.benchmark_type
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset-specific configuration
DATASET_CONFIG = {
    "News": {"img_dir": "News/News_imgs/", "max_workers": 5},
    "Paper": {"img_dir": "Paper/Paper_images/", "max_workers": 5},
    "Video": {"img_dir": "Video/Video_images/", "max_workers": 5}
}

# Use provided max_workers or default to dataset config
MAX_WORKERS = args.max_workers or DATASET_CONFIG[BENCHMARK_TYPE]["max_workers"]
IMAGE_BASE_DIR = os.path.join(BASE_DIR, DATASET_CONFIG[BENCHMARK_TYPE]["img_dir"])
BENCHMARK_FILE_PATH = os.path.join(BASE_DIR, BENCHMARK_TYPE, "benchmark.json")

# Setup output file paths
SAFE_QA_MODEL_NAME_PREFIX = QA_MODEL_NAME.replace('/', '_').replace(':', '_')
API_PROVIDER_SUFFIX = f"_{API_PROVIDER}" if API_PROVIDER != "openai" else ""
OUTPUT_JSON_FILENAME = os.path.join(BASE_DIR, BENCHMARK_TYPE, f"{SAFE_QA_MODEL_NAME_PREFIX}_details_{BENCHMARK_TYPE}{API_PROVIDER_SUFFIX}_grader_template.json")
OUTPUT_TXT_SUMMARY_FILENAME = os.path.join(BASE_DIR, BENCHMARK_TYPE, f"{SAFE_QA_MODEL_NAME_PREFIX}_summary_{BENCHMARK_TYPE}{API_PROVIDER_SUFFIX}_grader_template.txt")

# API call delay (seconds)
API_CALL_DELAY_SECONDS = 1

def get_openai_client():
    """Get thread-local OpenAI client"""
    if not hasattr(thread_local_storage, "openai_client"):
        if API_KEY == "YOUR API KEY":
            logger.warning("OpenAI API key not set or is using default value")
        
        try:
            thread_local_storage.openai_client = openai.OpenAI(api_key=API_KEY)
            logger.debug(f"OpenAI client initialized for thread {threading.current_thread().name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client for thread {threading.current_thread().name}: {e}")
            thread_local_storage.openai_client = None
    
    return thread_local_storage.openai_client

def get_openrouter_client():
    """Get thread-local OpenRouter client"""
    if not hasattr(thread_local_storage, "openrouter_client"):
        if API_KEY == "YOUR API KEY":
            logger.warning("OpenRouter API key not set or is using default value")
        
        try:
            thread_local_storage.openrouter_client = openai.OpenAI(
                api_key=API_KEY,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": HTTP_REFERER,
                    "X-Title": SITE_TITLE,
                }
            )
            logger.debug(f"OpenRouter client initialized for thread {threading.current_thread().name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client for thread {threading.current_thread().name}: {e}")
            thread_local_storage.openrouter_client = None
    
    return thread_local_storage.openrouter_client

def get_api_client():
    """Get appropriate API client based on provider setting"""
    if API_PROVIDER == "openrouter":
        return get_openrouter_client()
    else:
        return get_openai_client()

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file: 
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError: 
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e: 
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def get_image_mime_type(image_path):
    """Get MIME type based on image file extension"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]: return "image/jpeg"
    elif ext == ".png": return "image/png"
    elif ext == ".webp": return "image/webp"
    logger.warning(f"Unknown image extension {ext} for {image_path}, defaulting to image/jpeg")
    return "image/jpeg"

def get_open_ended_answer(full_image_path, question):
    """Get open-ended answer with confidence from vision model using the new prompt format"""
    client = get_api_client()
    default_response = {"answer": None, "confidence_score": 0, "error": "QA client/logic error"}
    if not client:
        return {**default_response, "error": f"{API_PROVIDER} client not available"}
    
    base64_image = encode_image_to_base64(full_image_path)
    if not base64_image:
        return {**default_response, "error": f"Failed to encode image: {full_image_path}"}
    
    mime_type = get_image_mime_type(full_image_path)
    qa_prompt_text_part = ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE.format(question=question)
    
    try:
        response = client.chat.completions.create(
            model=QA_MODEL_NAME,
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": qa_prompt_text_part},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        raw_response_content = response.choices[0].message.content.strip()
        
        # Enhanced JSON parsing logic
        try:
            # First try direct JSON parsing
            parsed_response = json.loads(raw_response_content)
            
            # Check if we have the expected structure
            if "answer" in parsed_response and "confidence_score" in parsed_response:
                # Try to convert confidence_score to float
                try:
                    parsed_response["confidence_score"] = float(parsed_response["confidence_score"])
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse confidence_score '{parsed_response.get('confidence_score')}' as float. Defaulting to 0.")
                    parsed_response["confidence_score"] = 0.0
                
                return parsed_response
            
            # If we're missing keys, check for alternative formats
            if "answer" not in parsed_response and "response" in parsed_response:
                logger.warning(f"Using 'response' key as 'answer' in model response")
                parsed_response["answer"] = parsed_response["response"]
            
            if "confidence_score" not in parsed_response:
                # Look for alternative confidence keys
                for alt_key in ["confidence", "score", "probability", "certainty"]:
                    if alt_key in parsed_response:
                        logger.warning(f"Using '{alt_key}' key as 'confidence_score' in model response")
                        parsed_response["confidence_score"] = parsed_response[alt_key]
                        break
            
            # If still missing keys after checking alternatives, use defaults
            if "answer" not in parsed_response:
                logger.warning("Missing 'answer' key in JSON response, using raw content as answer")
                parsed_response["answer"] = raw_response_content
            
            if "confidence_score" not in parsed_response:
                logger.warning("Missing 'confidence_score' key in JSON response, defaulting to 100")
                parsed_response["confidence_score"] = 100.0
            
            # Final conversion of confidence_score to float
            try:
                # Handle percentage strings like "90%" by removing non-numeric chars
                if isinstance(parsed_response["confidence_score"], str):
                    confidence_str = parsed_response["confidence_score"]
                    confidence_str = re.sub(r'[^0-9.]', '', confidence_str)
                    if confidence_str:
                        parsed_response["confidence_score"] = float(confidence_str)
                    else:
                        parsed_response["confidence_score"] = 0.0
                else:
                    parsed_response["confidence_score"] = float(parsed_response["confidence_score"])
                    
                # Ensure confidence is within 0-100 range
                if parsed_response["confidence_score"] > 1.0 and parsed_response["confidence_score"] <= 1.0:
                    parsed_response["confidence_score"] *= 100  # Convert 0-1 scale to 0-100
                    
            except (ValueError, TypeError):
                logger.warning("Failed to convert confidence_score to float, defaulting to 100")
                parsed_response["confidence_score"] = 100.0
                
            return parsed_response
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON substring
            logger.warning(f"Initial JSON parsing failed, attempting to extract JSON substring")
            json_pattern = r'(\{[\s\S]*\})'
            matches = re.search(json_pattern, raw_response_content)
            
            if matches:
                potential_json = matches.group(1)
                try:
                    extracted_json = json.loads(potential_json)
                    logger.info(f"Successfully extracted JSON substring from response")
                    
                    # Apply the same key checking logic as above
                    result = {"answer": None, "confidence_score": 0.0}
                    
                    if "answer" in extracted_json:
                        result["answer"] = extracted_json["answer"]
                    elif "response" in extracted_json:
                        result["answer"] = extracted_json["response"]
                    else:
                        result["answer"] = potential_json
                    
                    if "confidence_score" in extracted_json:
                        try:
                            result["confidence_score"] = float(extracted_json["confidence_score"])
                        except (ValueError, TypeError):
                            result["confidence_score"] = 100.0
                    elif "confidence" in extracted_json:
                        try:
                            result["confidence_score"] = float(extracted_json["confidence"])
                        except (ValueError, TypeError):
                            result["confidence_score"] = 100.0
                    
                    return result
                except json.JSONDecodeError:
                    pass
            
            # If all JSON parsing attempts fail, extract answer using regex patterns
            logger.warning(f"JSON extraction failed, attempting to infer answer using regex patterns")
            
            # Try to find answer in formats like "answer": "text" or "answer": text
            answer_pattern = r'"answer"\s*:\s*"([^"]+)"|"answer"\s*:\s*([^,"}\s]+)'
            confidence_pattern = r'"confidence_score"\s*:\s*(\d+\.?\d*)|"confidence_score"\s*:\s*"?(\d+\.?\d*%?)"?'
            
            answer_match = re.search(answer_pattern, raw_response_content)
            confidence_match = re.search(confidence_pattern, raw_response_content)
            
            answer = None
            if answer_match:
                answer = answer_match.group(1) or answer_match.group(2)
            
            confidence = 100.0
            if confidence_match:
                try:
                    confidence_str = confidence_match.group(1) or confidence_match.group(2)
                    confidence_str = re.sub(r'[^0-9.]', '', confidence_str)
                    confidence = float(confidence_str) if confidence_str else 100.0
                except (ValueError, TypeError):
                    confidence = 100.0
            
            if not answer:
                # If no structured answer found, use the full response as the answer
                answer = raw_response_content
                
            return {
                "answer": answer, 
                "confidence_score": confidence,
                "error": "Partial JSON extraction (fallback mode)"
            }
            
    except Exception as e:
        logger.error(f"OpenAI QA API call error (Model: {QA_MODEL_NAME}, Question: \"{question[:30]}...\", Image: {os.path.basename(full_image_path)}): {e}")
        return {**default_response, "error": f"QA API call failed: {str(e)}"}

def judge_answer_with_ai(question, model_answer_text, ground_truth_list):
    """Judge whether model's answer is correct using AI with GRADER_TEMPLATE"""
    client = get_api_client()
    default_error_response = {
        "grade": None,
        "reasoning": "Judge client/logic error.",
        "status_for_stats": "Error"
    }

    if not client:
        return {**default_error_response, "reasoning": f"{API_PROVIDER} client not available."}
    
    if model_answer_text is None or model_answer_text.strip() == "":
        logger.warning(f"QA model provided no answer text for question: {question}. Marking as NOT_ATTEMPTED.")
        return {
            "grade": "C",
            "reasoning": "Primary QA model did not provide an answer string.",
            "status_for_stats": "Not_Attempted"
        }

    if not ground_truth_list or not isinstance(ground_truth_list, list) or len(ground_truth_list) == 0:
        logger.error(f"Ground truth list is empty or invalid for question: {question}")
        return {**default_error_response, "reasoning": "Invalid or empty ground truth list provided."}
    
    # 修改: 不再仅使用第一个答案，而是格式化整个ground_truth_list
    # 将所有可能的正确答案用 "OR" 连接，这样评分模板可以考虑所有可能的答案
    formatted_target = " OR ".join([f'"{gt}"' for gt in ground_truth_list])
    
    # 不再显示警告，因为我们现在使用所有ground truth
    if len(ground_truth_list) > 1:
        logger.info(f"Multiple ground truths provided for question '{question}'. Using all {len(ground_truth_list)} options as valid answers.")

    final_judge_prompt = GRADER_TEMPLATE.format(
        question=question,
        target=formatted_target,
        predicted_answer=model_answer_text
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[{"role": "user", "content": final_judge_prompt}],
            temperature=0.0  # For deterministic grading
        )
        judge_raw_output = response.choices[0].message.content.strip().upper()  # Ensure uppercase

        status_for_stats = "Error"  # Default
        if judge_raw_output == "A":
            status_for_stats = "CORRECT"
        elif judge_raw_output == "B":
            status_for_stats = "INCORRECT"
        elif judge_raw_output == "C":
            status_for_stats = "NOT_ATTEMPTED"
        else:
            logger.warning(f"Judge output was not A, B, or C: '{judge_raw_output}'. Question: '{question[:30]}...'. Treating as Error.")
            # Keep status_for_stats as "Error"

        return {
            "grade": judge_raw_output if judge_raw_output in ["A", "B", "C"] else None,
            "reasoning": f"AI Grader result based on GRADER_TEMPLATE. Raw output: {judge_raw_output}",
            "status_for_stats": status_for_stats
        }

    except Exception as e: 
        logger.error(f"OpenAI Judge API call error (Model: {JUDGE_MODEL_NAME}, Question: \"{question[:30]}...\"): {e}")
        return {
            **default_error_response,
            "reasoning": f"Judge API call failed: {str(e)}",
        }

def process_single_qa_level(item_id, qa_level_key, qa_data, base_img_filename_from_item):
    """Process single QA level question and answer using the new GRADER_TEMPLATE approach"""
    question = qa_data.get("question")
    ground_truth_list = qa_data.get("Ground_Truth_List")
    question_type = qa_data.get("question_type", "unknown_type")
    
    # Get image path
    image_filename = qa_data.get("img_path", base_img_filename_from_item)
    
    full_image_path = None
    if image_filename:
        full_image_path = os.path.join(IMAGE_BASE_DIR, image_filename)
    else:
        logger.warning(f"   {qa_level_key} (ID: {item_id}) missing image filename.")

    # Prepare result data structure
    level_result = {
        "question": question,
        "question_type": question_type,
        "img_filename_from_json": image_filename, 
        "img_path_used": full_image_path,        
        "ground_truth_list": ground_truth_list,
        "model_qa_output": None,
        "judge_evaluation": None,
    }

    # Validate required data exists
    if not all([question, ground_truth_list, full_image_path]):
        logger.warning(f"   {qa_level_key} (ID: {item_id}) incomplete data (Question/GT/Image Path), skipping. Details: Q: {bool(question)}, GT: {bool(ground_truth_list)}, Img: {full_image_path}")
        level_result["model_qa_output"] = {"answer": None, "confidence_score": 0, "error": "Incomplete input data (missing Q, GT, or Img path)."}
        level_result["judge_evaluation"] = {
            "grade": None,
            "reasoning": "Incomplete input data for QA.",
            "status_for_stats": "Error"
        }
        return level_result

    # Process QA
    logger.info(f"   Processing {qa_level_key} (ID: {item_id}, Type: {question_type})")
    logger.info(f"     Question: {question}")
    logger.info(f"     Ground Truths: {ground_truth_list}")
    logger.info(f"     Using image: {full_image_path}")

    # Get model answer with confidence
    model_qa_response_json = get_open_ended_answer(full_image_path, question)
    time.sleep(API_CALL_DELAY_SECONDS)
    level_result["model_qa_output"] = model_qa_response_json
    
    model_answer_text = model_qa_response_json.get("answer")
    confidence_score = model_qa_response_json.get("confidence_score", 0)
    qa_error = model_qa_response_json.get("error")

    if qa_error:
        logger.info(f"     Model QA output (ID: {item_id}, {qa_level_key}): Error - {qa_error}. Answer text (if any): {model_answer_text if model_answer_text else 'N/A'}")
    else:
        logger.info(f"     Model QA output (ID: {item_id}, {qa_level_key}): \"{model_answer_text}\" (Confidence: {confidence_score}%)")

    # Judge model answer using GRADER_TEMPLATE
    judge_evaluation = judge_answer_with_ai(question, model_answer_text, ground_truth_list)
    time.sleep(API_CALL_DELAY_SECONDS)
    level_result["judge_evaluation"] = judge_evaluation
    
    logger.info(f"     Judge evaluation (ID: {item_id}, {qa_level_key}) - Grade: {judge_evaluation.get('grade', 'N/A')}, Status: {judge_evaluation.get('status_for_stats', 'Error')}")
    logger.info(f"     ------------------------------------")
    
    return level_result

def process_item_task(item):
    """Process a single benchmark item task, including level1 and level2 questions"""
    item_id = item.get("id", f"unknown_id_{hash(json.dumps(item))}")
    base_img_filename = item.get("img_path")
    
    # Prepare output result structure
    item_output = {
        "id": item_id,
        "original_img_filename": base_img_filename, 
        "source": item.get("source"),
        "time": item.get("time"),
        "level1": None,
        "level2": None
    }
    
    # Check image path
    if not base_img_filename and not (("level1_qa" in item and item["level1_qa"].get("img_path")) or \
                                      ("level2_qa" in item and item["level2_qa"].get("img_path"))):
        logger.warning(f"Item {item_id} missing top-level 'img_path' and no 'img_path' in individual QA levels.")
    
    logger.debug(f"Starting processing for item {item_id}")
    
    # Process level1 and level2 questions
    if "level1_qa" in item and item["level1_qa"]: 
        item_output["level1"] = process_single_qa_level(item_id, "level1_qa", item["level1_qa"], base_img_filename)
        
    if "level2_qa" in item and item["level2_qa"]: 
        item_output["level2"] = process_single_qa_level(item_id, "level2_qa", item["level2_qa"], base_img_filename)
        
    logger.debug(f"Completed processing for item {item_id}")
    return item_output

def save_item_result_incrementally(item_result_data, filepath):
    """Incrementally save item result to file"""
    with output_file_lock:
        current_results_list = []
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                with open(filepath, 'r', encoding='utf-8') as f_in:
                    content = f_in.read()
                    if content.strip(): 
                        current_results_list = json.loads(content)
                        if not isinstance(current_results_list, list):
                            logger.warning(f"Output file {filepath} content is not a JSON list. Will start anew.")
                            current_results_list = []
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {filepath}. File might be corrupted. Will start list anew.")
            current_results_list = []
        except Exception as e:
            logger.error(f"Error reading or parsing existing results file {filepath}: {e}. Will start list anew.")
            current_results_list = []

        # Add new result
        current_results_list.append(item_result_data)

        # Save results
        try:
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, 'w', encoding='utf-8') as f_out:
                json.dump(current_results_list, f_out, indent=2, ensure_ascii=False)
            os.replace(temp_filepath, filepath) 
            logger.debug(f"Result saved incrementally to {filepath} (New item ID: {item_result_data.get('id')})")
        except Exception as e:
            logger.error(f"Error writing result to {filepath}: {e}")

def load_existing_results(filepath):
    """Load existing results and extract processed item IDs"""
    processed_ids = set()
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
                if content.strip():
                    existing_results = json.loads(content)
                    if isinstance(existing_results, list):
                        for item in existing_results:
                            if "id" in item:
                                processed_ids.add(item["id"])
                        logger.info(f"Found {len(processed_ids)} previously processed items in existing results file.")
                    else:
                        logger.warning(f"Output file {filepath} content is not a JSON list.")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error reading existing results from {filepath}: {e}")
    
    return processed_ids

def main():
    """Main function: Run unified evaluation process"""
    # Validate API key and model configuration
    if API_KEY == "YOUR API KEY" or not API_KEY:
        logger.error(f"Error: API key for {API_PROVIDER} not properly configured. Exiting.")
        return
        
    # Validate QA model
    if not QA_MODEL_NAME:
        logger.error("Error: QA model name not specified. Exiting.")
        return

    logger.info(f"Using API provider: {API_PROVIDER.upper()}")
    logger.info(f"JSON detailed results will be saved to: {OUTPUT_JSON_FILENAME}")
    logger.info(f"TXT summary will be appended to: {OUTPUT_TXT_SUMMARY_FILENAME}")

    # Load benchmark data
    try:
        with open(BENCHMARK_FILE_PATH, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except FileNotFoundError: 
        logger.error(f"Benchmark file not found: {BENCHMARK_FILE_PATH}"); 
        return
    except json.JSONDecodeError: 
        logger.error(f"Could not decode benchmark JSON: {BENCHMARK_FILE_PATH}"); 
        return

    # Check for existing results and get processed IDs
    processed_ids = load_existing_results(OUTPUT_JSON_FILENAME)
    
    # Filter benchmark_data to only include unprocessed items
    unprocessed_items = []
    for item in benchmark_data:
        item_id = item.get("id", f"unknown_id_{hash(json.dumps(item))}")
        if item_id not in processed_ids:
            unprocessed_items.append(item)
    
    if processed_ids:
        logger.info(f"Resuming from checkpoint: {len(processed_ids)} items already processed, {len(unprocessed_items)} items remaining.")
    
    if len(unprocessed_items) == 0:
        logger.info("All items have been processed. No new items to evaluate.")
        # Load existing results for statistics
        all_item_results_for_stats = []
        try:
            if os.path.exists(OUTPUT_JSON_FILENAME):
                with open(OUTPUT_JSON_FILENAME, 'r', encoding='utf-8') as f_in:
                    all_item_results_for_stats = json.load(f_in)
                    logger.info(f"Loaded {len(all_item_results_for_stats)} completed results for statistics.")
        except Exception as e:
            logger.error(f"Error loading existing results for statistics: {e}")
            return
    else:
        # Process the unprocessed items
        all_item_results_for_stats = [] 
        
        logger.info(f"Starting {BENCHMARK_TYPE} benchmark evaluation (using {API_PROVIDER.upper()}, with GRADER_TEMPLATE). QA Model: {QA_MODEL_NAME}, Judge Model: {JUDGE_MODEL_NAME}")
        logger.info(f"Benchmark: {BENCHMARK_FILE_PATH}, {len(unprocessed_items)} items remaining. Concurrency: {MAX_WORKERS}.")
        logger.info(f"Images will be loaded from: {IMAGE_BASE_DIR} (Absolute path: {os.path.abspath(IMAGE_BASE_DIR)})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="VQAWorker") as executor:
            futures = [executor.submit(process_item_task, item) for item in unprocessed_items]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(unprocessed_items), desc="Processing benchmark items"):
                try:
                    item_result = future.result()
                    if item_result:
                        save_item_result_incrementally(item_result, OUTPUT_JSON_FILENAME)
                        all_item_results_for_stats.append(item_result)
                except Exception as exc:
                    logger.error(f"Exception occurred while processing item: {exc}", exc_info=True)
        
        logger.info(f"All remaining items processed. Total newly processed results: {len(all_item_results_for_stats)}.")
        
        # Load all results for statistics (including both previously processed and new ones)
        try:
            with open(OUTPUT_JSON_FILENAME, 'r', encoding='utf-8') as f_in:
                all_item_results_for_stats = json.load(f_in)
                logger.info(f"Loaded {len(all_item_results_for_stats)} total results for statistics.")
        except Exception as e:
            logger.error(f"Error loading complete results for statistics: {e}")
            # Continue with partial results if we have any
            if not all_item_results_for_stats:
                logger.error("No results available for statistics. Exiting.")
                return

    # Generate statistics based on A/B/C grading scheme
    stats = {'level1': defaultdict(int), 'level2': defaultdict(int), 'by_type': defaultdict(lambda: defaultdict(int))}
    
    for res_item in all_item_results_for_stats:
        for level_key_short, level_data_key in [('level1', 'level1'), ('level2', 'level2')]:
            level_res = res_item.get(level_data_key)
            if level_res and level_res.get("judge_evaluation"): 
                question_type = level_res.get("question_type", "unknown_type")
                judge_eval = level_res.get("judge_evaluation")
                evaluation_status = judge_eval.get("status_for_stats")

                stats[level_key_short]['total_questions'] += 1
                stats['by_type'][question_type]['total_questions'] += 1
                
                if evaluation_status == "Correct":  # Graded A
                    stats[level_key_short]['graded_A_correct'] += 1
                    stats['by_type'][question_type]['graded_A_correct'] += 1
                elif evaluation_status == "Incorrect":  # Graded B
                    stats[level_key_short]['graded_B_incorrect'] += 1
                    stats['by_type'][question_type]['graded_B_incorrect'] += 1
                elif evaluation_status == "Not_Attempted":  # Graded C
                    stats[level_key_short]['graded_C_not_attempted'] += 1
                    stats['by_type'][question_type]['graded_C_not_attempted'] += 1
                elif evaluation_status == "Error":  # API or Judge format error
                    stats[level_key_short]['errors_judge_processing'] += 1
                    stats['by_type'][question_type]['errors_judge_processing'] += 1
                else:  # Should not happen if status_for_stats is always one of the above
                    stats[level_key_short]['unknown_status'] += 1
                    stats['by_type'][question_type]['unknown_status'] += 1
    
    # Write results to summary file
    summary_file_handler = None
    try:
        summary_file_handler = logging.FileHandler(OUTPUT_TXT_SUMMARY_FILENAME, mode='a', encoding='utf-8')
        summary_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(summary_file_handler)

        logger.info(f"\n\n--- {QA_MODEL_NAME} @ {BENCHMARK_TYPE} Benchmark (OpenAI API, GRADER_TEMPLATE) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")

        logger.info("\n--- Evaluation Summary (Based on GRADER_TEMPLATE) ---")
        for level_key_stat in ['level1', 'level2']:
            s = stats[level_key_stat]
            total = s['total_questions']
            correct_A = s['graded_A_correct']
            incorrect_B = s['graded_B_incorrect']
            not_attempted_C = s['graded_C_not_attempted']
            errors = s['errors_judge_processing']
            
            logger.info(f"\n{level_key_stat.capitalize()}:")
            if total > 0:
                logger.info(f"  Total questions: {total}")
                logger.info(f"  A (CORRECT): {correct_A}")
                logger.info(f"  B (INCORRECT): {incorrect_B}")
                logger.info(f"  C (NOT_ATTEMPTED): {not_attempted_C}")
                logger.info(f"  Evaluation processing errors: {errors}")
                
                # Accuracy calculation (A / (A+B)) - excluding C and errors from denominator
                valid_AB_attempts = correct_A + incorrect_B
                accuracy_AB = (correct_A / valid_AB_attempts) * 100 if valid_AB_attempts > 0 else 0
                logger.info(f"  Accuracy (A / (A+B)): {accuracy_AB:.2f}% ({correct_A}/{valid_AB_attempts})")

                # Accuracy calculation (A / (A+B+C)) - excluding only errors
                valid_ABC_attempts = correct_A + incorrect_B + not_attempted_C
                accuracy_ABC = (correct_A / valid_ABC_attempts) * 100 if valid_ABC_attempts > 0 else 0
                logger.info(f"  Accuracy (A / (A+B+C)): {accuracy_ABC:.2f}% ({correct_A}/{valid_ABC_attempts})")
            else:
                logger.info(f"  {level_key_stat.capitalize()}: No evaluation data.")

        logger.info("\n--- Accuracy by Question Type (Based on GRADER_TEMPLATE) ---")
        if not stats['by_type']:
            logger.info("No type-categorized data available.")
        else:
            for q_type, type_s in sorted(stats['by_type'].items()):
                total = type_s['total_questions']
                correct_A = type_s['graded_A_correct']
                incorrect_B = type_s['graded_B_incorrect']
                not_attempted_C = type_s['graded_C_not_attempted']
                errors = type_s['errors_judge_processing']
                
                logger.info(f"\n  Type '{q_type}':")
                if total > 0:
                    logger.info(f"    Total questions: {total}")
                    logger.info(f"    A (CORRECT): {correct_A}")
                    logger.info(f"    B (INCORRECT): {incorrect_B}")
                    logger.info(f"    C (NOT_ATTEMPTED): {not_attempted_C}")
                    logger.info(f"    Evaluation processing errors: {errors}")
                    
                    valid_AB_attempts = correct_A + incorrect_B
                    accuracy_AB = (correct_A / valid_AB_attempts) * 100 if valid_AB_attempts > 0 else 0
                    logger.info(f"    Accuracy (A / (A+B)): {accuracy_AB:.2f}% ({correct_A}/{valid_AB_attempts})")
                    
                    valid_ABC_attempts = correct_A + incorrect_B + not_attempted_C
                    accuracy_ABC = (correct_A / valid_ABC_attempts) * 100 if valid_ABC_attempts > 0 else 0
                    logger.info(f"    Accuracy (A / (A+B+C)): {accuracy_ABC:.2f}% ({correct_A}/{valid_ABC_attempts})")
                else:
                    logger.info(f"    Type '{q_type}': N/A (0 questions)")
        
        # Calculate overall accuracy
        overall_total_q = stats['level1']['total_questions'] + stats['level2']['total_questions']
        overall_A = stats['level1']['graded_A_correct'] + stats['level2']['graded_A_correct']
        overall_B = stats['level1']['graded_B_incorrect'] + stats['level2']['graded_B_incorrect']
        overall_C = stats['level1']['graded_C_not_attempted'] + stats['level2']['graded_C_not_attempted']
        overall_errors = stats['level1']['errors_judge_processing'] + stats['level2']['errors_judge_processing']

        logger.info("\n--- Overall Accuracy (Based on GRADER_TEMPLATE) ---")
        if overall_total_q > 0:
            logger.info(f"  Total questions: {overall_total_q}")
            logger.info(f"  A (CORRECT): {overall_A}")
            logger.info(f"  B (INCORRECT): {overall_B}")
            logger.info(f"  C (NOT_ATTEMPTED): {overall_C}")
            logger.info(f"  Evaluation processing errors: {overall_errors}")
            
            overall_valid_AB = overall_A + overall_B
            overall_accuracy_AB = (overall_A / overall_valid_AB) * 100 if overall_valid_AB > 0 else 0
            logger.info(f"  Overall accuracy (A / (A+B)): {overall_accuracy_AB:.2f}% ({overall_A}/{overall_valid_AB})")

            overall_valid_ABC = overall_A + overall_B + overall_C
            overall_accuracy_ABC = (overall_A / overall_valid_ABC) * 100 if overall_valid_ABC > 0 else 0
            logger.info(f"  Overall accuracy (A / (A+B+C)): {overall_accuracy_ABC:.2f}% ({overall_A}/{overall_valid_ABC})")
        else:
            logger.info("  No questions evaluated.")
        
        logger.info("\n" + "="*50 + "\n")

    except Exception as e:
        logger.error(f"Error writing TXT summary: {e}")
    finally:
        if summary_file_handler:
            logger.removeHandler(summary_file_handler)
            summary_file_handler.close()
            
    logger.info("Evaluation completed.")

if __name__ == "__main__":
    main()
