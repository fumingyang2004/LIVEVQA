import os
import glob
import shutil
import base64
import requests
import json
import argparse
import time
import re
import logging
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# Add multiprocessing support
from multiprocessing import Pool, cpu_count

# --- Configuration ---
# Replace with your OpenAI API key
API_KEY = os.environ.get("OPENAI_API_KEY") # Make sure to set your valid API Key
# GPT-4 Vision model name (use a vision-capable model, e.g., gpt-4o)
GPT_MODEL = "gpt-4.1" # Update to the recommended model, e.g., gpt-4o
# OpenAI API endpoint
API_URL = "https://api.openai.com/v1/chat/completions"

# Input directory
DEFAULT_ROOT_DIR = ""
# Image file extensions
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

# --- Logging Setup ---
def setup_logger():
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create a timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"image_selection_{timestamp}.log")

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging setup, log file: {log_file}")
    return log_file

# --- Helper Functions ---

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None

def call_gpt4_vision(topic, content, image_paths, max_images_per_request=5):
    """
    Calls the GPT-4 Vision API to select images.
    Uses a two-stage selection process:
    1. First, rates images in batches.
    2. From the top-rated images, selects the best 5 diverse images.
    Returns a list of full paths to the selected images.
    """
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        logging.error("ERROR: Please set your OpenAI API key in the script or as an environment variable.")
        return []
    if not image_paths:
        logging.warning("WARNING: No images provided for processing.")
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # --- Stage 1: Rate images in batches ---
    logging.info("Stage 1: Evaluating image quality and relevance for each batch and scoring.")

    image_scores = []

    # --- English Prompt for Batch Scoring (使用 corrected_prompt_v2 的内容) ---
    batch_prompt_text = """
**Prompt Objective:**
You are an expert image analyst tasked with selecting images for a Question-Answering (QA) generation system. Your selections will be used to test a Large Language Model's (LLM) visual understanding, so images with minimal textual clues are paramount.

**Core Task:**
Evaluate EACH image provided in the current batch based on the Topic and Content Description below. Assign a score from 1 to 10 (10 is best) and provide a concise justification, focusing on its suitability for QA generation and the level of textual interference.

**IMPORTANT SCORING GUIDANCE:**
- Assign 8–10 to images that perform strongly on most criteria and do not have major flaws.
  Minor imperfections (e.g., small background text, mild quality issues, or faint watermarks/media logos) can still receive scores in the 7–9 range if overall relevance and informativeness are high.
- Images with some visual or contextual issues may still score 6–7 if they are otherwise useful for question generation.
- Only assign very low scores (1–3) to images that are blurry, of extremely poor quality, or have large overlaid text that clearly reveals answers or dominates the content.
- **News-style captions, watermarks, or channel graphics** are acceptable as long as they do not contain direct answers or overwhelm the main visual content.

General Advice: When in doubt, favor moderate to high scores for images that are clearly useful for QA purposes. Extreme scores (1 or 10) should be reserved for clearly unusable or exceptional cases.

**Topic:**
"{topic}"

**Content Description:**
"{content}"

**Evaluation Criteria (Score each image from 1-10):**

1.  **High Content Relevance (Weight: High):**
    * MUST be strongly related to the Topic and Content Description.
    * Focus: Does the image offer rich visual context for generating insightful questions about the topic?

2.  **Visual Clarity & Quality (Weight: High):**
    * MUST be clear, well-focused, and well-composed. Reject blurry or very low-quality images (assign score 1-2).
    * Focus: Are visual details easily discernible for LLM interpretation?

3.  **Information Richness & Element Diversity (Weight: Medium-High):**
    * Prioritize images showing varied scenes, multiple relevant objects, interactions, or activities. Avoid overly simplistic or empty images.
    * Focus: Does the image provide multiple distinct visual elements or sub-topics for questioning?

4.  **Minimal Textual Interference (Weight: CRITICAL - Low score for significant text):**
    * CRITICAL: Images with significant overlay text (captions, large logos, direct answers) that could "give away" information to the LLM should be scored very low (e.g., 1-3). The goal is to test visual understanding, not text reading.
    * Acceptable: Incidental background text (e.g., a distant street sign) is usually fine if not prominent or central to understanding the core content.
    * Focus: Does the image primarily convey information visually, or does text play a major role that would simplify QA for an LLM? Less text is better.

5.  **No Personal/Sensitive Identifiers (Weight: High - Reject if present):**
    * MUST NOT contain visible PII (names, faces of non-public figures unless anonymized/consented), or private organizational details. Score 1 if present.
    * Focus: Is the image safe and appropriate for general use?

6.  **Context over Sole Presenter (Weight: Medium):**
    * Avoid images SOLELY of a speaker/presenter unless their specific action/expression is key and described in the content. Prefer images with more contextual elements.
    * Focus: Does the image offer more than just a portrait?

**Output Format (STRICTLY FOLLOW - Your entire response MUST be a single, valid JSON object as described below):**

Your response must be a single JSON object. This object must contain one top-level key: "image_evaluations".
The value of "image_evaluations" must be a JSON array.
Each element in this array must be a JSON object representing one image, with the following fields:
- "image_number": (Integer) The 1-based index of the image as it was presented in the batch.
- "score": (Float or Integer) The score assigned, from 1 to 10.
- "justification": (String) A concise justification for the score, specifically mentioning relevance, visual quality, and especially the level/impact of any text.
- "contains_problematic_text": (Boolean) true if the image contains significant overlay text, captions, or labels that could directly provide answers or make QA too easy; false otherwise.

**Example of the EXACT JSON output format (for a batch of 2 images):**
```json
{{
    "image_number": 1,
    "score": 8.5,
    "justification": "High relevance, excellent clarity. Minimal non-distracting background text.",
    "contains_problematic_text": false
}}
"""

    # Process images in batches
    for i in range(0, len(image_paths), max_images_per_request):
        batch_image_paths = image_paths[i:i + max_images_per_request]
        logging.info(f"  Processing image batch {i // max_images_per_request + 1} ({len(batch_image_paths)} images)...")

        # Ensure topic and content are strings, in case they are None
        current_topic = topic if topic is not None else "N/A"
        current_content = content if content is not None else "N/A"

        # The batch_prompt_text is defined as an f-string, so {topic} and {content} are interpolated.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": batch_prompt_text},
                ]
            }
        ]

        valid_batch_indices = [] # Track indices of images actually added to the request in the original batch_image_paths

        for idx, img_path in enumerate(batch_image_paths):
            base64_image = encode_image_to_base64(img_path)
            if base64_image:
                mime_type = "image/jpeg"
                if img_path.lower().endswith(".png"):
                    mime_type = "image/png"
                elif img_path.lower().endswith(".webp"):
                    mime_type = "image/webp"
                image_url = f"data:{mime_type};base64,{base64_image}"
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
                valid_batch_indices.append(idx) # Track this image as valid, and save its original index in batch_image_paths
            else:
                logging.warning(f"  Could not encode image: {os.path.basename(img_path)}")

        if len(messages[0]["content"]) == 1: # Only text prompt, no valid images
            logging.warning("  WARNING: No valid images in the current batch to send.")
            continue

        payload = {
            "model": GPT_MODEL,
            "messages": messages,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }

        try:
            session = requests.Session()
            retries = Retry(
                total=5,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(API_URL, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json() # API response should be JSON

            # With "response_format": {"type": "json_object"},
            # result['choices'][0]['message']['content'] should be a string suitable for json.loads().
            if 'choices' in result and len(result['choices']) > 0:
                content_response_str = result['choices'][0]['message']['content']
                logging.debug(f"  GPT Raw JSON Response for batch scoring: {content_response_str}")
                # content_response_str is already a JSON string and can be passed to parse_image_ratings
                image_ratings = parse_image_ratings(content_response_str, batch_image_paths, valid_batch_indices)
                image_scores.extend(image_ratings)
                logging.info(f"  Batch {i // max_images_per_request + 1} scoring complete, got {len(image_ratings)} ratings.")
            else:
                logging.error(f"API response format error or no choices: {result}")

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"  Response content: {e.response.text}")
        except json.JSONDecodeError as e: # Add this exception handling for invalid JSON from API
            logging.error(f"Failed to decode JSON from GPT's content response: {e}")
            logging.error(f"  GPT's problematic content string: {content_response_str}")
        except Exception as e:
            logging.error(f"Error processing API response: {e}")

        time.sleep(2)

    if not image_scores:
        logging.warning("No image ratings obtained, returning empty list.")
        return []

    image_scores.sort(key=lambda x: x['score'], reverse=True)
    top_10_images = [img for img in image_scores if img['score'] >= 6][:10]

    logging.info(f"Scoring complete. Top {len(top_10_images)} images (up to 10):")
    for i_img, img in enumerate(top_10_images):
        logging.info(f"  Top {i_img+1}: {os.path.basename(img['path'])} - Score: {img['score']}")

    if not top_10_images:
        logging.warning("No images in top 10 after scoring.")
        return []
    if len(top_10_images) <= 5:
        logging.info(f"Only {len(top_10_images)} images rated high enough (or total images < 10), returning all of them.")
        return [img['path'] for img in top_10_images]

    # --- Stage 2: Select best and diverse 5 from top 10 ---
    logging.info("Stage 2: Selecting 5 best and diverse images from the top 10.")
    top_10_paths = [img['path'] for img in top_10_images]
    selected_5_images = select_best_5_images_from_top(top_10_paths, topic, content, headers)

    if selected_5_images: #  None 也是 False
        logging.info(f"Final selected {len(selected_5_images)} images:")
        for i_img, path in enumerate(selected_5_images):
            logging.info(f"  Final selection {i_img+1}: {os.path.basename(path)}")
        return selected_5_images
    else:
        # select_best_5_images_from_top should return a list, even if empty.
        # If it returns None, this log will be odd. Ensure it always returns a list.
        logging.warning("GPT determined that no images met the quality criteria in stage 2, or an error occurred.")
        return []

def parse_image_ratings(response_text_json_str, batch_image_paths, valid_indices):
    """
    Parse the JSON string of image scores returned from GPT.
    response_text_json_str: JSON string returned by GPT API.
    batch_image_paths: List of all image paths in the current batch (including those that may not have been encoded).
    valid_indices: List of indices for images that were actually encoded and sent to GPT, referencing batch_image_paths.
    """
    image_ratings = []
    try:
        # response_text_json_str should be the JSON string returned by GPT
        data = json.loads(response_text_json_str)
        evaluations = data.get("image_evaluations")

        if not evaluations or not isinstance(evaluations, list):
            logging.error(f"JSON response does not contain 'image_evaluations' list or it's not a list. Response: {response_text_json_str[:300]}")
            return []

        for eval_item in evaluations:
            if not isinstance(eval_item, dict):
                logging.warning(f"Skipping non-dict item in image_evaluations: {eval_item}")
                continue

            try:
                img_num = eval_item.get("image_number") # GPT returns 1-based index
                score = eval_item.get("score")
                # justification = eval_item.get("justification", "")
                # problematic_text = eval_item.get("contains_problematic_text", False)

                if img_num is None or score is None:
                    logging.warning(f"Missing 'image_number' or 'score' in evaluation item: {eval_item}")
                    continue

                img_num = int(img_num)
                score = float(score)

                # Map GPT's 1-based img_num to valid_indices, then use valid_indices to get the actual path from batch_image_paths
                if 1 <= img_num <= len(valid_indices):
                    # valid_indices stores original indices (0-based) of batch_image_paths that were sent
                    # e.g., if valid_indices is [0, 2, 3], batch_image_paths[0], batch_image_paths[2], batch_image_paths[3] were sent
                    # GPT returns image_number = 1 -> valid_indices[0]
                    # GPT returns image_number = 2 -> valid_indices[1]
                    original_batch_idx_for_gpt_img_num = valid_indices[img_num - 1]

                    if original_batch_idx_for_gpt_img_num < len(batch_image_paths):
                        img_path = batch_image_paths[original_batch_idx_for_gpt_img_num]
                        image_ratings.append({
                            'path': img_path,
                            'score': score,
                            'original_index_in_batch': original_batch_idx_for_gpt_img_num
                        })
                        logging.debug(f"Successfully parsed rating for GPT image_number {img_num} (path: {os.path.basename(img_path)}): {score}/10")
                    else:
                        # This should not happen; valid_indices should always be valid
                        logging.warning(f"Logic error: original_batch_idx_for_gpt_img_num {original_batch_idx_for_gpt_img_num} out of range for batch_image_paths (len: {len(batch_image_paths)}).")
                else:
                    logging.warning(f"Parsed image_number {img_num} from GPT is out of range for the number of images sent ({len(valid_indices)} images). Item: {eval_item}")
            except (ValueError, TypeError) as e:
                logging.error(f"Error parsing item in image_evaluations: {eval_item} - {e}")
            except Exception as e: # General catch
                logging.error(f"Unexpected error processing evaluation item {eval_item}: {e}")


    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response for image ratings: {e}. Response text: {response_text_json_str[:500]}")
        return [] # Return empty list if JSON parsing fails
    except Exception as e:
        logging.error(f"Unexpected error in parse_image_ratings: {e}. Response: {response_text_json_str[:500]}")
        return []


    if not image_ratings:
        logging.info(f"Could not parse any ratings from JSON. GPT response excerpt: '{response_text_json_str[:300]}'")
        
    return image_ratings

def select_best_5_images_from_top(image_paths_top_10, topic, content, headers): 
    """
    Selects the best and most diverse set of images from the top-rated images.
    Returns a list of selected image paths (may be fewer than 5 or empty).
    """
    if not image_paths_top_10: # If input list is empty
        return []
    if len(image_paths_top_10) <= 5:
        logging.info(f"Less than or equal to 5 images in top_10 ({len(image_paths_top_10)}), returning all.")
        return image_paths_top_10

    # --- Prompt for selecting high-quality diverse images ---
    # Ensure {topic} and {content} are replaced correctly
    current_topic = topic if topic is not None else "N/A"
    current_content = content if content is not None else "N/A"

    select_prompt_text = f"""
**Objective:**
You are an expert visual curator with a CRITICAL task: to select a final set of images (0 to 5 images) for a Question-Answering (QA) system. The images you select MUST be of high quality and relevance, and CRUCIALLY, they must NOT violate any of the strict exclusion criteria. The goal is to test an LLM's visual understanding, so images with textual clues or quality issues are detrimental.

**Input:**
You will be provided with a set of pre-screened images. Each image will be numbered sequentially starting from 1 based on the order it is presented to you.

**Topic:**
"{current_topic}"

**Content Description:**
"{current_content}"

**CRITICAL Requirements (STRICTLY ENFORCE):**

1. **NO Textual Interference:** * REJECT images with significant text overlays, captions or labels that directly provide answers
   * Small background text is acceptable if not prominent

2. **NO Multiple Similar Images:**
   * CRITICAL: DO NOT select multiple images of the same object/person/scene
   * If you see multiple images of the same subject (e.g., same presenter, same product, same diagram), select ONLY ONE (the best one)
   * Each selected image MUST show different subjects or completely different perspectives

3. **Maximum Diversity Required:**
   * Selected images must be visually diverse from each other
   * Each image should contribute unique visual information

**Selection Guidelines:**
* Select UP TO 5 images that meet ALL criteria above
* It's better to select FEWER high-quality diverse images than to include lower quality or similar ones
* If NO images meet the quality threshold, return "No suitable images found"

**Output Format (FOLLOW EXACTLY):**
* If selecting images: `Selected Images: 2,5,1` (listing image numbers in order of preference, 1-based index from the input to this stage)
* If no images meet criteria: `No suitable images found.`
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": select_prompt_text},
            ]
        }
    ]

    # Limit to processing at most 10 images (as before)
    image_paths_to_send = image_paths_top_10[:10] # Python slicing is safe for out-of-range

    valid_images_sent_for_selection = [] # Track paths of images that were actually sent for selection

    for path in image_paths_to_send:
        base64_image = encode_image_to_base64(path)
        if base64_image:
            mime_type = "image/jpeg"
            if path.lower().endswith(".png"):
                mime_type = "image/png"
            elif path.lower().endswith(".webp"):
                mime_type = "image/webp"
            image_url = f"data:{mime_type};base64,{base64_image}"
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
            valid_images_sent_for_selection.append(path) # Track successfully sent image path
        else:
            logging.warning(f"Could not encode image for final selection: {os.path.basename(path)}")

    if not valid_images_sent_for_selection:
        logging.error("No images could be encoded and sent for the final selection stage.")
        return []

    payload = {
        "model": GPT_MODEL,
        "messages": messages,
        "max_tokens": 200 # 稍微增加一点，以防理由过长或有其他文本
    }

    try:
        logging.info(f"Sending request to select best diverse high-quality images from {len(valid_images_sent_for_selection)} candidates...")
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        response = session.post(API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            content_response = result['choices'][0]['message']['content']
            logging.debug(f"GPT Response for selecting (Stage 2): {content_response}")
            
            if "no suitable images found" in content_response.lower(): # Case-insensitive match
                logging.info("GPT (Stage 2) determined that no images meet the quality threshold or diversity criteria.")
                return []
                
            selected_indices_1based = parse_final_selected_image_indices(content_response, len(valid_images_sent_for_selection))
            logging.info(f"GPT (Stage 2) returned 1-based indices: {selected_indices_1based}")

            selected_paths = []
            if selected_indices_1based:
                # Ensure indices are unique and within valid range
                # selected_indices_1based is 1-based, referencing valid_images_sent_for_selection
                seen_indices_0based = set()
                for idx_1based in selected_indices_1based:
                    if 1 <= idx_1based <= len(valid_images_sent_for_selection):
                        idx_0based = idx_1based - 1
                        if idx_0based not in seen_indices_0based:
                            selected_paths.append(valid_images_sent_for_selection[idx_0based])
                            seen_indices_0based.add(idx_0based)
                        else:
                            logging.debug(f"Skipping duplicate image index from Stage 2 GPT response: {idx_1based}")
                    else:
                        logging.warning(f"GPT (Stage 2) returned invalid image index: {idx_1based}, valid range is 1-{len(valid_images_sent_for_selection)}")
                
                return selected_paths
            else:
                logging.warning("Could not parse any valid image indices from GPT Stage 2 response.")
                return []
        else:
            logging.error("API response (Stage 2) format error or no choices in response.")
            return []
            
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for final selection (Stage 2): {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"  Response content: {e.response.text}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during final image selection (Stage 2): {e}")
        return [] # Always return a list

def parse_final_selected_image_indices(response_text, num_images_actually_sent):
    """Parses the 'Selected Images: N,N,N' format from GPT response."""
    indices = []
    # More flexibly match "Selected Images:" (optional, case-insensitive) and number list
    match = re.search(r"(?:[Ss]elected\s*[Ii]mages?\s*[:\-]?\s*)?((?:\d+\s*,\s*)*\d+)", response_text.strip())

    if match:
        numbers_str = match.group(1) if match.lastindex == 1 else match.group(0) # Handle number-only case
        numbers_str = re.sub(r'[^\d,]', '', numbers_str) # Remove non-digit and non-comma characters
        try:
            raw_indices = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
            # Filter valid indices (1-based, not exceeding actual number sent)
            indices = [idx for idx in raw_indices if 1 <= idx <= num_images_actually_sent]
            
            # Remove duplicates while preserving order
            unique_indices = []
            seen = set()
            for idx in indices:
                if idx not in seen:
                    unique_indices.append(idx)
                    seen.add(idx)
            indices = unique_indices

            if len(indices) > 5:
                logging.warning(f"GPT returned more than 5 indices ({len(indices)}), will use the first 5 valid ones: {indices[:5]}")
                indices = indices[:5]
            elif len(indices) > 0:
                 logging.info(f"GPT returned {len(indices)} indices for final selection: {indices}")
            else: # raw_indices had content but filtering produced nothing, or parsing failed
                 logging.warning(f"No valid numeric indices found in the selection string '{numbers_str}' after filtering (num_images_sent: {num_images_actually_sent}).")

        except ValueError:
            logging.error(f"Could not parse numbers from selection string: '{numbers_str}'")
            return []
    else:
        # If main pattern fails, try to extract all numbers from the text as a fallback
        logging.warning(f"Could not find 'Selected Images:' pattern or a clear list of numbers in response: \"{response_text[:150]}...\" Trying fallback.")
        all_numbers_in_response = re.findall(r'\b(\d+)\b', response_text)
        if all_numbers_in_response:
            try:
                raw_indices = [int(n) for n in all_numbers_in_response]
                indices = [idx for idx in raw_indices if 1 <= idx <= num_images_actually_sent]
                
                unique_indices = []
                seen = set()
                for idx in indices:
                    if idx not in seen:
                        unique_indices.append(idx)
                        seen.add(idx)
                indices = unique_indices
                
                if indices:
                    logging.info(f"Fallback parsing found indices: {indices}. Will use up to the first 5 distinct.")
                    if len(indices) > 5 : indices = indices[:5]
                else:
                    logging.warning("Fallback parsing found numbers, but none were valid indices.")
            except ValueError:
                logging.warning("Fallback parsing of numbers failed (ValueError).")
                return []
        else:
            logging.warning("Fallback parsing found no numbers in the response.")
            return []

    return indices


#
# --- Main Processing Logic ---

def find_final_metadata_file(directory):
    """Find the final_metadata file in the directory."""
    final_metadata_files = glob.glob(os.path.join(directory, "*_final_metadata.json"))
    if final_metadata_files:
        final_metadata_files.sort(key=len)
        return final_metadata_files[0]
    return None

def extract_content_from_final_metadata(metadata_file):
    """Extract the topic and content fields from the final_metadata file."""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            topic = metadata.get("topic", "").strip()
            content = metadata.get("content", "").strip()
            if not topic and content: # If topic is empty but content exists
                topic = (content[:75] + "...") if len(content) > 75 else content
            elif not content and topic: # If content is empty but topic exists
                content = topic
            elif not topic and not content:
                return "Unknown Topic", "No content available"
            return topic, content
    except Exception as e:
        logging.error(f"Error reading metadata file {metadata_file}: {e}")
        return "Unknown Topic", "No content available"

def process_directory(root_dir, *, clean_selected0=False, skip_processed=False, max_img_per_req=5, start_index=0, num_workers=1):
    """Main processing function for all subdirectories."""
    logging.info("Logger initialized.")

    logging.info(f"Starting processing for directory: {root_dir}")
    logging.info(f"Selected images will be saved in 'selected0' folder within each subdirectory.")

    if not os.path.isdir(root_dir):
        logging.error(f"Root directory not found: {root_dir}")
        return

    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]) # Sort for consistent order
    total_subdirs = len(subdirs)
    logging.info(f"Found {total_subdirs} subdirectories.")
    
    # Start from start_index
    if start_index > 0:
        if start_index >= len(subdirs):
            logging.error(f"Start index {start_index} exceeds the number of subdirectories ({len(subdirs)})")
            return
        subdirs = subdirs[start_index:]
        logging.info(f"Starting from index {start_index}, remaining {len(subdirs)} subdirectories")

    if num_workers > 1:
        logging.info(f"Using {num_workers} parallel workers")
        # Prepare argument list
        args_list = [(i + start_index, subdir_name, root_dir, clean_selected0, skip_processed, max_img_per_req) 
                     for i, subdir_name in enumerate(subdirs)]
        
        # Use multiprocessing
        with Pool(processes=num_workers) as pool:
            pool.map(process_single_directory, args_list)
    else:
        # Single-threaded processing
        for i, subdir_name in enumerate(subdirs):
            process_single_directory((i + start_index, subdir_name, root_dir, clean_selected0, skip_processed, max_img_per_req))

    logging.info("\nAll subdirectory processing finished.")

def process_single_directory(args):
    """Process a single subdirectory. Used for parallel processing."""
    i, subdir_name, root_dir, clean_selected0, skip_processed, max_img_per_req = args
    
    subdir_path = os.path.join(root_dir, subdir_name)
    logging.info(f"\n[{i+1}] Processing subdirectory: {subdir_name}")
    
    current_output_dir = os.path.join(subdir_path, "selected0")

    if skip_processed and os.path.isdir(current_output_dir) and os.listdir(current_output_dir):
        logging.info(f"  Skipping '{subdir_name}' ('selected0' already exists and is not empty).")
        return

    if clean_selected0 and os.path.isdir(current_output_dir):
        try:
            shutil.rmtree(current_output_dir)
            logging.info(f"  Deleted existing 'selected0' folder.")
        except Exception as e:
            logging.warning(f"  Could not delete 'selected0': {e}")

    os.makedirs(current_output_dir, exist_ok=True)

    final_metadata_file = find_final_metadata_file(subdir_path)
    if not final_metadata_file:
        logging.warning(f"  No final_metadata file found in {subdir_name}. Skipping.")
        with open(os.path.join(current_output_dir, "_skipped_no_metadata.txt"), "w") as f:
            f.write("Skipped due to missing final_metadata.json file.")
        return

    topic, content = extract_content_from_final_metadata(final_metadata_file)
    if topic == "Unknown Topic" or content == "No content available":
        logging.warning(f"  Topic or Content is empty/default in metadata for {subdir_name} (from {os.path.basename(final_metadata_file)}). Skipping.")
        with open(os.path.join(current_output_dir, "_skipped_empty_metadata.txt"), "w") as f:
            f.write(f"Skipped due to empty topic/content in {os.path.basename(final_metadata_file)}.")
        return

    logging.info(f"  Extracted from {os.path.basename(final_metadata_file)}:")
    logging.info(f"    Topic: {topic[:100]}{'...' if len(topic)>100 else ''}")
    logging.info(f"    Content (first 100 chars): {content[:100]}{'...' if len(content)>100 else ''}")

    tag2_path = os.path.join(subdir_path, "tag2")
    if not os.path.isdir(tag2_path):
        logging.warning(f"  'tag2' directory not found in {subdir_name}. Skipping.")
        with open(os.path.join(current_output_dir, "_skipped_no_tag2_dir.txt"), "w") as f:
            f.write("Skipped due to missing 'tag2' directory.")
        return

    all_image_files = []
    for ext in IMAGE_EXTENSIONS:
        all_image_files.extend(glob.glob(os.path.join(tag2_path, ext)))
    all_image_files.sort() # Sort for consistent processing order

    if not all_image_files:
        logging.warning(f"  No images found in 'tag2' directory for {subdir_name}. Skipping.")
        with open(os.path.join(current_output_dir, "_skipped_no_images_in_tag2.txt"), "w") as f:
            f.write("Skipped due to no images found in 'tag2' directory.")
        return
    logging.info(f"  Found {len(all_image_files)} images in 'tag2'. Processing up to {max_img_per_req * ((len(all_image_files) + max_img_per_req -1) // max_img_per_req)} images in batches for Stage 1.")


    logging.info(f"  Filtering images using GPT (Model: {GPT_MODEL})...")
    selected_image_paths = call_gpt4_vision(topic, content, all_image_files, max_images_per_request=max_img_per_req)

    if selected_image_paths:
        logging.info(f"  GPT finally selected {len(selected_image_paths)} images for {subdir_name}: {[os.path.basename(p) for p in selected_image_paths]}")
        copied_count = 0
        for img_idx, src_path in enumerate(selected_image_paths):
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                # Sanitize filename for safety, though os.path.basename should be okay
                base, ext = os.path.splitext(filename)
                safe_base = re.sub(r'[^\w\-\.]', '_', base) # Keep alphanumeric, hyphen, dot; replace others with underscore
                
                dest_filename = f"selected_{img_idx+1}_{safe_base}{ext}"
                dest_path = os.path.join(current_output_dir, dest_filename)
                try:
                    shutil.copy2(src_path, dest_path)
                    # logging.info(f"    ✅ Copied: {filename} -> {dest_filename}") # Verbose
                    copied_count += 1
                except Exception as e:
                    logging.error(f"    ❌ Failed to copy file {filename} to {dest_path}: {e}")
            else:
                logging.warning(f"    ⚠️ Path '{src_path}' was selected by GPT but not found during copy phase for {subdir_name}.")
        logging.info(f"  Subdirectory {subdir_name} processing complete. Copied {copied_count} images to '{os.path.basename(current_output_dir)}'.")
        if copied_count == 0 and len(selected_image_paths) > 0:
             logging.warning(f"  GPT selected {len(selected_image_paths)} images, but 0 were copied. Check paths and permissions.")
        elif copied_count == 0: # No images selected by GPT
            with open(os.path.join(current_output_dir, "_gpt_selected_no_images.txt"), "w") as f:
                f.write("GPT processing completed, but no images were selected that met the criteria.")

    else: # selected_image_paths is empty or None
        logging.info(f"  GPT did not select any images for {subdir_name} that met the quality criteria.")
        with open(os.path.join(current_output_dir, "_gpt_selected_no_images.txt"), "w") as f:
            f.write("GPT processing completed, but no images were selected that met the criteria.")

# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use GPT-4 Vision to select images matching a topic and content.")
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR,
                        help="Root directory containing subdirectories with images and metadata.")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (overrides environment variable or script default).")
    parser.add_argument("--model", type=str, default=GPT_MODEL,
                        help=f"GPT model name to use (e.g., gpt-4o, gpt-4-vision-preview). Default: {GPT_MODEL}")
    parser.add_argument("--batch_size", type=int, default=3,
                        help="Maximum number of images per API request in the first stage (scoring). Default: 3")
    parser.add_argument("--clean_selected0", action="store_true",
                        help="If set, deletes any existing 'selected0' folder before processing a subdirectory.")
    parser.add_argument("--no_skip", action="store_true",
                        help="Process all subdirectories, even those that already have a non-empty 'selected0' folder.")
    # 添加新的命令行参数
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start processing from this index (0-based). Default: 0")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers to use. Default: 1")

    args = parser.parse_args()

    # Reset log handlers and setup logger before any logging.info or logging.warning
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    setup_logger()

    if args.api_key:
        API_KEY = args.api_key
        logging.info("Using API key from command-line argument.")
    elif not API_KEY or API_KEY == "YOUR_API_KEY_HERE": # Check if it's the placeholder
        logging.warning("⚠️ WARNING: OpenAI API key is not set or is a placeholder 'YOUR_API_KEY_HERE'.")
        logging.warning("Please set the OPENAI_API_KEY environment variable, use the --api_key argument, or edit the script.")
        # exit(1) # Consider exiting if API key is critical and missing/placeholder
    else:
        logging.info("Using API key from environment variable or script default.")

    GPT_MODEL = args.model # Update global GPT_MODEL based on arg
    logging.info(f"Using GPT model: {GPT_MODEL}")

    if args.batch_size <= 0:
        logging.warning("⚠️ WARNING: Batch size must be greater than 0. Setting to default (3).")
        args.batch_size = 3
    
    # 根据系统CPU核心数自动确定默认工作进程数量
    if args.num_workers <= 0:
        args.num_workers = max(1, min(cpu_count() - 1, 8))  # 默认使用CPU核心数-1，但最多8个
        logging.info(f"Auto-detected optimal worker count: {args.num_workers}")
    
    logging.info(f"Processing root directory: {args.root_dir}")
    logging.info(f"Starting from index: {args.start_index}")
    logging.info(f"Using {args.num_workers} parallel workers")
    logging.info(f"Batch size for Stage 1 scoring: {args.batch_size}")
    logging.info(f"Clean 'selected0' before processing: {args.clean_selected0}")
    logging.info(f"Skip processed subdirectories (if 'selected0' exists and is not empty): {not args.no_skip}")

    process_directory(
        args.root_dir,
        clean_selected0=args.clean_selected0,
        skip_processed=not args.no_skip, # no_skip flag means skip_processed is False
        max_img_per_req=args.batch_size,
        start_index=args.start_index,
        num_workers=args.num_workers
    )
