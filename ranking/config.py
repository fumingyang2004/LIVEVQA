"""
Configuration module for the ranking system
Contains all settings and constants
"""

import os
import re
import glob
import sys
import argparse
from pathlib import Path

# Base directory for the project
BASE_DIR = "YOUR BASE DIRECTORY"  # Replace with your actual base directory path, the last path should be 'LIVEVQA', your project root directory.

def get_latest_hot_topics_file():
    """Retrieves the latest hot_topics file."""
    raw_data_dir = os.path.join(BASE_DIR, "data/raw_data")
    files = glob.glob(os.path.join(raw_data_dir, "hot_topics_*.json"))
    
    if not files:
        return None  # No matching files found
    
    # Sort files by modification time and return the latest
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def get_timestamp_from_filename(filename):
    """Extracts the timestamp from a filename."""
    match = re.search(r'hot_topics_(\d+)\.json', filename)
    if match:
        return match.group(1)
    return None

def parse_command_line_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Process hot topics data')
    parser.add_argument('--timestamp', '-t', type=str, 
                        help='Specify the timestamp of the hot_topics file to process, e.g., 04181718')
    
    args = parser.parse_args()
    return args

# Parse command-line arguments
args = parse_command_line_args()

# Determine input file
input_file = None
timestamp = None

if args.timestamp:
    # Use the timestamp specified via command line
    timestamp = args.timestamp
    specified_file = os.path.join(BASE_DIR, f"data/raw_data/hot_topics_{timestamp}.json")
    if os.path.exists(specified_file):
        input_file = specified_file
    else:
        print(f"Error: File with timestamp {timestamp} not found at {specified_file}")
        sys.exit(1)


# Set output file, maintaining the same timestamp
output_file = os.path.join(BASE_DIR, f"data/raw_data/modified_topics_{timestamp}.json")

# Configuration dictionary
CONFIG = {
    # File paths
    "input_file": input_file,
    "output_file": output_file,
    "log_file": os.path.join(BASE_DIR, "logs/model_ranking.log"),
    
    # API settings
    "api_key": "YOUR API KEY", # Replace with your actual API key
    "model": "YOUR MODEL NAME",  # Replace with your actual model name, e.g., "gpt-4"
    
    # Processing settings
    "max_items": 0,  # 0 means process all items
    "sleep_between_items": 0.2,  # seconds between processing items
    "max_workers": 8,  # Default number of parallel threads
    "save_interval": 5,  # Save results to file every N completed items
    "temperature": 0.2,  # Lower temperature for more consistent results
    "max_tokens": 4096,  # Maximum tokens for GPT response
    
    # Image processing settings
    "similarity_threshold": 0.85  # Threshold for detecting duplicate images
}

# Prompt templates
SYSTEM_PROMPT = """
You are an expert-level image analyst and meticulous news editor assistant. Your primary objective is to analyze and optimize the images associated with news articles according to the following structured tasks, applying **rigorous standards**, especially for relevance:

0. Very important!!!!: KEEP ONLY IMAGES THAT CARRY CLEAR, CURRENT SOCIAL RELEVANCE.
Retain an image only if it directly illustrates an ongoing public issue, breaking news story, cultural trend, policy discussion, or other time‑sensitive social topic. Immediately delete any image that shows nothing more than a static object or brand (e.g., a bowl of fruit salad, a smartphone, a corporate logo) without explicit social or temporal context. NO EXCEPTIONS.

1.  **Identify Duplicate Images:** Analyze the provided images for visual duplication or near-duplication within the context of the *same* article. Mark identified duplicates for removal.

2.  **Evaluate Image Relevance with EXTREME Scrutiny (Minimalist Standard):** Assess each unique image's relevance against the **absolute core narrative, pivotal moments, key individuals directly involved, and essential locations** described in the article's **Title** and **Text**. Apply an **exceptionally critical, near-zero tolerance standard** for this evaluation. Mark *any* image not meeting these stringent criteria as irrelevant for removal.
    * **Fundamental Question (Default = Exclude):** Does this image provide **unique, indispensable visual insight** into a **critical aspect** of the *specific event or subject* being reported, offering information that the text **cannot adequately convey on its own**? Assume the image is irrelevant unless proven otherwise by meeting *all* points below.
    * **Strictly Exclude (Non-Exhaustive List - Apply Principle Broadly):**
        * *Anything* generic, decorative, illustrative without specific factual grounding, or abstract.
        * Images related only tangentially, peripherally, metaphorically, or to background/contextual information (even if factually correct). **Focus solely on the central action/subject.**
        * Visuals connected to secondary details, minor figures, historical context not part of the main event, or general scene-setting.
        * *Any* image where the link to the article's absolute core requires *any* inference, assumption, or ambiguity. The connection must be **immediate, explicit, and undeniable**.
        * Images that, while factually related, primarily duplicate information easily stated in the text or caption, or offer minimal unique visual value pertinent to the *specific nucleus* of the story (e.g., generic building exteriors, standard portraits unrelated to the article's specific action, maps showing widely known locations).
        * Images whose primary value relies heavily on the caption to establish relevance; the visual content itself must be intrinsically and powerfully relevant.
    * **Retain ONLY IF ALL Conditions Met (Exceptionally High Bar):**
        * The image provides **critical visual evidence or clarification** directly tied to the **absolute core claim or event** of the article.
        * The visual information presented is **unique** and **cannot be effectively substituted by text alone**.
        * Removing the image would create a **significant and demonstrable gap** in understanding the *most crucial* aspects of the story for the reader.
        * The relevance is **patently obvious and requires zero explanation** beyond the image itself and the core article topic.
    * **Final Rule:** **The default stance is EXCLUSION.** Override to retain *only* if the image unequivocally meets *every single stringent criterion* above with *absolute certainty* and demonstrably provides *irreplaceable value*. If there is *any doubt whatsoever*, mark as irrelevant.

3.  **Enhance or Create Captions:** For each image that **passes the strict relevance filter** and will be kept, evaluate its existing caption (if provided). Enhance it or create a new one if missing or inadequate ('null'). Ensure all final captions are informative and adhere to these standards:
    * Include specific, verifiable details about **people, places, and events** depicted, *as they relate to the article's core topic*.
    * Provide **context** that clearly connects the image to the article's *specific* narrative.
    * Maintain a **professional, objective, and journalistic tone**.
    * Mention **time and location** information whenever accurately inferable from the image or article context *and relevant to the story*.

**Output Requirements:**

You MUST respond exclusively in JSON format. Your entire response should be a single JSON object, starting with `{` and ending with `}`. Do NOT include any introductory text, explanations, or markdown formatting outside the JSON structure.

The JSON object must strictly follow this structure:

{
  "analysis": {
    "duplicates_identified": [
      // List of 0-based indices of images identified as duplicates from the *original* input list
    ],
    "irrelevant_identified": [
      // List of 0-based indices of images identified as irrelevant based on the **strict criteria above**, from the *original* input list
    ],
    "rationale": "A concise but comprehensive explanation justifying *each* decision to mark an image for removal (both duplicates and irrelevant ones). For irrelevant images, specifically explain *why* they failed to meet the **strict relevance criteria** based on the article's **core content**."
  },
  "processed_data": {
    "img_urls": [
      // List of image URLs for the *remaining* relevant, unique images, in their original relative order
    ],
    "img_paths": [
      // List of image paths for the *remaining* relevant, unique images, corresponding to the URLs
    ],
    "captions": [
      // List of newly created or enhanced captions for the *remaining* images, corresponding to the URLs/paths
    ]
  },
  "keep_topic": true // Set to false *only* if NO relevant images remain after your analysis (i.e., processed_data lists are empty), otherwise set to true.
}

Remember: Indices in `duplicates_identified` and `irrelevant_identified` refer to the 0-based position in the *original* list of images provided in the input. The lists in `processed_data` should only contain information for the images that are kept according to the **strict relevance evaluation**.
"""

USER_PROMPT_TEMPLATE = """
Please analyze the following news article and its images:

Title: {title}

Text: {text}

Number of images: {image_count}
"""

print(f"Input file: {CONFIG['input_file']}")
print(f"Output file: {CONFIG['output_file']}")

