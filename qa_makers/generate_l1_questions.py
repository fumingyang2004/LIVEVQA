"""
Level 1 Question Generator

This script generates level 1 questions for news topics with images,
focusing on questions that require social knowledge to answer.
"""

import os
import sys
import json
import base64
import logging
import argparse
import threading
import time
import glob
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import BASE_DIR, DATA_DIR, CONFIG

# Add project root path to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project configuration
from ranking.client import setup_client, get_thread_client, call_gpt_with_retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR+"/logs", "l1_question_generator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Synchronization locks
output_lock = threading.Lock()
save_lock = threading.Lock()
last_save_time = time.time()

# Get the latest modified_topics file
def get_latest_modified_topics_file():
    """Gets the latest modified_topics file"""
    files = glob.glob(os.path.join(DATA_DIR, "modified_topics_*.json"))
    if not files:
        return None  # No matching file found
    # Sort by file modification time, return the latest file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

# Extract timestamp from filename
def get_timestamp_from_filename(filename):
    """Extracts timestamp from filename"""
    match = re.search(r'modified_topics_(\d+)\.json', filename)
    if match:
        return match.group(1)
    return None

# Determine input and output files
def determine_io_files(args):
    """Determines input and output file paths"""
    input_file = None
    timestamp = None
    
    if args.timestamp:
        # Use timestamp specified in command line
        timestamp = args.timestamp
        specified_file = os.path.join(DATA_DIR, f"modified_topics_{timestamp}.json")
        if os.path.exists(specified_file):
            input_file = specified_file
        else:
            logger.error(f"Error: File with timestamp {timestamp} not found: {specified_file}")
            sys.exit(1)
    elif args.input:
        # Use input file specified in command line
        input_file = args.input
        # Try to extract timestamp from input filename
        timestamp = get_timestamp_from_filename(input_file)
    else:
        # Use the latest file
        input_file = get_latest_modified_topics_file()
        if input_file:
            timestamp = get_timestamp_from_filename(input_file)
        else:
            logger.error("Error: No modified_topics file found")
            sys.exit(1)
    
    # Set output file, keep the same timestamp
    if args.output:
        # If output file is specified, prioritize it
        output_file = args.output
    elif timestamp:
        # Use the same timestamp
        output_file = os.path.join(DATA_DIR, f"l1_topics_{timestamp}.json")
    else:
        # Use default output filename
        output_file = os.path.join(DATA_DIR, "l1_topics.json")
    
    return input_file, output_file, timestamp


def encode_image_to_base64(image_path):
    """Encodes an image to a base64 string"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image file does not exist: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error {image_path}: {str(e)}")
        return None

def create_prompt_for_topic(topic, img_index, used_question_types=None, used_questions=None):
    """Creates a prompt for a given topic and image index"""
    
    system_prompt = """You are an AI assistant specialized in generating high-quality Level 1 multi-hop questions that require social knowledge to answer. Your task is to create image-and-text-based questions that focus on factual information rather than inference or reasoning.

Your generated question MUST follow these strict requirements:

1. Question format: Always start with "Based on the provided image, " followed by a clear, concise question
2. Answer source: The answer MUST be explicitly findable in the provided text (not just inferrable)
3. Answer format: The answer must be a short phrase or a few words (NOT a sentence or paragraph)
4. Question categories: The question MUST belong to one of these categories ONLY:
    - location (where something is happening)
    - person (who is in the image, but avoid asking about very famous people like Trump or Musk)
    - organization (which company, team, group, etc.)
    - time (when something occurred)
    - object (what specific item is shown)
    - event (ONLY allowed to ask "what event is taking place?")

5. Question simplicity: The question must be concise and avoid revealing too many details from the article
6. Required integration: Question must relate to what can be seen in the image, while having an answer in the text
7. Knowledge requirement: The question should test knowledge that cannot be directly answered by computer vision alone

CRUCIAL QUALITY CRITERIA - AVOID THESE COMMON ISSUES:
1. FAMOUS FIGURES: DO NOT create questions asking about extremely well-known figures (e.g., "who is this person?" when Donald Trump is in the image). These are too obvious.

2. SPECIFIC ANSWERS ONLY: Ensure answers are HIGHLY SPECIFIC and uniquely identifiable. AVOID vague/generic answers like:
    - BAD: "Designer sneakers", "high-end sneakers" (too generic, could be many brands)
    - GOOD: "Nike Air Force 1", "Louis Vuitton Trainers" (specific identifiable items)
    
3. TEMPORAL CONTEXT REQUIRED: NEVER create questions about images that lack clear temporal context. 
    - AVOID: Close-up images of food, products, or objects with no time-specific indicators
    - ESPECIALLY AVOID: Questions like "what food is this?" for a generic food close-up

4. NO COUNTING QUESTIONS: Never create questions asking to count objects in the image (e.g., "how many people are in the image?")

5. AVOID BOOK COVER QUESTIONS: Don't ask about book covers with answers like "book cover", "memoir cover", or "book jacket"

6. NO VISIBLE TEXT ANSWERS: Don't create questions whose answers appear as visible text in the image (e.g., asking about a company when its logo and name are clearly visible)

7. SPECIFIC LOCATIONS ONLY: Location answers must be specific places, not generic establishment types
    - BAD: "textile factory", "shopping mall", "clothing store", "garment factory" (generic)
    - GOOD: "Nike Factory in Vietnam", "Galeries Lafayette in Paris" (specific identifiable locations)

8. SPECIFIC EVENT IDENTIFIERS: When asking about events, answers should be specific named events
    - BAD: "stunt performance", "protest", "fashion show" (generic event types)
    - GOOD: "2023 Paris Fashion Week", "Black Lives Matter protest in Portland" (specific identifiable events)

9. NO CHART DATA QUESTIONS: Don't ask questions about data that is already visible in charts or graphs shown in the image

10. COMPLETE CONTENT REQUIRED: Ensure the topic has both questions and images

11. SPECIFIC PEOPLE IDENTIFIERS: When asking about people, answers must be specific named individuals, not job titles
    - BAD: "police officer", "protestor", "doctor" (generic roles)
    - GOOD: "Emmanuel Macron", "Taylor Swift" (specific identifiable people)

12. AVOID ERROR PATTERN EXAMPLES:
    - ❌ "Based on the provided image, who is speaking at the podium?" → "President Donald Trump" (too obvious)
    - ❌ "Based on the provided image, what type of footwear is shown?" → "Designer sneakers" (too vague)
    - ❌ "Based on the provided image, what dish is being prepared?" → "Pizza" (food close-up without context)
    - ❌ "Based on the provided image, how many protesters are visible?" → "24" (counting question)
    - ❌ "Based on the provided image, what is shown on the book cover?" → "Book jacket" (generic book cover)
    - ❌ "Based on the provided image, what company logo is displayed?" → "Google" (visible in image)
    - ❌ "Based on the provided image, what type of factory is shown?" → "Clothing factory" (generic location)
    - ❌ "Based on the provided image, what event is taking place?" → "A protest" (generic event)
    - ❌ "Based on the provided image, what does the graph show?" → "Rising stock price" (chart data)
    - ❌ "Based on the provided image, who is the person in uniform?" → "Police officer" (generic descriptor)

NEW CRITICAL REQUIREMENTS:
13. DO NOT include excessive article details in your questions
14. DO NOT mention specific names, dates, or unique details from the article in the question itself
15. Create questions that could stand alone with just the image, without requiring the article context
16. Questions should be generic enough that they don't reveal the answer within the question
17. AVOID generating questions similar to those already created for other images in this topic

EXAMPLES OF BAD QUESTIONS (TOO MUCH INFORMATION REVEALED):
- "Based on the provided image, what is the name of the memorial site where the graves of Zambia's 1993 national football team are located?" (reveals too much specific context)
- "Based on the provided image, who is the CEO that announced the company's new AI strategy at the June conference?" (reveals too many details)

EXAMPLES OF GOOD QUESTIONS (APPROPRIATE BALANCE):
- "Based on the provided image, what is the location shown?" (simple, focused on image)
- "Based on the provided image, who is the person at the podium?" (asks about visible element without revealing context)
- "Based on the provided image, what organization does this logo represent?" (focused on visual element)
- "Based on the provided image, what event is taking place?" (standard event question)

AVOID these types of questions:
- Questions about visible attributes (clothing color, number of people, etc.)
- Questions with ambiguous or subjective answers
- Questions that can be answered without social/factual knowledge
- Questions about extremely obvious information
- Questions whose answers are directly visible as text in the image
"""

    # Get topic information
    title = topic.get('topic', 'No title')
    text = topic.get('text', 'No text')
    img_paths = topic.get('img_paths', [])
    img_urls = topic.get('img_urls', [])
    captions = topic.get('captions', [])
    
    # Check if image index is valid
    if img_index >= len(img_paths) or not img_paths[img_index]:
        return None
    
    # Get information for the current image
    img_path = img_paths[img_index]
    img_url = img_urls[img_index] if img_index < len(img_urls) else ""
    caption = captions[img_index] if img_index < len(captions) else "No caption"
    
    # Build information about already used question types
    used_types_info = ""
    if used_question_types:
        used_types_str = ", ".join([f"'{qt}'" for qt in used_question_types])
        used_types_info = f"\nALREADY USED QUESTION TYPES: {used_types_str}"
    
    # List of already used questions
    used_questions_info = ""
    if used_questions and len(used_questions) > 0:
        used_questions_str = "\n- " + "\n- ".join([f'"{q}"' for q in used_questions])
        used_questions_info = f"\nQUESTIONS ALREADY GENERATED FOR OTHER IMAGES IN THIS TOPIC: {used_questions_str}"
    
    # Build user prompt - in English
    user_prompt = f"""Please generate a Level 1 multi-hop question based on the following news article and image. This question should test social knowledge rather than just visual perception.

ARTICLE TITLE: {title}

ARTICLE TEXT: {text}

IMAGE PATH: {img_path}
IMAGE URL: {img_url}
IMAGE CAPTION: {caption}{used_types_info}{used_questions_info}

REQUIREMENTS:
1. The question MUST start with "Based on the provided image, "
2. The answer MUST be explicitly found in the article text
3. The answer must be a short phrase or a few words (not a sentence)
4. The question must belong to one of these categories only: location, person, organization, time, object, or event
5. If asking about an event, the question must be "what event is taking place?"

CRITICAL QUALITY CONSTRAINTS:
1. DO NOT ask about obvious public figures (e.g., "who is this?" for Donald Trump)
2. ENSURE answers are specific and uniquely identifiable (e.g., "Nike Factory in Vietnam", not just "factory")
3. DO NOT create questions for images lacking temporal context (e.g., food close-ups, generic product shots)
4. NEVER include counting questions ("how many people/objects...")
5. AVOID book cover questions with generic answers like "book jacket"
6. DO NOT create questions whose answers are directly visible in the image as text/logos
7. Location answers must be specific places, not generic types like "shopping mall" or "clothing store"
8. Event answers must be specific named events, not generic types like "protest" or "fashion show"
9. DO NOT ask about data already visible in charts or graphs
10. People answers must be specific named individuals, not job roles like "police officer" or "doctor"

CRITICAL CONSTRAINTS:
11. Create a SIMPLE, CONCISE question that does NOT reveal too much information from the article
12. DO NOT include specific details, names, dates or unique information from the article in your question
13. The question should work as a standalone with just the image (we are creating a benchmark where users will only see the image and question)
14. Focus on what can be visually identified in the image, while ensuring the answer is in the text
15. Avoid questions that reveal the answer or provide too much context about the subject
16. VERY IMPORTANT: Your question MUST be substantially different from questions already generated for other images in this topic
17. DO NOT ask about the same people, objects, or locations that were already asked about in previous questions for this topic

BAD EXAMPLE: "Based on the provided image, what is the name of the memorial site where the graves of Zambia's 1993 national football team are located?"
GOOD EXAMPLE: "Based on the provided image, what is this memorial site called?"

Please provide your response in the following JSON format:

```json
{{
  "question": "Based on the provided image, [your simple, concise question]?",
  "question_type": "[category: location/person/organization/time/object/event]",
  "options": [
    "A. [option A]",
    "B. [option B]",
    "C. [option C]",
    "D. [option D]",
    "E. [option E]"  
  ],
  "Ground_Truth": "[correct letter, e.g., A]",
  "Ground_Truth_List": ["[correct answer]", "[alternative phrasing 1]", "[alternative phrasing 2]", ...]
}}
```

IMPORTANT FORMAT INSTRUCTIONS:
1. Include 3-5 multiple-choice options, with one being the correct answer, the position of the correct answer can be randomized, i.e. A~E can be.
2. Make incorrect options plausible and challenging to distinguish
3. The Ground_Truth_List should include multiple valid phrasings of the answer (up to 10)
4. If you cannot create a suitable question, return: {{"error": "Unable to generate an appropriate question"}}
5. Ensure all content is in English
"""

    return {"system": system_prompt, "user": user_prompt, "img_path": img_path}

def generate_question_for_image(client, prompt_data):
    """Generates a question for a single image"""
    system_prompt = prompt_data["system"]
    user_prompt = prompt_data["user"]
    img_path = prompt_data["img_path"]
    
    # Create message list
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Add image
    base64_image = encode_image_to_base64(img_path)
    if base64_image:
        messages.append({
            "role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    
    # Call API
    try:
        response = call_gpt_with_retry(
            client, 
            messages, 
            max_retries=3, 
            retry_delay=2
        )
        
        if not response:
            logger.error("API call failed")
            return None
            
        content = response.choices[0].message.content
        
        # Extract JSON part
        import re
        json_match = re.search(r'```json\n([\s\S]*?)\n```|(\{[\s\S]*\})', content)
        
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            try:
                result = json.loads(json_str)
                
                # Check for errors
                if "error" in result:
                    logger.warning(f"Unable to generate question: {result['error']}")
                    return None
                    
                return result
            except json.JSONDecodeError:
                logger.error(f"JSON parsing error: {content[:200]}")
        else:
            logger.error(f"Could not extract JSON from response: {content[:200]}")
            
        return None
    except Exception as e:
        logger.error(f"Error generating question: {str(e)}")
        return None

def process_topic(topic, output_data):
    """Processes all images for a single topic"""
    # Check if it's a placeholder topic
    if topic.get('id') is not None and topic.get('topic') is None:
        logger.info(f"Skipping placeholder topic ID: {topic.get('id')}")
        # Add to output for consistency but don't generate questions
        topic_copy = {k: topic.get(k) for k in topic.keys()}
        topic_copy["level1_qas"] = []
        
        with output_lock:
            output_data.append(topic_copy)
        return True
    
    # Create a copy of the topic
    topic_copy = {k: topic.get(k) for k in topic.keys()}
    topic_copy["level1_qas"] = []
    
    # Get client
    client = get_thread_client()
    
    # Get image paths
    img_paths = topic.get('img_paths', [])
    
    if not img_paths:
        logger.info(f"Topic has no images: {topic.get('topic', '')[:50]}")
        with output_lock:
            output_data.append(topic_copy)
        return True
    
    # Track used question types and question content to avoid duplicates
    used_question_types = set()
    used_questions = []
    
    # Process each image
    for img_index, img_path in enumerate(img_paths):
        if not img_path:
            continue
            
        logger.info(f"Processing image {img_index+1}/{len(img_paths)} for topic '{topic.get('topic', '')[:30]}'")
        
        # Create prompt, passing in used question types and question content
        prompt_data = create_prompt_for_topic(topic, img_index, used_question_types, used_questions)
        if not prompt_data:
            continue
            
        # Generate question
        question_data = generate_question_for_image(client, prompt_data)
        
        if question_data:
            # Detect if question is a duplicate
            is_duplicate = False
            new_question = question_data.get('question', '').lower()
            new_type = question_data.get('question_type', '').lower()
            
            # Check question similarity
            for existing_question in used_questions:
                # Simple similarity check: if key parts of the question are similar, consider it a duplicate
                existing_processed = existing_question.lower().replace("based on the provided image, ", "")
                new_processed = new_question.lower().replace("based on the provided image, ", "")
                
                # Extract the main part of the question (usually content after wh-word)
                import re
                existing_focus = re.sub(r'^(what|who|where|when|which|how|why)\s+', '', existing_processed)
                new_focus = re.sub(r'^(what|who|where|when|which|how|why)\s+', '', new_processed)
                
                if (new_focus in existing_focus or existing_focus in new_focus) and len(new_focus) > 10:
                    is_duplicate = True
                    logger.warning(f"Duplicate question detected: '{new_question}' with '{existing_question}'")
                    break
            
            # If question is not a duplicate, add to results
            if not is_duplicate:
                # Update used question types and question content
                used_question_types.add(new_type)
                used_questions.append(new_question)
                
                # Add to question list
                topic_copy["level1_qas"].append(question_data)
                logger.info(f"Successfully generated question for topic '{topic.get('topic', '')[:30]}': {question_data.get('question', '')[:50]}")
            else:
                logger.info(f"Skipping duplicate question: {new_question[:50]}")
    
    # Add to output
    with output_lock:
        output_data.append(topic_copy)
        
    return True

def process_topic_thread(topic, output_data):
    """Thread function: processes a single topic"""
    try:
        success = process_topic(topic, output_data)
        
        # Save results after processing each topic
        save_results(output_data)
            
        return success
    except Exception as e:
        logger.error(f"Error in topic processing thread: {str(e)}")
        return False

def save_results(output_data):
    """Thread-safely saves results to the output file"""
    global last_save_time
    
    # Use lock to prevent multiple threads from writing to the file simultaneously
    with save_lock:
        # Reduce save frequency to avoid too frequent I/O
        current_time = time.time()
        if current_time - last_save_time >= 1:  # Save at least once per second
            try:
                # Create output directory (if it doesn't exist)
                os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
                
                # Write to file
                with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Real-time saving results to {CONFIG['output_file']}")
                last_save_time = current_time
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")

def load_topics(file_path):
    """Safely loads a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading topics: {str(e)}")
        return []

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Level 1 Multi-Hop Questions')
    parser.add_argument('--workers', type=int, default=CONFIG["max_workers"], 
                        help='Number of threads for parallel processing')
    parser.add_argument('--input', type=str, 
                        help='Input file path (overrides automatic selection of the latest file)')
    parser.add_argument('--output', type=str,
                        help='Output file path (overrides automatically generated path)')
    parser.add_argument('--continue', dest='continue_processing', action='store_true',
                        help='Continue processing an existing output file')
    parser.add_argument('--timestamp', '-t', type=str, 
                        help='Specify timestamp of the modified_topics file to process, e.g., 04181718')
    args = parser.parse_args()
    
    # Determine input and output files
    input_file, output_file, timestamp = determine_io_files(args)
    
    # Update configuration
    CONFIG["max_workers"] = args.workers
    CONFIG["input_file"] = input_file
    CONFIG["output_file"] = output_file
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    if timestamp:
        logger.info(f"Timestamp: {timestamp}")
    
    # Load input topics
    input_topics = load_topics(CONFIG["input_file"])
    if not input_topics:
        logger.error(f"Could not load input file or file is empty: {CONFIG['input_file']}")
        return
    
    logger.info(f"Loaded {len(input_topics)} topics from {CONFIG['input_file']}")
    
    # Initialize output data and set of processed IDs
    output_data = []
    processed_ids = set()
    
    # Check if output file exists and load processed IDs
    if os.path.exists(CONFIG["output_file"]):
        existing_output = load_topics(CONFIG["output_file"])
        if existing_output:
            # If --continue argument is used, use existing output as base
            if args.continue_processing:
                output_data = existing_output
                logger.info(f"Loaded {len(output_data)} processed topics from {CONFIG['output_file']}")
            
            # Regardless of --continue, extract processed IDs for deduplication
            processed_ids = {topic.get('id') for topic in existing_output if topic.get('id') is not None}
            logger.info(f"Found {len(processed_ids)} processed IDs, these will be skipped")
    
    # Filter out already processed topics
    topics_to_process = []
    for topic in input_topics:
        topic_id = topic.get('id')
        if topic_id is None:
            # Topics without an ID also need to be processed
            topics_to_process.append(topic)
        elif topic_id not in processed_ids:
            # Only process unprocessed IDs
            topics_to_process.append(topic)
        else:
            logger.info(f"Skipping already processed topic ID: {topic_id}")
    
    logger.info(f"Filtered, {len(topics_to_process)} new topics to process, using {CONFIG['max_workers']} threads")
    
    # If no new topics to process, return directly
    if not topics_to_process:
        logger.info("No new topics to process, program finished")
        return
    
    # Use thread pool to process topics in parallel
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        # Submit all tasks
        future_to_topic = {
            executor.submit(process_topic_thread, topic, output_data): topic 
            for topic in topics_to_process
        }
        
        # Process completed tasks
        with tqdm(total=len(topics_to_process), desc="Processing topics") as pbar:
            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    success = future.result()
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Worker thread failed: {str(e)}")
    
    # Ensure final save
    try:
        os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
        with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"All processing completed and saved to {CONFIG['output_file']}")
    except Exception as e:
        logger.error(f"Error saving final results: {str(e)}")

if __name__ == "__main__":
    main()
