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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_makers.question_processor import setup_client, get_thread_client, call_gpt_with_retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "l1_question_generator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = ""
DATA_DIR = os.path.join(BASE_DIR, "data/raw_data")

output_lock = threading.Lock()
save_lock = threading.Lock()
last_save_time = time.time()

def get_latest_modified_topics_file():
    files = glob.glob(os.path.join(DATA_DIR, "modified_topics_*.json"))
    if not files:
        return None  
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def get_timestamp_from_filename(filename):
    match = re.search(r'modified_topics_(\d+)\.json', filename)
    if match:
        return match.group(1)
    return None


def determine_io_files(args):
    
    input_file = None
    timestamp = None
    
    if args.timestamp:
       
        timestamp = args.timestamp
        specified_file = os.path.join(DATA_DIR, f"modified_topics_{timestamp}.json")
        if os.path.exists(specified_file):
            input_file = specified_file
        else:
            logger.error(f"Not found: {specified_file}")
            sys.exit(1)
    elif args.input:
        input_file = args.input
        timestamp = get_timestamp_from_filename(input_file)
    else:
        input_file = get_latest_modified_topics_file()
        if input_file:
            timestamp = get_timestamp_from_filename(input_file)
        else:
            logger.error("Error: Not found modified_topics files")
            sys.exit(1)
    
    if args.output:
        output_file = args.output
    elif timestamp:
        output_file = os.path.join(DATA_DIR, f"l1_topics_{timestamp}.json")
    else:
        output_file = os.path.join(DATA_DIR, "l1_topics.json")
    
    return input_file, output_file, timestamp

CONFIG = {
    "api_key": "",
    "model": "gpt-4.1",
    "max_workers": 8, 
    "temperature": 0.7,  
    "max_tokens": 2000
}

def encode_image_to_base64(image_path):
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error {image_path}: {str(e)}")
        return None


# LLM-based topic filtering function
def llm_filter_topics_with_prompt(topics, client):
    meaningful = []
    filtered = []
    for topic in topics:
        title = topic.get("topic", "")
        text = topic.get("text", "")
        img_paths = topic.get("img_paths", [])
        prompt = f"""You are a content filtering assistant.

Please determine if the following topic is meaningful for generating a visual question that combines image and text understanding.

Topic title: "{title}"
Text content: "{text}"

Rules:
- Reject if the content is a presenter’s greeting, farewell, or generic transition (e.g., "Welcome to the show", "Thank you for watching").
- Reject if the text has no clear event, person, place, or object described.
- Reject if it is conversational filler or lacks informative content.
- Accept only if the content supports visual QA based on external factual entities.

Reply only with "meaningful" or "meaningless"."""
        try:
            # If there is an image, add the first image as image_url content
            if img_paths and img_paths[0]:
                base64_image = encode_image_to_base64(img_paths[0])
                if base64_image:
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }]
                else:
                    messages = [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
            response = call_gpt_with_retry(client, messages, max_retries=3, retry_delay=2)
            answer = response.choices[0].message.content.strip().lower()
            if "meaningless" in answer:
                filtered.append(topic)
            else:
                meaningful.append(topic)
        except Exception as e:
            logger.warning(f"Error in LLM filtering: {title[:30]} - {str(e)}")
            meaningful.append(topic)
    return meaningful, filtered
"""
Level 1 Question Generator

This script generates level 1 questions for news topics with images,
focusing on questions that require social knowledge to answer.
"""

def create_prompt_for_topic(topic, img_index, used_question_types=None, used_questions=None):
    
    system_prompt = """You are an AI assistant specialized in generating high-quality Level 1 multi-hop questions.

Level 1 multi-hop means the question requires directly linking information seen in the image with factual details found in the accompanying text, without complex reasoning.
These questions should test social knowledge, which refers to external factual knowledge about the real world (e.g., names of people, places, organizations, specific events, objects) beyond simple visual description.
Your task is to create image-and-text-based questions focusing on factual information, not inference or reasoning.

Your generated question MUST follow these strict requirements:

Question Format: Always start with "Based on the provided image, " followed by a clear, concise question.
Answer Source & Grounding: The answer MUST be explicitly findable and directly verifiable within the provided text (or topic context). Base the answer ONLY on the given information to prevent hallucination or reliance on external knowledge not present in the source material. Do not infer information beyond what is stated. (强调答案来源，防止幻觉)
Answer Format: The answer must be a short phrase or a few words (NOT a sentence or paragraph).
Answer Specificity: The answer MUST be a highly specific, uniquely identifiable name or term (e.g., "Nike Air Force 1", "Galeries Lafayette in Paris", "2023 Paris Fashion Week", "Emmanuel Macron"), NOT a generic category or description (e.g., "designer sneakers", "shopping mall", "protest", "police officer"). This applies to all answer types.
Question Categories: The question MUST belong to one of these categories ONLY:
location (where something is happening - answer must be specific place)
person (who is in the image - answer must be specific name, avoid trivially famous people)
organization (which company, team, group, etc. - answer must be specific name)
time (when a specific, named event shown occurred, as stated in text)
object (what specific, uniquely named item is shown)
event (ONLY allowed to ask "what event is taking place?" - answer must be specific named event)
Question Simplicity and Neutrality: The question must be concise and generic. It MUST NOT reveal specific names, dates, or unique details from the article that could hint at the answer or unnecessary context. Frame the question based on what is visually present without giving away context only found in the text.
Image-Text Integration: The question must relate to something clearly visible or identifiable as a category in the image, while requiring the text for the specific answer. The question's subject should make sense focusing on the image alone.
Knowledge Requirement: The question should test knowledge that cannot be directly answered by computer vision alone (requires factual lookup in text).
CRUCIAL QUALITY CRITERIA - AVOID THESE COMMON ISSUES:

FAMOUS FIGURES: DO NOT create questions asking about extremely well-known figures recognizable by almost anyone (e.g., asking "who is this person?" when Donald Trump or Elon Musk is clearly depicted).
TEMPORAL CONTEXT REQUIRED: NEVER create questions about images lacking clear temporal context (e.g., generic close-ups of food, products, or objects with no time-specific indicators or event context). Especially avoid simple identification questions for these ("what food is this?").
NO COUNTING QUESTIONS: Never ask to count objects or people.
AVOID BOOK COVER QUESTIONS: Don't ask about book covers if the answer is simply "book cover", "memoir cover", or "book jacket".
NO VISIBLE TEXT ANSWERS: Don't create questions where the answer is clearly legible text within the image itself. For example, if a person's name (e.g., "John Doe") is visible on a name tag or caption in the image, do not ask who that person is. If an organization's name or common abbreviation (e.g., "NASA" or "FBI") is clearly readable on a sign, logo, or clothing in the image, do not ask what the organization is. (e.g., asking about Google when 'Google' is written on the building).
NO CHART DATA QUESTIONS: Don't ask about data explicitly shown in charts or graphs within the image.
COMPLETE CONTENT REQUIRED: Ensure both the image and text context are available to formulate the question and find the answer.
AVOID DUPLICATES: Avoid generating questions very similar to others already created for different images within the same topic/article.

EXAMPLES OF BAD QUESTIONS (Illustrating Violations):

❌ "Based on the provided image, who is speaking at the podium?" → Answer: "President Donald Trump" (Violates #9 Famous Figure)
❌ "Based on the provided image, what type of footwear is shown?" → Answer: "Designer sneakers" (Violates #4 Answer Specificity)
❌ "Based on the provided image, what dish is being prepared?" → Answer: "Pizza" (Violates #10 Temporal Context / Generic Object)
❌ "Based on the provided image, how many protesters are visible?" → Answer: "24" (Violates #11 No Counting)
❌ "Based on the provided image, what is shown on the book cover?" → Answer: "Book jacket" (Violates #12 Book Cover)
❌ "Based on the provided image, what company logo is displayed?" → Answer: "Google" (Violates #13 Visible Text - assuming 'Google' text is readable)
❌ "Based on the provided image, who is the person shown?" → Answer: "Jane Smith" (Violates #13 Visible Text - assuming 'Jane Smith' is written on her badge in the image)
❌ "Based on the provided image, what organization's initials are on the building?" → Answer: "MIT" (Violates #13 Visible Text - assuming 'MIT' is clearly readable on the building)
❌ "Based on the provided image, what type of factory is shown?" → Answer: "Clothing factory" (Violates #4 Answer Specificity - Location)
❌ "Based on the provided image, what event is taking place?" → Answer: "A protest" (Violates #4 Answer Specificity - Event)
❌ "Based on the provided image, what does the graph show?" → Answer: "Rising stock price" (Violates #14 Chart Data)
❌ "Based on the provided image, who is the person in uniform?" → Answer: "Police officer" (Violates #4 Answer Specificity - Person)
❌ "Based on the provided image, what is the name of the memorial site where the graves of Zambia's 1993 national football team are located?" (Violates #6 Question Simplicity/Neutrality - reveals too much context)
❌ "Based on the provided image, who is the CEO that announced the company's new AI strategy at the June conference?" (Violates #6 Question Simplicity/Neutrality - reveals too many details)

EXAMPLES OF GOOD QUESTIONS (Meeting Requirements):

✅ "Based on the provided image, what is the location shown?" (Simple, focused on image, answer requires specific place name from text)
✅ "Based on the provided image, who is the person at the podium?" (Asks about visible element, answer requires specific name from text, assuming name is not visible in image and person isn't trivially famous)
✅ "Based on the provided image, what organization does this logo represent?" (Focuses on visual element/logo, answer requires specific org name from text, assuming logo itself doesn't contain the readable name/abbreviation)
✅ "Based on the provided image, what event is taking place?" (Standard event question, answer requires specific event name from text)

FINAL CHECK - AVOID questions about:

Trivial visual attributes (color, easily counted items).
Ambiguous or subjective answers.
Things answerable without the provided text's factual/social knowledge.
Extremely obvious information (unless testing the text link).
Answers directly readable in the image (including names, acronyms, etc.).
Answers requiring guessing, inference beyond the text, or external knowledge. 
"""

    title = topic.get('topic', 'No title')
    text = topic.get('text', 'No text')
    img_paths = topic.get('img_paths', [])
    img_urls = topic.get('img_urls', [])
    captions = topic.get('captions', [])
    
    if img_index >= len(img_paths) or not img_paths[img_index]:
        return None
    
    img_path = img_paths[img_index]
    img_url = img_urls[img_index] if img_index < len(img_urls) else ""
    caption = captions[img_index] if img_index < len(captions) else "No caption"
    
    used_types_info = ""
    if used_question_types:
        used_types_str = ", ".join([f"'{qt}'" for qt in used_question_types])
        used_types_info = f"\nALREADY USED QUESTION TYPES: {used_types_str}"
    
    used_questions_info = ""
    if used_questions and len(used_questions) > 0:
        used_questions_str = "\n- " + "\n- ".join([f'"{q}"' for q in used_questions])
        used_questions_info = f"\nQUESTIONS ALREADY GENERATED FOR OTHER IMAGES IN THIS TOPIC: {used_questions_str}"
    
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
6. If you need to identify people, especially when there are multiple people in the image or text, please add clear references, such as "Who is the person on the left in the picture?"

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
    system_prompt = prompt_data["system"]
    user_prompt = prompt_data["user"]
    img_path = prompt_data["img_path"]
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
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
                
                # Check for error
                if "error" in result:
                    logger.warning(f"Cannot generate questions: {result['error']}")
                    return None
                    
                return result
            except json.JSONDecodeError:
                logger.error(f"JSON parsing error: {content[:200]}")
        else:
            logger.error(f"Unable to extract JSON from response: {content[:200]}")
            
        return None
    except Exception as e:
        logger.error(f"Error in generating: {str(e)}")
        return None

def process_topic(topic, output_data):
    """Process all images for a single topic"""
    # Check for placeholder topic
    if topic.get('id') is not None and topic.get('topic') is None:
        logger.info(f"Skipping placeholder topic ID: {topic.get('id')}")
        # Add to output for consistency but do not generate questions
        topic_copy = {k: topic.get(k) for k in topic.keys()}
        topic_copy["level1_qas"] = []
        
        with output_lock:
            output_data.append(topic_copy)
        return True
    
    # Skip if topic is meaningless
    if topic.get('is_meaningful') is False:
        logger.info(f"Skipping meaningless topic ID: {topic.get('id', 'No ID')}, Topic: {topic.get('topic', '')[:30]}")
        # Add to output for consistency but do not generate questions
        topic_copy = {k: topic.get(k) for k in topic.keys()}
        topic_copy["level1_qas"] = []
        
        with output_lock:
            output_data.append(topic_copy)
        return True
    
    # Create topic copy
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

    # Track used question types and content to avoid duplicates
    used_question_types = set()
    used_questions = []

    # Process each image
    for img_index, img_path in enumerate(img_paths):
        if not img_path:
            continue

        # Skip QA generation for questions deemed not meaningful
        if not topic.get("is_meaningful", True):
            continue  # Skip questions deemed not meaningful

        logger.info(f"Processing topic '{topic.get('topic', '')[:30]}' image {img_index+1}/{len(img_paths)}")

        # Create prompt, pass used types and questions
        prompt_data = create_prompt_for_topic(topic, img_index, used_question_types, used_questions)
        if not prompt_data:
            continue

        # Generate question
        question_data = generate_question_for_image(client, prompt_data)

        if question_data:
            # Detect duplicate questions
            is_duplicate = False
            new_question = question_data.get('question', '').lower()
            new_type = question_data.get('question_type', '').lower()

            # Simple similarity check: if question key part is similar, treat as duplicate
            for existing_question in used_questions:
                existing_processed = existing_question.lower().replace("based on the provided image, ", "")
                new_processed = new_question.lower().replace("based on the provided image, ", "")

                # Extract main part of the question (usually after wh-word)
                import re
                existing_focus = re.sub(r'^(what|who|where|when|which|how|why)\s+', '', existing_processed)
                new_focus = re.sub(r'^(what|who|where|when|which|how|why)\s+', '', new_processed)

                if (new_focus in existing_focus or existing_focus in new_focus) and len(new_focus) > 10:
                    is_duplicate = True
                    logger.warning(f"Detected duplicate question: '{new_question}' and '{existing_question}'")
                    break

            # If not duplicate, add to results
            if not is_duplicate:
                used_question_types.add(new_type)
                used_questions.append(new_question)

                topic_copy["level1_qas"].append(question_data)
                logger.info(f"Successfully generated question for topic '{topic.get('topic', '')[:30]}': {question_data.get('question', '')[:50]}")
            else:
                logger.info(f"Skipped duplicate question: {new_question[:50]}")

    # Add to output
    with output_lock:
        output_data.append(topic_copy)

    return True

def process_topic_thread(topic, output_data):
    """Thread function: process single topic"""
    try:
        success = process_topic(topic, output_data)
        
        # Save results after each topic
        save_results(output_data)
            
        return success
    except Exception as e:
        logger.error(f"Error in topic processing thread: {str(e)}")
        return False

def save_results(output_data):
    """Thread-safe save results to output file"""
    global last_save_time
    
    with save_lock:
        current_time = time.time()
        if current_time - last_save_time >= 1:  # Save at least once per second
            try:
                os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
                with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saving results to {CONFIG['output_file']}")
                last_save_time = current_time
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")

def load_topics(file_path):
    """Safely load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading topics: {str(e)}")
        return []

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Level 1 Multi-hop Questions')
    parser.add_argument('--workers', type=int, default=CONFIG["max_workers"], 
                        help='Number of parallel processing threads')
    parser.add_argument('--input', type=str, 
                        help='Input file path (overrides automatic latest file)')
    parser.add_argument('--output', type=str,
                        help='Output file path (overrides auto-generated path)')
    parser.add_argument('--continue', dest='continue_processing', action='store_true',
                        help='Continue processing from existing output file')
    parser.add_argument('--timestamp', '-t', type=str, 
                        help='Specify the timestamp of the modified_topics file to process, e.g., 04181718')
    args = parser.parse_args()
    
    # Determine input/output files
    input_file, output_file, timestamp = determine_io_files(args)
    
    # Update config
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
        logger.error(f"Failed to load input file or file is empty: {CONFIG['input_file']}")
        return
    logger.info(f"Loaded {len(input_topics)} topics from {CONFIG['input_file']}")

    # Filter meaningless topics, output two files (using LLM filtering)
    filtered_output_path = input_file.replace("modified_topics", "filteredout_topics")
    meaningful_output_path = input_file.replace("modified_topics", "meaningful_topics")
    client = setup_client()
    input_topics, filtered_topics = llm_filter_topics_with_prompt(input_topics, client)
    logger.info(f"Retained {len(input_topics)} items, filtered out {len(filtered_topics)} items")
    with open(filtered_output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_topics, f, ensure_ascii=False, indent=2)
    with open(meaningful_output_path, 'w', encoding='utf-8') as f:
        json.dump(input_topics, f, ensure_ascii=False, indent=2)
    
    # Initialize output data and processed ID set
    output_data = []
    processed_ids = set()
    
    # Check if output file exists and load processed IDs
    if os.path.exists(CONFIG["output_file"]):
        existing_output = load_topics(CONFIG["output_file"])
        if existing_output:
            # If --continue is used, use existing output as base
            if args.continue_processing:
                output_data = existing_output
                logger.info(f"Loaded {len(output_data)} processed topics from {CONFIG['output_file']}")
            
            # Extract processed IDs for deduplication
            processed_ids = {topic.get('id') for topic in existing_output if topic.get('id') is not None}
            logger.info(f"Found {len(processed_ids)} processed IDs, will skip these")
    
    # Filter out already processed topics
    topics_to_process = []
    for topic in input_topics:
        topic_id = topic.get('id')
        if topic_id is None:
            topics_to_process.append(topic)
        elif topic_id not in processed_ids:
            topics_to_process.append(topic)
        else:
            logger.info(f"Skipping already processed topic ID: {topic_id}")
    
    logger.info(f"After filtering, {len(topics_to_process)} new topics to process with {CONFIG['max_workers']} threads")
    
    if not topics_to_process:
        logger.info("No new topics to process, exiting")
        return
    
    # Use thread pool to process topics in parallel
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        future_to_topic = {
            executor.submit(process_topic_thread, topic, output_data): topic 
            for topic in topics_to_process
        }
        
        with tqdm(total=len(topics_to_process), desc="Processing topics") as pbar:
            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    success = future.result()
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Worker thread failed: {str(e)}")
    
    # Final save
    try:
        os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
        with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"All processing complete, saved to {CONFIG['output_file']}")
    except Exception as e:
        logger.error(f"Error saving final results: {str(e)}")

if __name__ == "__main__":
    main()
