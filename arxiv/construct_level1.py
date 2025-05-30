import os
import json
import glob
import argparse
import random
from typing import List, Dict, Any, Optional
import multiprocessing
from tqdm import tqdm
from openai import OpenAI

# Create Manager for shared objects
manager = multiprocessing.Manager()

# Create shared dict for token usage
token_usage = manager.dict({
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
})

# Add lock for token usage update
token_lock = manager.Lock()

# Add lock for file writing
file_lock = manager.Lock()

def update_token_usage(usage_data):
    """Update token usage statistics"""
    with token_lock:
        token_usage["prompt_tokens"] += usage_data.prompt_tokens
        token_usage["completion_tokens"] += usage_data.completion_tokens
        token_usage["total_tokens"] += usage_data.total_tokens

# Question templates
questions = [
    "What is the title of the paper from this image source?",
    "Can you identify the research paper from the image provided?",
    "What is the full title of this paper depicted in the image?",
    "Which paper does this figure come from?",
    "What is the name of the paper associated with this image?",
    "From which research paper was this image taken?",
    "Could you tell me the title of the paper shown here?",
    "What paper does the figure in this image belong to?",
    "Please provide the title of the paper related to this image.",
    "What is the official title of the paper shown in the figure?",
    "What is the paper's title based on the image provided?",
    "Identify the paper that contains this figure.",
    "What paper does this image originate from?",
    "Based on the image, what is the title of the paper?",
    "What is the title of the paper containing this image?",
    "Who is the first author of the study shown in the image?",
    "Who conducted the research presented in this image?",
    "Who is the lead researcher of the paper shown in the figure?",
    "Who is the primary author of the paper shown here?",
    "Who is the first author of the study depicted in this image?"
]

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Load author pool for distractor generation
try:
    with open('authors.json', 'r', encoding='utf-8') as f:
        author_pool = json.load(f)
except Exception as e:
    print(f"Warning: Error loading author pool: {e}. Using empty author pool.")
    author_pool = []

# Prompt template for title distractors
TITLE_PROMPT = """You are an AI assistant specialized in generating academic paper title distractors.
Given the real title of a research paper, create four alternative titles that sound plausible but are clearly different from the original.
These should be believable as academic paper titles in a similar field, but not actual existing papers.

Original Title: {title}

Your response should be in JSON format:
{{
  "distractors": [
    "First title",
    "Second title",
    "Third title",
    "Fourth title"
  ]
}}
"""

# Prompt template for detailed explanation
DETAILED_PROMPT = """You are an AI assistant specialized in academic papers. 
Given the information about a research paper, create a detailed explanation for a question-answer pair.

Paper Title: {title}
Paper Abstract: {abstract}
Question: {question}
Correct Answer: {answer}

Your response should:
1. First clearly state the correct answer
2. Then provide a brief summary of the paper based on its abstract
3. Maximum length: 100 words
4. Format: "The correct answer is [correct answer]. [Paper summary]"

Your response:
"""

def call_gpt_for_paper_summary(title: str, abstract: str) -> str:
    """Call GPT to generate a summary for the paper, only once per paper"""
    try:
        # Truncate abstract if too long
        if abstract and len(abstract) > 1500:
            abstract = abstract[:1500] + "..."
        elif not abstract:
            abstract = "No abstract available for this paper."
            
        prompt = f"""You are an AI assistant specialized in academic papers. 
Given the information about a research paper, create a concise summary.

Paper Title: {title}
Paper Abstract: {abstract}

Your response should:
1. Provide a brief summary of the paper based on its abstract
2. Maximum length: 80 words
3. Format: clear, concise summary in 1-2 sentences

Your response:
"""
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        
        # Record token usage
        if response.usage:
            update_token_usage(response.usage)
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating paper summary: {e}")
        return "This paper discusses relevant research in its field."

def call_gpt_for_title_distractors(title: str) -> List[str]:
    """Call GPT to generate distractor titles for the paper"""
    global total_tokens_used
    try:
        prompt = TITLE_PROMPT.format(title=title)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # Record token usage
        if response.usage:
            update_token_usage(response.usage)
        
        content = response.choices[0].message.content
        parsed_content = json.loads(content)
        return parsed_content.get("distractors", [])
    except Exception as e:
        print(f"Error generating title distractors: {e}")
        return [f"Alternative Title {i} for Research Paper" for i in range(1, 5)]
    

def call_gpt_for_detailed_explanation(title: str, abstract: str, question: str, answer: str) -> str:
    """Call GPT to generate a detailed explanation"""
    global total_tokens_used
    try:
        # Truncate abstract if too long
        if abstract and len(abstract) > 1500:
            abstract = abstract[:1500] + "..."
        elif not abstract:
            abstract = "No abstract available for this paper."
            
        prompt = DETAILED_PROMPT.format(
            title=title,
            abstract=abstract,
            question=question,
            answer=answer
        )
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        # Record token usage
        if response.usage:
            update_token_usage(response.usage)
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating detailed explanation: {e}")
        return f"The correct answer is {answer}. This paper discusses relevant research in its field."


def create_qa_from_template(title: str, abstract: str, authors: List[str]) -> Optional[Dict[str, Any]]:
    """Create a QA pair based on the template"""
    # Randomly select a question template
    question_idx = random.randint(0, 19)
    question_type = 'title' if question_idx <= 14 else 'author'
    question = questions[question_idx]
    
    # Handle title-type questions
    if question_type == 'title':
        correct_answer = title
        # Get distractors
        distractors = call_gpt_for_title_distractors(title)
        
        # Use all four distractors
        options = distractors[:4]
        
        # Randomly insert correct answer
        correct_pos = random.randint(0, 4)
        options.insert(correct_pos, correct_answer)
        
    # Handle author-type questions
    else:  
        if not authors or len(authors) == 0:
            return None  # Return None if no author info
            
        correct_answer = authors[0]  # First author
        options = []
        
        # Add non-first authors if any
        other_authors = []
        if len(authors) > 1:
            other_authors = authors[1:]
        
        # Number of authors needed from pool
        needed_authors = 4 - len(other_authors)
        
        # Add authors from pool
        if needed_authors > 0 and author_pool:
            # Ensure not to select the correct answer from pool
            filtered_pool = [a for a in author_pool if a != correct_answer]
            random_authors = random.sample(filtered_pool, min(needed_authors, len(filtered_pool)))
            options.extend(random_authors)
        
        # Add other authors as distractors
        options.extend(other_authors[:4-len(options)])
        
        # Ensure only 4 distractors
        options = options[:4]
        
        # Randomly insert correct answer
        correct_pos = random.randint(0, 4)
        options.insert(correct_pos, correct_answer)
        
    # Find correct answer position
    correct_pos = options.index(correct_answer)
    formatted_options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
    correct_letter = chr(65 + correct_pos)
    
    # Generate detailed explanation using GPT
    detailed = call_gpt_for_detailed_explanation(
        title=title,
        abstract=abstract,
        question=question,
        answer=f"{correct_letter}. {correct_answer}"
    )
    
    # Build QA pair
    return {
        "question": question,
        "options": formatted_options,
        "Ground_Truth": correct_letter,
        "Ground_Truth_List": [correct_answer],
        "detailed": detailed
    }


def construct_qa(title: str, abstract: Optional[str], authors: Optional[List[str]],
                 figure_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct a QA pair"""
    # Check required info
    if not title:
        return None
        
    if not authors or len(authors) == 0:
        # If no author info, only generate title questions
        # To keep randomness, return None in 20% of cases (since ~20% are author questions)
        if random.random() < 0.2:
            return None
    
    return create_qa_from_template(title, abstract or "", authors)


def process_single_directory(args):
    """Process a single directory, generate QA pairs, and save results immediately"""
    json_file_path, output_file = args
    
    try:
        # Read associations.json
        associations_file_path = os.path.join(json_file_path, "associations.json")
        if not os.path.exists(associations_file_path):
            print(f"Error: associations.json file not found: {associations_file_path}")
            return None

        # Find and read selected_images.json
        selected_images_files = glob.glob(os.path.join(json_file_path, "*selected_images.json"))
        if not selected_images_files:
            print(f"Error: *selected_images.json file not found: {json_file_path}")
            return None
        selected_images_file_path = selected_images_files[0]

        # Read and merge data from both files
        with open(associations_file_path, 'r', encoding='utf-8') as f:
            associations_data = json.load(f)

        with open(selected_images_file_path, 'r', encoding='utf-8') as f:
            selected_images_data = json.load(f)

        article_data = {**associations_data}
        article_data["selected_figures"] = selected_images_data.get("selected_figures", [])
        
        # Check if selected_figures exists and is not empty
        if not article_data.get("selected_figures") or len(article_data.get("selected_figures")) == 0:
            print(f"Warning: No selected_figures or length is 0 in file {json_file_path}. Skipping this article.")
            return None
            
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON file: {json_file_path}")
        return None
    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
        return None
    except Exception as e:
        print(f"Error processing file {json_file_path}: {e}")
        return None

    # Extract article-level info
    processed_article = {
        "title": article_data.get("title"),
        "abstract": article_data.get("full_text") or article_data.get("abstract") or article_data.get("text_content"),
        "img_urls": [],
        "img_paths": [],
        "captions": [],
        "source": article_data.get("source") or article_data.get("publisher"),
        "url": article_data.get("url") or article_data.get("article_url"),
        "paper_id": article_data.get("paper_id"),
        "level1_qas": []
    }
    
    # Ensure article title exists
    if not processed_article["title"]:
        print(f"Warning: File {json_file_path} missing title info. Skipping this article.")
        return None

    # Process selected images
    selected_figures = article_data.get("selected_figures", [])
    
    # Skip if not exists or empty
    if not selected_figures or len(selected_figures) == 0:
        print(f"Warning: No selected images in file {json_file_path}. Skipping this article.")
        return None

    article_title = processed_article["title"]
    article_abstract = article_data.get("abstract")
    article_authors = article_data.get("authors")

    # If authors is a string, try to parse as list
    if isinstance(article_authors, str):
        try:
            article_authors = json.loads(article_authors)
        except:
            article_authors = [author.strip() for author in article_authors.split(',')]

    # Generate summary only once per paper
    paper_summary = call_gpt_for_paper_summary(
        title=article_title,
        abstract=processed_article["abstract"] or ""
    )

    # Collect image info and QA pairs
    all_img_urls = []
    all_img_paths = []
    all_captions = []
    all_context = []
    qa_pairs = []
    
    for figure in selected_figures:
        if not isinstance(figure, dict):
            continue

        # Generate QA pair for each image
        qa_pair = construct_qa(
            title=article_title,
            abstract=article_abstract,
            authors=article_authors,
            figure_info=figure
        )
        if qa_pair:
            qa_pairs.append(qa_pair)

        # Collect image info, keep correspondence
        img_url = figure.get("image_url")
        img_path = figure.get("image_path")
        caption = figure.get("caption")
        context = figure.get("context")
        
        # Only add if necessary info exists
        if img_url or img_path:
            all_img_urls.append(img_url)
            all_img_paths.append(img_path)
            all_captions.append(caption)
            all_context.append(context)

    # Add detailed explanation for each QA pair using generated summary
    for qa in qa_pairs:
        question = qa["question"]
        ground_truth = qa["Ground_Truth"]
        
        # Find the full text of the correct option
        correct_option = ""
        for option in qa["options"]:
            if option.startswith(f"{ground_truth}. "):
                correct_option = option[3:]  # Remove prefix like "A. "
                break
                
        # Build detailed explanation using template
        qa["detailed"] = f"The answer is {ground_truth}. {correct_option}. {paper_summary}"
            
        processed_article["level1_qas"].append(qa)
            
    # Save collected image info
    processed_article["img_urls"] = all_img_urls
    processed_article["img_paths"] = all_img_paths
    processed_article["captions"] = all_captions
    processed_article["context"] = all_context
    
    # Only return result if QA pairs were generated
    if processed_article["level1_qas"]:
        # Use lock to ensure thread-safe file writing
        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(processed_article, ensure_ascii=False) + '\n')
        return 1  # Return success flag
    else:
        print(f"Note: No QA pairs generated for file {json_file_path}.")
        return None


def process_directory(base_path: str, output_file: str, workers: int = 1, limit: int = None) -> int:
    """Traverse directory with multiprocessing and generate QA pairs, save as JSONL"""
    # Clear or create output file
    with open(output_file, 'w', encoding='utf-8') as f:
        pass  # Create empty file or clear existing
    
    # Collect all eligible file paths
    all_json_paths = []
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('selected_images.json'):
                    all_json_paths.append((subdir_path, output_file))  # Pass output file path as arg
                    break  # Only need one file per dir
                
    # Sort file paths
    all_json_paths.sort(key=lambda x: x[0])  # Sort by first element (file path)
    print(f"Found {len(all_json_paths)} eligible directories")
    
    # Limit number of files to process
    if limit is not None and limit > 0 and limit < len(all_json_paths):
        all_json_paths = all_json_paths[:limit]
        print(f"Limit parameter set, only processing first {limit} files")
    
    print(f"Found {len(all_json_paths)} files to process, using {workers} processes")
    
    # Count successful processing
    success_count = 0
    
    # Use multiprocessing
    if workers > 1:
        # Ensure correct multiprocessing start method on Windows
        if os.name == 'nt' and multiprocessing.get_start_method(False) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        
        # Create process pool to handle files
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_directory, all_json_paths),
                total=len(all_json_paths),
                desc="Processing files"
            ))
            # Count successfully processed files
            success_count = sum(1 for result in results if result is not None)
    else:
        # Single process
        for args in tqdm(all_json_paths, desc="Processing files", unit="file"):
            json_file_path = args[0]
            print(f"\nProcessing file: {json_file_path}")
            result = process_single_directory(args)
            if result is not None:
                success_count += 1
    
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files in directory and generate QA pairs")
    parser.add_argument('--input-path', '-i', type=str, help='Directory path to process', default='cate/processed/json/2404')
    parser.add_argument('--output-file', '-o', type=str, help='Output file path', default='output3.jsonl')  # Changed to .jsonl extension
    parser.add_argument('--workers', '-w', type=int, help='Number of processes to use', default=1)
    parser.add_argument('--limit', '-l', type=int, help='Limit number of files to process', default=None)
    args = parser.parse_args()

    target_directory = args.input_path
    
    if not os.path.isdir(target_directory):
        print(f"Error: Provided path '{target_directory}' is not a valid directory.")
    else:
        print(f"Start processing directory: {target_directory}")
        success_count = process_directory(target_directory, args.output_file, args.workers, args.limit)

        # Print statistics
        print(f"\nProcessing complete. Data saved to: {args.output_file}")
        print(f"Successfully processed QA data for {success_count} articles.")
        
        # Count total QA pairs generated
        total_qa_pairs = 0
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    qa_count = len(article.get('level1_qas', []))
                    total_qa_pairs += qa_count
                except json.JSONDecodeError:
                    continue
        
        print(f"Total {total_qa_pairs} QA pairs generated.")
        
        # Print API token usage statistics
        print("\nAPI Token Usage Statistics:")
        usage_dict = dict(token_usage)
        print(f"Prompt tokens: {usage_dict['prompt_tokens']}")
        print(f"Completion tokens: {usage_dict['completion_tokens']}")
        print(f"Total tokens: {usage_dict['total_tokens']}")