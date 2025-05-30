import os
import json
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI
import multiprocessing
from tqdm import tqdm
import jsonlines

# Create Manager for shared objects
manager = multiprocessing.Manager()

# Create shared dict for token usage
token_usage = manager.dict({
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
})

# Add lock to protect token usage updates in multiprocessing
token_lock = manager.Lock()

# Set OpenAI API key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def update_token_usage(usage_data):
    """Update token usage statistics"""
    with token_lock:
        token_usage["prompt_tokens"] += usage_data.prompt_tokens
        token_usage["completion_tokens"] += usage_data.completion_tokens
        token_usage["total_tokens"] += usage_data.total_tokens

def generate_level2_qa_for_image(paper_item, image_index):
    """Generate Level 2 QA for a specific image"""
    try:
        # Get basic paper info
        title = paper_item.get("title", "")
        abstract = paper_item.get("abstract", "")
        paper_id = paper_item.get("paper_id", "unknown")
        
        # Get specific image's caption and context
        image_caption = paper_item["captions"][image_index] if image_index < len(paper_item["captions"]) else ""
        image_context = "\n".join([" ".join(paragraph) for paragraph in paper_item["context"]]) if "context" in paper_item else ""
        
        # Build content for QA
        content_for_qa = f"""Paper Title: {title}

Abstract: {abstract}

Image Caption: {image_caption}

Context:
{image_context}
"""
        
        # Build prompt
        prompt = f"""You are an AI tasked with generating multiple-choice questions. Your goal is to create questions that appear to be based solely on an image from a scientific paper.

I will provide you with the full textual content related to this image, including the paper's title, abstract, and any relevant contextual details:
{content_for_qa}
You will use this information to craft your questions and answers. However, your generated questions and explanations must be framed as if the end-user was only initially provided with the image itself and no other information.

Please generate 1-2 multiple-choice questions. For each question, adhere to these specific instructions:

1. Challenge:
   - Craft a question that pushes the limits of search and reasoning for both humans and AI.
   - It should not be answerable by a simple keyword lookup; it must require careful reading and inference.

2. Focus on Text Details Only:
   - Target one specific, simple detail from the provided text.
   - Do NOT mention or describe any visual content from the image.
   - Refer to the paper’s methods abstractly (e.g., “the method described in the paper”).

3. Answer Types (choose exactly one):
   a. A specific data value (e.g., a number or percentage) from the paper.
   b. A precise year mentioned in the paper.
   c. An objective research result statement that appears verbatim in the text.

4. Uniqueness & Definiteness:
   - The answer must be unambiguous and unique in the text.

5. Non‐Visual:
   - The correct answer cannot be derived from any visual element; it must depend on the text.

6. Self‐Contained Answerability:
   - The question must be answerable using only the given abstract or context, without external knowledge.


For each question, provide the following:

  * A clear, concise question text.
  * Five options (labeled A through E).
  * The correct answer's letter (this letter should be randomly chosen from A-E for each question).
  * A list containing the correct answer phrased in one or more ways (e.g., `["The primary finding was X.", "X was identified as the main result."]` ).
  * Detailed reasoning process to get the correct answer. MUST NOT mention about other options, they are not needed. 

Format your entire response as a single JSON object. Do not include any markdown formatting or any text outside of this JSON object.

{{
  "level2_qas": [
    {{
      "question": "[Your question text here]",
      "options": [
        "A. [Option A text]",
        "B. [Option B text]",
        "C. [Option C text]",
        "D. [Option D text]",
        "E. [Option E text]"
      ],
      "Ground_Truth": "[Correct letter]",
      "Ground_Truth_List": ["[The correct answer phrased as in the text]", "[An alternative phrasing of the correct answer]"],
      "reasoning": "[Detailed reasoning process: Start with 'The correct answer is [correct answer string]. The source paper is [the paper]'. Explain step-by-step how the correct answer is derived from the specific details within the provided abstract or contextual information of that identified paper. This reasoning should not suggest the answer comes directly from the abstract or context you were given but rather from the text *of the paper found via the image*]"
    }},
    {{ ... more questions in the same format ... }}
  ]
}}
"""

        # Send request to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates multiple-choice questions about scientific articles and their associated images."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Record token usage
        if response.usage:
            update_token_usage(response.usage)
        
        # Parse response
        qa_content = response.choices[0].message.content
        
        # Extract JSON object
        try:
            # Clean possible Markdown formatting
            qa_content_clean = re.sub(r'```json|```', '', qa_content).strip()
            # Parse JSON
            qa_json = json.loads(qa_content_clean)
            
            if "level2_qas" in qa_json and isinstance(qa_json["level2_qas"], list):
                return qa_json["level2_qas"]
            else:
                print(f"Response does not contain proper level2_qas structure for image {image_index} in paper {paper_id}")
                return []
                
        except Exception as e:
            print(f"JSON parsing error for image {image_index} in paper {paper_id}: {str(e)}")
            print("Response content preview:", qa_content[:200] + "...")
            return []
    
    except Exception as e:
        print(f"Error processing image {image_index} in paper {paper_id}: {str(e)}")
        return []

def process_paper_item(paper_item):
    """Process a single paper item and generate questions for each image"""
    try:
        paper_id = paper_item.get("paper_id", "unknown")
        
        # Generate Level 2 QA for each image
        num_images = len(paper_item.get("img_urls", []))
        for i in range(num_images):
            # Generate questions
            qa_list = generate_level2_qa_for_image(paper_item, i)
            
            # Add questions to paper item
            key = f"level2_qas{i+1}"
            paper_item[key] = qa_list
            
            # Prevent API requests from being too frequent
            time.sleep(1)
        
        return paper_item
    except Exception as e:
        print(f"Error processing paper {paper_item.get('paper_id', 'unknown')}: {str(e)}")
        return None

def worker_function(paper_item, output_file, lock):
    """Worker process function, processes a single paper and writes the result"""
    try:
        result = process_paper_item(paper_item)
        if result:
            # Use lock to ensure no conflict when writing in multiprocessing
            with lock:
                with jsonlines.open(output_file, mode='a') as writer:
                    writer.write(result)
            return True
        return False
    except Exception as e:
        print(f"Worker error: {str(e)}")
        return False

def main(input_file, output_file, num_processes=4, limit=None):
    """Main function: process input file and generate output"""
    start_time = time.time()
    try:
        print(f"Start processing. Input file: {input_file}, Output file: {output_file}")
        print(f"Using {num_processes} processes in parallel" + (f", limiting to {limit} entries" if limit else ""))
        
        # Create output directory if it does not exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If output file exists, delete it first
        if output_path.exists():
            output_path.unlink()
            print(f"Deleted existing output file: {output_file}")
        
        # Read input JSONL file
        data = []
        with jsonlines.open(input_file, mode='r') as reader:
            for item in reader:
                data.append(item)
                if limit is not None and len(data) >= limit:
                    break
        
        print(f"Loaded {len(data)} papers, containing {sum(len(item.get('img_urls', [])) for item in data)} images in total")
        
        # Create process pool and shared lock
        pool = multiprocessing.Pool(processes=num_processes)
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        
        # Use process pool for parallel processing
        results = []
        for paper_item in data:
            result = pool.apply_async(worker_function, args=(paper_item, output_file, lock))
            results.append(result)
        
        # Show progress with tqdm
        total_items = len(results)
        success_count = 0
        failed_count = 0
        
        for result in tqdm(results, total=total_items, desc="Processing papers"):
            if result.get():
                success_count += 1
            else:
                failed_count += 1
            result.wait()
        
        # Close process pool
        pool.close()
        pool.join()
        
        # Count number of generated questions
        total_qa_count = 0
        papers_with_qa = 0
        
        with jsonlines.open(output_file, mode='r') as reader:
            for item in reader:
                qa_keys = [k for k in item.keys() if k.startswith('level2_qas')]
                paper_qa_count = sum(len(item.get(key, [])) for key in qa_keys)
                if paper_qa_count > 0:
                    papers_with_qa += 1
                    total_qa_count += paper_qa_count
        
        processing_time = time.time() - start_time
        hours, remainder = divmod(processing_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nProcessing complete! Results saved to {output_file}")
        print(f"Processing statistics:")
        print(f"- Total papers processed: {total_items}")
        print(f"- Successfully processed: {success_count}")
        print(f"- Failed: {failed_count}")
        print(f"- Papers with questions: {papers_with_qa}")
        print(f"- Total number of questions: {total_qa_count}")
        print(f"- Average questions per paper: {total_qa_count/papers_with_qa:.2f}" if papers_with_qa > 0 else "- Average questions per paper: 0")
        print(f"- Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"- Average processing time per paper: {processing_time/total_items:.2f} seconds" if total_items > 0 else "- Average processing time per paper: 0 seconds")
        
        # Print token usage statistics
        print("\nAPI Token Usage Statistics:")
        usage_dict = dict(token_usage)
        print(f"Prompt tokens: {usage_dict['prompt_tokens']:,}")
        print(f"Completion tokens: {usage_dict['completion_tokens']:,}")
        print(f"Total tokens: {usage_dict['total_tokens']:,}")
    
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Level 2 QA dataset for images in scientific papers")
    parser.add_argument("--input-file", '-i', default="output3.jsonl", help="Input jsonl path")
    parser.add_argument("--output-file", '-o', default="output_with_level2.jsonl", help="Output JSONL file path")
    parser.add_argument("--processes", '-p', type=int, default=4, help="Number of processes to use")
    parser.add_argument("--limit", '-l', type=int, default=None, help="Limit the number of papers to process")
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.processes, args.limit)