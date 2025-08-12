import json
import os
import glob
import re
import requests
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

class L2QuestionFilter:
    def __init__(self, openrouter_api_key: str):
        self.api_key = openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "LIVEVQA L2 Filter",
            "Content-Type": "application/json"
        }
    
    def find_latest_l23_topics_file(self, data_dir: str) -> Optional[str]:
        """Find the latest l23_topics_{timestamp}.json file"""
        pattern = os.path.join(data_dir, "l23_topics_*.json")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Sort by timestamp to get the latest file
        def extract_timestamp(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'l23_topics_(\d+)\.json', filename)
            return int(match.group(1)) if match else 0
        
        latest_file = max(files, key=extract_timestamp)
        return latest_file
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def call_gpt4_api(self, question: str, options: List[str], text: str, image_path: str) -> Optional[str]:
        """Call OpenRouter GPT-4.1 API to get answer"""
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            
            # Build options string
            options_text = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Please answer the following multiple-choice question based on the provided image and text context.

Context Text:
{text}

Question: {question}

Options:
{options_text}

Please provide only the letter of your answer (A, B, C, D, or E). Do not provide any explanation."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            payload = {
                "model": "openai/gpt-4-turbo",
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0
            }
            
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            # Extract answer letter
            answer_match = re.search(r'[A-E]', answer.upper())
            if answer_match:
                return answer_match.group()
            return None
            
        except Exception as e:
            print(f"API call error: {e}")
            return None
    
    def check_answer_match(self, api_answer: str, ground_truth: str, ground_truth_list: List[str]) -> bool:
        """Check if API answer matches ground truth"""
        if not api_answer:
            return False
            
        # Check if option letter matches
        if api_answer.upper() == ground_truth.upper():
            return True
        
        return False
    
    def filter_level2_questions(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Filter level2 questions for a single entry"""
        if not entry.get('level2_qas_img1'):
            return entry
        
        filtered_questions = []
        text = entry.get('text', '')
        
        for qa in entry['level2_qas_img1']:
            question = qa.get('question', '')
            options = qa.get('options', [])
            ground_truth = qa.get('Ground_Truth', '')
            ground_truth_list = qa.get('Ground_Truth_List', [])
            img_path = qa.get('img_path', '')
            
            # Check required fields
            if not all([question, options, ground_truth, img_path]):
                print(f"Skipping incomplete question: {question[:50]}...")
                continue
            
            # Check if image file exists
            if not os.path.exists(img_path):
                print(f"Image file not found, skipping question: {img_path}")
                continue
            
            print(f"Validating question: {question[:50]}...")
            
            # Call API to get answer
            api_answer = self.call_gpt4_api(question, options, text, img_path)
            
            # Check if answer matches
            if self.check_answer_match(api_answer, ground_truth, ground_truth_list):
                filtered_questions.append(qa)
                print(f"✓ Question passed validation (API: {api_answer}, GT: {ground_truth})")
            else:
                print(f"✗ Question failed validation (API: {api_answer}, GT: {ground_truth})")
            
            # Add delay to avoid API limits
            time.sleep(1)
        
        # Update entry's level2 questions
        entry['level2_qas_img1'] = filtered_questions
        return entry
    
    def process_json_file(self, input_file: str, output_file: str):
        """Process entire JSON file"""
        print(f"Starting to process file: {input_file}")
        
        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        filtered_data = []
        total_entries = len(data)
        
        for i, entry in enumerate(data):
            print(f"\nProcessing entry {i+1}/{total_entries}")
            
            # Check if already discarded
            if entry.get('discarded', False):
                print("Entry already discarded, skipping")
                continue
            
            # Filter level2 questions
            filtered_entry = self.filter_level2_questions(entry)
            
            # Check if any level2 questions remain
            if not filtered_entry.get('level2_qas_img1'):
                print("All level2 questions failed, discarding entire entry")
                continue
            
            filtered_data.append(filtered_entry)
            print(f"Entry retained, remaining level2 questions: {len(filtered_entry['level2_qas_img1'])}")
        
        # Save filtered data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nProcessing completed!")
        print(f"Original entries: {total_entries}")
        print(f"Filtered entries: {len(filtered_data)}")
        print(f"Output file: {output_file}")

def main():
    # Configuration parameters - project root directory
    PROJECT_ROOT = "YOUR LOCAL PATH"  # Project root directory
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_data")  # Auto-build path to data/raw_data
    OPENROUTER_API_KEY = "YOUR API KEY HERE"
    
    # Check if data/raw_data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory does not exist - {DATA_DIR}")
        print("Please ensure data/raw_data folder exists under project root")
        return
    
    # Create filter instance
    filter_instance = L2QuestionFilter(OPENROUTER_API_KEY)
    
    # Find latest input file
    input_file = filter_instance.find_latest_l23_topics_file(DATA_DIR)
    if not input_file:
        print("No l23_topics_*.json file found")
        return
    
    print(f"Found input file: {input_file}")
    
    # Extract timestamp from input filename
    input_filename = os.path.basename(input_file)
    timestamp_match = re.search(r'l23_topics_(\d+)\.json', input_filename)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        print("Unable to extract timestamp from input filename, using current time")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Generate output filename using same timestamp
    output_filename = f"l23_filtered_topics_{timestamp}.json"
    output_file = os.path.join(DATA_DIR, output_filename)
    
    # Process file
    try:
        filter_instance.process_json_file(input_file, output_file)
    except Exception as e:
        print(f"Error occurred during processing: {e}")

if __name__ == "__main__":
    main()
