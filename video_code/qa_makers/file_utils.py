import os
import glob
import re
import json
import logging
import time
from typing import List, Dict, Any, Tuple, Optional

from qa_makers.config import DATA_DIR

print (f"file_utils:DATA_DIR: {DATA_DIR}")

logger = logging.getLogger(__name__)

def get_latest_modified_topics_file():
    files = glob.glob(os.path.join(DATA_DIR, "modified_topics_*.json"))
    if not files:
        return None  # 没有找到符合条件的文件
    # 按文件修改时间排序，返回最新的文件
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def get_timestamp_from_filename(filename):
    match = re.search(r'modified_topics_(\d+)\.json', filename)
    if match:
        return match.group(1)
    return None

def determine_io_files(args):
    if args.input:
        input_file = args.input
    else:
        if args.timestamp:
            timestamp = args.timestamp
            input_file = os.path.join(DATA_DIR, f"modified_topics_{timestamp}.json")
            if not os.path.exists(input_file):
                logger.error(f"Not found input file: {input_file}")
                return None, None, None
        else:

            files = sorted(
                [f for f in os.listdir(DATA_DIR) if f.startswith("modified_topics_") and f.endswith(".json")],
                key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)),
                reverse=True
            )
            if not files:
                logger.error(f"Not found input file in {DATA_DIR}")
                return None, None, None
            input_file = os.path.join(DATA_DIR, files[0])
            timestamp = files[0].replace("modified_topics_", "").replace(".json", "")
    
    if not args.timestamp:
        basename = os.path.basename(input_file)
        if "modified_topics_" in basename and ".json" in basename:
            timestamp = basename.replace("modified_topics_", "").replace(".json", "")
        else:
            timestamp = time.strftime("%m%d%H%M", time.localtime())
    
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(DATA_DIR, f"l1_topics_{timestamp}.json")
    
    return input_file, output_file, timestamp

def load_topics(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Processing topics error: {str(e)}")
        return []

def save_results(output_file, output_data, last_save_time):
    current_time = time.time()
    if current_time - last_save_time >= 1: 
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Save to {output_file}")
            return current_time
        except Exception as e:
            logger.error(f"Saving results error: {str(e)}")
    
    return last_save_time
