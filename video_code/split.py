import os
import json
import glob
import re
import time
import argparse
import string
from openai import OpenAI
import logging
import webvtt
import google.generativeai as genai
import html
import concurrent.futures
from tqdm import tqdm
import threading

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
# Ensure the logs directory is under the script's path
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Create log file with absolute path
log_file = os.path.join(log_dir, f'process_captions_{time.strftime("%Y%m%d_%H%M%S")}.log')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4.1" 
# New - Split into full sentence chunks
def split_into_sentences(text, max_chunk_len=2500):
    sentences = re.split(r'(?<=[.!?;])\s+', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_len:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def read_text_file(file_path):
    """Read timestamped text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return ""

def read_metadata(file_path):
    """Read metadata JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read metadata {file_path}: {e}")
        return {}

def save_text_metadata(file_path, data):
    """Save processed text metadata"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save to {file_path}: {e}")
        return False

def save_text_file(file_path, content):
    """Save text file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save to {file_path}: {e}")
        return False

def format_timestamp(time_str):
    """Standardize timestamp format as HH:MM:SS.SSS"""
    # Add milliseconds if missing
    if '.' not in time_str:
        time_str += '.000'
    
    parts = time_str.split('.')
    time_parts = parts[0].split(':')
    ms = parts[1].ljust(3, '0')[:3]
    
    # Ensure hours, minutes, seconds are two digits
    hours = time_parts[0].zfill(2)
    minutes = time_parts[1].zfill(2)
    seconds = time_parts[2].zfill(2)
    
    return f"{hours}:{minutes}:{seconds}.{ms}"

def process_vtt_file(vtt_file_path, output_dir=None, subdir=None):
    if not os.path.exists(vtt_file_path):
        logger.error(f"VTT file {vtt_file_path} does not exist")
        return ""
    try:
        with open(vtt_file_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()

        # Remove empty lines and useless headers
        raw_lines = [line.strip() for line in raw_lines if line.strip()]
        raw_lines = [line for line in raw_lines if not line.startswith(('WEBVTT', 'Kind:', 'Language:'))]

        # Remove all <...> tags
        cleaned_lines = [re.sub(r"<[^>]+>", "", line) for line in raw_lines]

        # Parse by timestamps and text blocks
        blocks = []
        current_start = ""
        current_end = ""
        buffer = ""

        for line in cleaned_lines:
            time_match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})", line)
            if time_match:
                if buffer:
                    blocks.append({
                        "start_time": current_start,
                        "end_time": current_end,
                        "content": buffer.strip()
                    })
                    buffer = ""
                current_start, current_end = time_match.group(1), time_match.group(2)
            else:
                buffer += " " + line

        # Don't miss the last segment
        if buffer:
            blocks.append({
                "start_time": current_start,
                "end_time": current_end,
                "content": buffer.strip()
            })

        # Output raw txt format
        raw_formatted_text = ""
        for block in blocks:
            if block["content"]:
                raw_formatted_text += f"[{block['start_time']} --> {block['end_time']}] {block['content'].strip()}\n"

        # Save raw_txt
        if output_dir and subdir:
            raw_txt_file = os.path.join(output_dir, subdir, f"{subdir}_raw_vtt.txt")
            save_text_file(raw_txt_file, raw_formatted_text)

        # Step 1: Use prefix matching to merge blocks with the same prefix
        prefix_cleaned_blocks = pre_clean_captions(blocks)
        
        # Step 2: Handle completely duplicate content (only for text content)
        seen_contents = set()
        final_blocks = []
        for block in prefix_cleaned_blocks:
            content = block["content"].strip()
            if content and content not in seen_contents:
                seen_contents.add(content)
                final_blocks.append(block)

        # Generate processed text
        dedup_text = ""
        for block in final_blocks:
            dedup_text += f"[{block['start_time']} --> {block['end_time']}] {block['content']}\n"

        if output_dir and subdir:
            vtt_txt_file = os.path.join(output_dir, subdir, f"{subdir}_vtt.txt")
            save_text_file(vtt_txt_file, dedup_text)

        # Return the processed text, not the original
        return dedup_text

    except Exception as e:
        logger.error(f"Failed to process VTT file: {e}")
        return ""

def format_time(time_str):
    """Convert webvtt time format to HH:MM:SS.SSS format"""
    # Handle possible format differences
    if '.' not in time_str:
        time_str += '.000'
    hours, minutes, rest = time_str.split(':', 2)
    if '.' in rest:
        seconds, milliseconds = rest.split('.', 1)
    else:
        seconds = rest
        milliseconds = '000'
    milliseconds = milliseconds.ljust(3, '0')[:3]
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds}"

def check_text_validity(text_lines):
    """Check if text is valid"""
    if not text_lines:
        return False, "No text content found"
    
    # Build a text sample for checking
    sample_text = "\n".join([line["content"] for line in text_lines[:min(20, len(text_lines))]])
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You analyze the validity of transcribed text. Respond in JSON format."},
                {"role": "user", "content": f"""
Analyze if the following transcribed text is valid and meaningful. 
Check for:
1. Empty content or only background noises/music descriptions
2. Only filler words or meaningless utterances
3. Completely incoherent or nonsensical content

Text sample:
{sample_text}

Respond in JSON with the following structure:
{{
  "is_valid": true/false,
  "reason": "Brief explanation if invalid"
}}
"""}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("is_valid", False), result.get("reason", "Unknown")
    
    except Exception as e:
        logger.error(f"Failed to check text validity: {e}")
        return False, f"Error checking validity: {str(e)}"

def segment_text_by_events(text_lines, use_gemini=False):
    """
    Segments the transcript lines into coherent paragraphs based on topic content using an LLM.

    Args:
        text_lines (list): A list of dictionaries, each representing a transcript line
                           with 'start_time', 'end_time', and 'content'.
        use_gemini (bool): Flag to use Google Gemini instead of OpenAI.

    Returns:
        list: A list of segment dictionaries, each with 'start_time', 'end_time',
              'content', and 'is_outro'. Returns empty list on failure.
    """
    if not text_lines:
        return []

    formatted_text = ""
    for line in text_lines:
        # Ensure timestamps are in the expected format for the prompt if needed
        # Assuming start_time/end_time are already formatted like HH:MM:SS.SSS
        formatted_text += f"[{line['start_time']} --> {line['end_time']}] {line['content']}\n"

    # Updated System Prompt in English
    prompt_system = """
You are an expert at segmenting timestamped transcripts into coherent paragraphs based on **TOPIC CONTENT**.

**Segmentation Principles:**
1.  **MINIMIZE the number of segments - this is CRITICAL!** Aim for as few segments as possible.
2.  **TOPIC CONTINUITY is the PRIMARY criterion** – keep all content about the same event/topic together.
3.  If multiple sentences contain **SHARED KEYWORDS** or related concepts, they **MUST** be grouped together.
4.  If adjacent or nearby sentences mention the same entities (people, places, events), they **MUST** be merged.
5.  Changes in speaker or dialogue format should **NOT** create new segments if the topic remains related.
6.  **Only** create a new segment for a **COMPLETE TOPIC CHANGE** to an unrelated subject.

**Special Notes:**
-   If content is discussing different aspects of the same general topic (e.g., different angles or consequences of one event), keep it all in **ONE** segment.
-   Look for **semantic relationships** between sentences, not just superficial connections.
-   The goal is to create **COMPREHENSIVE segments** that cover complete topics, not short fragments.
-   **(New) Avoid creating excessively short segments:** If a segment's duration (end_time - start_time) is too short (e.g., less than 0.5 seconds), carefully verify if the segmentation is correct. Unless it's a very brief, distinct utterance representing a complete topic, try to merge it with adjacent, topically related segments.
-   If the content is a TV program outro, credits, music, or thank-you message that is distinct and lengthy, always segment it separately and mark it with `"is_outro": true`; these segments can be flagged as non-content. 

**Additional Guidance:**
-   You MAY use large time gaps (e.g., > 3 seconds) between subtitles as a **secondary** clue for segmentation, but **ALWAYS prioritize topic continuity over timing**. 
-   **PRIORITIZE content similarity over timestamp gaps** - related content should stay together even with pauses. 
-   **(New) Tendency to merge on short intervals:** As a **secondary signal**, if the time gap between the end of one line and the start of the next is **very short** (e.g., less than 1 second), and their topics are **related or continuous**, then they are **more likely** to belong to the same segment. This supports merging when topic continuity is present, but should not override a clear topic change.

**Example – Should be ONE segment (same policy topic):**
[00:00:01.000 --> 00:00:10.000] Content about Taiwan policy by different speakers or at different times.

**Example - All this should be ONE segment:**
[00:00:01.000 --> 00:00:05.000] Prime Minister says Malaysia will adopt a whole of nation approach to address the tariffs.
[00:00:05.000 --> 00:00:10.000] Criminal elements and negligence are factors in the probe into the gas pipeline explosion.
[00:00:10.000 --> 00:00:15.000] Gas supply disruptions are expected to last until April 20th.
*(Note: This example seems to contradict the "COMPLETE TOPIC CHANGE" rule as it mentions tariffs, an explosion probe, and gas disruptions which seem like distinct events/topics. A better example for ONE segment would show related aspects of *one* main issue. However, I will keep it as it was in your original prompt unless you want to change it.)*

**Timestamp Rules (Very Important):**
-   **(Refined)** For each segment in the final output:
    -   The `start_time` **MUST** be the **earliest `start_time` among all original lines included** in that segment.
    -   The `end_time` **MUST** be the **latest `end_time` among all original lines included** in that segment.
-   All timestamps must come directly from the original input lines.
-   **(New) Time Validity Check:** Ensure that for every segment, the `start_time` is strictly chronologically earlier than its `end_time`.
-   **(New) Segment Ordering and Gaps:** Ensure the segments in the final JSON output list are ordered **chronologically** based on their `start_time`. Theoretically, the `start_time` of the next segment should be later than or equal to the `end_time` of the preceding segment, potentially with a gap between them.

**Format your response as JSON:**
{
  "segments": [
    {
      "start_time": "Earliest start time from included lines",
      "end_time": "Latest end time from included lines",
      "content": "Full text content of segment",
      "is_outro": boolean (true if this segment contains program endings, credits or thank you messages)
    }
    // ... more segments
  ]
}

"""


    prompt_user = f"""
The following is a timestamped transcript from a video.
Each line follows this format: [START_TIME --> END_TIME] content

Your task:
1.  Segment this transcript into as **FEW** coherent segments as possible based on **topic content**.
2.  Keep all content discussing the same topic/event together in **ONE** segment.
3.  Look for shared keywords and semantic relationships between sentences to determine which should be merged.
4.  For segments that appear to be program endings, thank-you messages, or credits (e.g., "Thanks for watching", "See you tomorrow", "This has been News at 9"), mark them with `"is_outro": true`.
5.  For each segment, **strictly adhere to all rules regarding timestamp generation, validity, and segment spacing outlined in the system prompt**.

Here's the transcript:
{formatted_text}

Respond strictly in the JSON format described in the system prompt.
"""


    try:
        result = {}
        if use_gemini:
            # Configure Gemini (ensure API key is set via environment variables or other secure means)
            # genai.configure(api_key="YOUR_GEMINI_API_KEY") # Best practice: use env vars
            # model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred Gemini model
            # response = model.generate_content([prompt_system, prompt_user])
            # try:
            #     # Clean potential markdown code block markers
            #     cleaned_text = response.text.strip().lstrip('```json').rstrip('```').strip()
            #     result = json.loads(cleaned_text)
            # except (json.JSONDecodeError, AttributeError, ValueError) as e: # Added ValueError
            #     logger.error(f"Gemini returned invalid JSON or failed parsing: {response.text[:100]}... Error: {e}")
            #     result = {"segments": []}
            logger.warning("Gemini usage is placeholder - skipping actual call.")
            result = {"segments": []} # Placeholder if Gemini part not fully implemented/configured
        else:
            # Ensure client is OpenAI client and MODEL is defined
            if 'client' not in globals() or 'MODEL' not in globals():
                 logger.error("OpenAI client or MODEL not defined.")
                 return []

            response = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"}, # Requires compatible OpenAI model
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ],
                # Optional: Add temperature, max_tokens etc. if needed
                # temperature=0.2,
            )
            result = json.loads(response.choices[0].message.content)

        segments = result.get("segments", [])

        # Optional: Post-processing validation in Python code
        valid_segments = []
        last_end_time_sec = -1.0
        for i, segment in enumerate(segments):
            if isinstance(segment, dict) and \
               "start_time" in segment and \
               "end_time" in segment and \
               "content" in segment:

                # Ensure 'is_outro' field exists
                if "is_outro" not in segment:
                    segment["is_outro"] = False # Default to False if missing

                try:
                    start_sec = time_to_seconds(segment["start_time"])
                    end_sec = time_to_seconds(segment["end_time"])

                    # 1. Check start < end
                    if start_sec >= end_sec:
                        logger.warning(f"Segment {i} invalid timestamp (start >= end): {segment['start_time']} --> {segment['end_time']}")
                        continue # Skip invalid segment

                    # 2. Check duration > 0.5s (adjust threshold as needed)
                    duration = end_sec - start_sec
                    if duration < 0.5 and not segment.get("is_outro", False): # Allow short outros
                        logger.warning(f"Segment {i} potentially too short: {duration:.3f}s, Content: {segment['content'][:50]}...")
                        # Decide whether to keep or skip based on policy (keeping for now)

                    # 3. Check chronological order
                    if start_sec < last_end_time_sec:
                        # Calculate previous end time string for logging
                        prev_end_hms = time.strftime('%H:%M:%S', time.gmtime(last_end_time_sec))
                        prev_end_ms = int((last_end_time_sec % 1) * 1000)
                        prev_end_str = f"{prev_end_hms}.{prev_end_ms:03d}"
                        logger.warning(f"Segment {i} start time ({segment['start_time']}) is earlier than previous segment's end time ({prev_end_str}). Possible overlap.")
                        # Decide policy: skip, adjust, or allow? (Allowing for now)

                    valid_segments.append(segment)
                    # Update last end time only if the current segment is valid and chronologically sound
                    # If allowing overlap, we might want to take max(last_end_time_sec, end_sec)
                    # For simplicity, just using current end_sec here.
                    last_end_time_sec = end_sec

                except (ValueError, TypeError) as e: # Catch potential errors in time_to_seconds or type issues
                    logger.warning(f"Segment {i} timestamp format error or type issue: start='{segment.get('start_time')}', end='{segment.get('end_time')}'. Error: {e}")
                    continue # Skip segment with bad timestamps

            else:
                logger.warning(f"Segment {i} is malformed or missing required fields: {segment}")
                continue

        return valid_segments # Return only the validated segments

    except Exception as e:
        logger.error(f"Segmentation processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc()) # Log detailed stack trace
        return []

def organize_text_from_word_times(subdir_path, output_dir, subdir):
    """Directly read word timestamps from word_times.txt, use double-pointer method to organize into complete sentences and add timestamps"""
    try:
        logger.info(f"Start processing word timestamp file: {subdir}")
        
        # Read word_times.txt file
        word_times_file = os.path.join(subdir_path, "word_times.txt")
        if not os.path.exists(word_times_file):
            logger.warning(f"{subdir}: word_times.txt file not found")
            return ""
            
        # Parse words and their timestamps
        words_with_times = []
        with open(word_times_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse format: word <timestamp>
                match = re.match(r"(.+?)\s*<(\d{2}:\d{2}:\d{2}\.\d{3})>", line)
                if match:
                    word = match.group(1).strip()
                    timestamp = match.group(2)
                    words_with_times.append((word, timestamp))
        
        if not words_with_times:
            logger.warning(f"{subdir}: word_times.txt does not contain valid data")
            return ""
            
        # Build the complete raw text (without punctuation)
        raw_text = " ".join(word for word, _ in words_with_times)
        
        # Use LLM to add punctuation and organize sentences
        system_prompt = """
You are an expert at organizing transcribed speech into coherent text.

Your tasks:
1. Add proper punctuation (periods, commas, question marks, etc.) to make the text readable.
2. Structure the text into logical sentences, ensuring grammatical correctness.
3. Do NOT change any words, only add punctuation, do not add or delete any word.
4. Do NOT condense, summarize, or add additional words that aren't in the original text.
5. Break the text into complete sentences, with each sentence ending with appropriate terminal punctuation (.?!).
6. Output each sentence on a new line.

The input will be raw transcription text without punctuation.
"""

        user_prompt = f"""
Here is a raw speech transcript without punctuation. Please add appropriate punctuation and structure this into complete sentences.

Remember:
- Only add punctuation marks (,.!?;:""')
- Don't change the words or their order
- Don't add words that aren't in the transcript
- Output each sentence on a new line

Input text:
{raw_text}
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            temperature=0,
        )

        # Get punctuated text from LLM
        punctuated_text = response.choices[0].message.content.strip()
        logger.debug(f"Punctuated text from LLM: {punctuated_text[:100]}...")

        # Split into sentences
        sentences = [s.strip() for s in punctuated_text.split('\n') if s.strip()]
        
        # Preprocess word sequence for later matching
        processed_words = []
        for word, time in words_with_times:
            cleaned_word = clean_text(word)
            if cleaned_word:
                processed_words.append((cleaned_word, time))
        
        if not processed_words:
            logger.error(f"{subdir}: No valid words after cleaning")
            return ""
        
        # Find start and end time for each sentence
        result_with_timestamps = []
        word_pointer = 0  # Current position in processed_words
        
        logger.debug(f"Processed text: {sentences}")
        for sentence in sentences:
            # Extract words in sentence (lowercase and remove punctuation)
            sentence1 = sentence
            sentence_words = sentence1.split()
            if not sentence_words:
                continue
            
            # Initialize start and end positions
            start_idx = -1
            end_idx = -1
            matched_positions = []
            
            # Double-pointer matching algorithm
            s_pointer = 0  # Pointer to current sentence word
            skipped_words = 0  # Number of skipped words
            max_skips = 5  # Maximum allowed skipped words
            
            search_start = word_pointer  # Start searching from current position
            
            # Find the sentence start position
            while s_pointer < len(sentence_words) and word_pointer < len(processed_words):
                s_word = clean_text(sentence_words[s_pointer])
                w_word = clean_text(processed_words[word_pointer][0])
                
                logger.debug(f"Compare: '{w_word}' vs '{s_word}'")
                
                # Word matches or similarity is high enough
                if s_word == w_word or similarity_score(s_word, w_word) > 0.7:
                    if start_idx == -1:
                        start_idx = word_pointer
                    
                    matched_positions.append(word_pointer)
                    word_pointer += 1
                    s_pointer += 1
                    logger.debug(f"Match success, matches: {len(matched_positions)}")
                else:
                    # Not match, try skipping words in the original text
                    skip_found = False
                    
                    # Look ahead up to max_skips words to find a match
                    for ahead in range(1, max_skips + 1):
                        if word_pointer + ahead < len(processed_words):
                            next_word = clean_text(processed_words[word_pointer + ahead][0])
                            if s_word == next_word or similarity_score(s_word, next_word) > 0.7:
                                # Found match, skip intermediate words
                                skipped_words += ahead
                                word_pointer += ahead
                                if start_idx == -1:
                                    start_idx = word_pointer
                                
                                matched_positions.append(word_pointer)
                                word_pointer += 1
                                s_pointer += 1
                                skip_found = True
                                logger.debug(f"Match after skipping {ahead} words")
                                break
                    
                    # If no match found, try skipping sentence word
                    if not skip_found:
                        s_pointer += 1
                        logger.debug(f"Skip word in sentence")
            
            # If no valid start position found, use current position
            if start_idx == -1:
                start_idx = search_start
                logger.warning(f"Could not find good match for sentence '{sentence[:30]}...', using position {start_idx}")
            
            # Set end index as the last matched position
            end_idx = matched_positions[-1] if matched_positions else (start_idx + len(sentence_words) - 1)
            end_idx = min(end_idx, len(processed_words) - 1)
            
            logger.info(f"Sentence: '{sentence[:30]}...' matched from {start_idx} to {end_idx}, matched {len(matched_positions)}/{len(sentence_words)} words")
            
            # Get timestamps
            start_time = words_with_times[start_idx][1]
            end_time = words_with_times[end_idx][1]
            
            # Ensure sentence time is continuous (start time not earlier than previous end time)
            if result_with_timestamps:
                prev_end_time = result_with_timestamps[-1].split("] ")[0].split(" --> ")[1]
                if time_to_seconds(start_time) < time_to_seconds(prev_end_time):
                    start_time = prev_end_time
            
            # Ensure time order is reasonable
            if time_to_seconds(start_time) > time_to_seconds(end_time):
                end_time = start_time
            
            # Add to result
            result_with_timestamps.append(f"[{start_time} --> {end_time}] {sentence}")
            
            # Ensure pointer moves forward
            if word_pointer <= end_idx:
                word_pointer = end_idx + 1
        
        # Merge result
        final_text = "\n".join(result_with_timestamps)

        # Save final result
        final_txt_file = os.path.join(output_dir, subdir, f"{subdir}_final.txt")
        save_text_file(final_txt_file, final_text)

        return final_text
        
    except Exception as e:
        logger.error(f"Failed to organize text: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""

def similarity_score(word1, word2):
    """Calculate similarity between two words"""
    # Exact match
    if word1 == word2:
        return 1.0
    
    # Words with very different lengths are likely not the same
    if len(word1) == 0 or len(word2) == 0:
        return 0
    
    if abs(len(word1) - len(word2)) > max(len(word1), len(word2)) / 2:
        return 0
    
    # Simple character matching similarity
    common_chars = sum(1 for a, b in zip(word1, word2) if a == b)
    max_len = max(len(word1), len(word2))
    
    return common_chars / max_len
    
def extract_timestamps_and_content(text):
    """Extract timestamps and content from timestamped text, with relaxed format requirements"""
    # Preprocess: remove extra blank lines and spaces
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # More relaxed timestamp matching pattern
    pattern = r'\[\s*(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s*-->\s*(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s*\]\s*(.*?)(?=\n\[\s*\d{2}|\Z)'
    
    # Use re.DOTALL to match multiline content
    matches = re.findall(pattern, text, re.DOTALL)
    
    lines = []
    for start_time, end_time, content in matches:
        # Format timestamp, ensure milliseconds are standardized
        start_time = format_timestamp(start_time)
        end_time = format_timestamp(end_time)
        
        # Clean text content (remove extra spaces, newlines, etc.)
        cleaned_content = re.sub(r'\s+', ' ', content).strip()
        
        if cleaned_content:  # Only add non-empty content
            lines.append({
                "start_time": start_time, 
                "end_time": end_time,
                "content": cleaned_content
            })
    return lines

# Global lock to protect OpenAI API calls
api_lock = threading.Lock()

# Function to process a single directory, suitable for parallel execution
def process_single_directory(args):
    """Process a single directory, suitable for parallel execution"""
    subdir = args["subdir"]
    source_dir = args["source_dir"]
    output_dir = args["output_dir"]
    skip_existing = args["skip_existing"]
    use_gemini = args.get("use_gemini", False)
    
    subdir_path = os.path.join(source_dir, subdir)
    output_subdir = os.path.join(output_dir, subdir)
    os.makedirs(output_subdir, exist_ok=True)
    
    result = {
        "subdir": subdir,
        "processed": False,
        "empty_caption": False,
        "invalid_text": False,
        "skipped": False,
        "error": None
    }
    
    try:
        # Check if processed result already exists
        text_metadata_file = os.path.join(output_subdir, f"{subdir}_text_metadata.json")
        if skip_existing and os.path.exists(text_metadata_file):
            result["skipped"] = True
            return result
        
        metadata_file = os.path.join(subdir_path, f"{subdir}_metadata.json")
        if not os.path.exists(metadata_file):
            result["error"] = "Metadata file not found"
            return result
        
        metadata = read_metadata(metadata_file)

        # Check if word_times.txt file exists
        word_times_file = os.path.join(subdir_path, "word_times.txt")
        if not os.path.exists(word_times_file):
            result["error"] = "word_times.txt file not found"
            return result
            
        # Directly build timestamped full sentences from word_times.txt
        final_text = organize_text_from_word_times(subdir_path, output_dir, subdir)
        
        if not final_text:
            result["empty_caption"] = True
            result["error"] = "Unable to generate valid text from word_times.txt"
            return result

        # Extract text lines for further processing
        text_lines = extract_timestamps_and_content(final_text)
        
        # Check validity
        with api_lock:
            is_valid, invalid_reason = check_text_validity(text_lines)

        text_metadata = metadata.copy()
        text_metadata["is_text_valid"] = is_valid
        text_metadata["transcript_source"] = "word_times"

        if not is_valid:
            text_metadata["invalid_reason"] = invalid_reason
            text_metadata["segments"] = []
            result["invalid_text"] = True
        else:
            # LLM segmentation, pass use_gemini parameter
            with api_lock:
                segments = segment_text_by_events(text_lines, use_gemini=use_gemini)
            
            # Count outro segments
            outro_count = sum(1 for seg in segments if seg.get("is_outro", False))
                
            text_metadata["segments"] = segments
            text_metadata["contains_outro"] = outro_count > 0
        
        # Save text_metadata.json
        output_file = os.path.join(output_subdir, f"{subdir}_text_metadata.json")
        if save_text_metadata(output_file, text_metadata):
            result["processed"] = True
            
        return result
        
    except Exception as e:
        result["error"] = str(e)
        import traceback
        logger.error(f"Error processing {subdir}: {e}\n{traceback.format_exc()}")
        return result

def process_directory(source_dir, output_dir=None, use_gemini=False, skip_existing=False, max_workers=4):
    """Process all video transcripts in the specified directory in parallel"""
    if not output_dir:
        output_dir = source_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    logger.info(f"Found {len(subdirs)} subdirectories")
    
    # Prepare task parameters
    tasks = []
    for subdir in subdirs:
        tasks.append({
            "subdir": subdir,
            "source_dir": source_dir, 
            "output_dir": output_dir,
            "skip_existing": skip_existing,
            "use_gemini": use_gemini
        })
    
    # Use thread pool for parallel processing
    processed_count = 0
    empty_caption_count = 0
    invalid_text_count = 0
    skipped_count = 0
    error_count = 0
    
    # Use tqdm to show progress
    with tqdm(total=len(tasks), desc="Processing video directories") as pbar:
        # Use thread pool instead of process pool since most time is spent waiting for API response
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in tasks:
                futures.append(executor.submit(process_single_directory, task))
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                
                # Update progress bar
                pbar.update(1)
                
                # Update counters based on result
                if result["skipped"]:
                    skipped_count += 1
                    pbar.set_postfix({"Processed": processed_count, "Skipped": skipped_count})
                    continue
                    
                if result["error"]:
                    error_msg = result["error"]
                    subdir = result["subdir"]
                    error_count += 1
                    # Display error below progress bar
                    tqdm.write(f"❌ {subdir}: {error_msg}")
                    
                if result["processed"]:
                    processed_count += 1
                    if result["invalid_text"]:
                        invalid_text_count += 1
                    # Suppress output for successful processing to reduce output
                
                if result["empty_caption"]:
                    empty_caption_count += 1
                
                # Update progress bar status
                pbar.set_postfix({"Processed": processed_count, "Skipped": skipped_count})
    
    # Summary statistics
    total_dirs = len(subdirs)
    non_skipped = total_dirs - skipped_count
    empty_caption_ratio = empty_caption_count / non_skipped if non_skipped else 0
    invalid_text_ratio = invalid_text_count / (non_skipped - empty_caption_count) if (non_skipped - empty_caption_count) else 0

    logger.info("=" * 50)
    logger.info(f"Processing complete, successfully processed {processed_count}/{non_skipped} directories")
    logger.info(f"Skipped processed directories: {skipped_count}/{total_dirs} ({skipped_count/total_dirs:.2%})")
    logger.info(f"Directories with empty captions: {empty_caption_count}/{non_skipped} ({empty_caption_ratio:.2%})")
    logger.info(f"Directories with unclear text: {invalid_text_count}/{(non_skipped - empty_caption_count)} ({invalid_text_ratio:.2%})")
    logger.info(f"Directories with errors: {error_count}/{total_dirs}")
    logger.info("=" * 50)

    return processed_count, empty_caption_count, invalid_text_count

def time_to_seconds(time_str):
    """Convert HH:MM:SS.SSS format time to seconds"""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_ms = parts[2].split('.')
    seconds = int(seconds_ms[0])
    milliseconds = int(seconds_ms[1]) / 1000
    return hours * 3600 + minutes * 60 + seconds + milliseconds


def clean_text(text):
    """
    Clean text: remove punctuation, zero-width spaces, and other invisible characters, case-insensitive,
    decode HTML entities, filter all whitespace characters.
    :param text: Original text.
    :return: Cleaned text.
    """
    if not text:
        return ""
        
    text = text.lower()  # 转为小写

    # Decode HTML entities, convert &amp; etc. to corresponding characters
    text = html.unescape(text)

    # Remove zero-width spaces and other special invisible characters
    # U+200B (zero-width space), U+200C (zero-width non-joiner), U+200D (zero-width joiner),
    # U+FEFF (zero-width no-break space/BOM), U+200E (left-to-right mark), U+200F (right-to-left mark)
    # and other control characters that may cause issues
    text = re.sub(r'[\u200B-\u200F\uFEFF\u0000-\u001F\u007F-\u009F]', '', text)

    # Remove all other punctuation (at this point &amp; has become &)
    # Note: default string.punctuation does not include &, so & will be retained
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Filter all whitespace characters (including spaces, newlines, tabs, etc.)
    # Use regex \s+ to match one or more whitespace and replace with empty string
    text = re.sub(r'\s+', '', text)

    return text.strip()

def main():
    parser = argparse.ArgumentParser(description='Process video transcript text, check validity and segment events')
    parser.add_argument('--source_dir', type=str, default="",
                       help='Source directory containing video subdirectories')
    parser.add_argument('--output_dir', type=str, default="",
                       help='Output directory, defaults to same as source')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip folders with existing text_metadata.json')
    # parser.add_argument('--use_gemini', action='store_true', help='Use Gemini 2.0 Flash for segmentation')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of threads for parallel processing, default 4')
    args = parser.parse_args()

    start_time = time.time()
    processed, empty_count, invalid_count = process_directory(
        args.source_dir, 
        args.output_dir,
        skip_existing=args.skip_existing,
        max_workers=args.workers  
    )
    end_time = time.time()

    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logger.info(f"Successfully processed: {processed} videos")
    # Calculate average processing time per video
    if processed > 0:
        avg_time = (end_time - start_time) / processed
        logger.info(f"Average processing time per video: {avg_time:.2f} seconds")
        # Estimate total time for serial processing
        estimated_serial_time = avg_time * processed
        logger.info(f"Estimated total serial processing time: {estimated_serial_time:.2f} seconds")
        logger.info(f"Parallel speedup: {estimated_serial_time / (end_time - start_time):.2f}x")

if __name__ == "__main__":
    main()
