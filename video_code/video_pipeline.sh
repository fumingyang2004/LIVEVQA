#!/bin/bash

# Define common variables
SOURCE_DIR="Input your source directory here"
SEG_DIR="Input your segment directory here"

# Enable strict error handling
set -e  # Exit immediately on error
trap 'echo "Error occurred at line $LINENO: Process terminated"; exit 1' ERR

##Remember set the api keys list in the code
echo "Step 0: Getting videos"
python get_video.py --output_dir $SOURCE_DIR --start_date 2025-01-01 --end_date 2025-01-31 --max_videos_per_day 100|| { echo "Step 0 failed. Aborting."; exit 1; }

export OPENAI_API_KEY="Input your OpenAI API key here"
# Run processing steps
echo "Step 1: Retrieving video time information"
python get_time.py --source_dir $SOURCE_DIR || { echo "Step 1 failed. Aborting."; exit 1; }

echo "Step 2: Splitting videos"
python split.py --source_dir $SOURCE_DIR || { echo "Step 2 failed. Aborting."; exit 1; }

echo "Step 3: Extracting video segments"
python split_videos.py --source_dir $SOURCE_DIR --output_dir $SEG_DIR || { echo "Step 3 failed. Aborting."; exit 1; }

echo "Step 4: Processing topics"
python process_topic.py --segments-dir $SEG_DIR || { echo "Step 4 failed. Aborting."; exit 1; }

echo "Step 5: Extracting keyframes"
python UVD/uvd.py \
--root $SEG_DIR \
--preprocessor CLIP \
--shard_index 0 \
--shard_count 1 || { echo "Step 5 failed. Aborting."; exit 1; }

echo "Step 6: First deduplication"
python merge.py \
--root_dir $SEG_DIR || { echo "Step 6 failed. Aborting."; exit 1; }

echo "Step 7: Cropping videos"
python DocLayout-YOLO/DocLayout.py   \
  --model your_model_path\
  --root-dir $SEG_DIR || { echo "Step 7 failed. Aborting."; exit 1; }

echo "Step 8: Second deduplication"
python merge2.py \
--root_dir $SEG_DIR || { echo "Step 8 failed. Aborting."; exit 1; }

echo "Step 9: Selecting images"
python image_subtitle_selector.py \
--root_dir $SEG_DIR || { echo "Step 9 failed. Aborting."; exit 1; }

echo "Final Step: Generating JSON"
python process_video_segments.py \
--segments_dir $SEG_DIR \
--output_dir $OUTPUT_DIR || { echo "Final step failed. Aborting."; exit 1; }

echo "✅ Processing completed successfully."
