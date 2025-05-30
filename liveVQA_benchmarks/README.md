# VQA Benchmark Evaluation Framework

A comprehensive evaluation framework for Visual Question Answering (VQA) benchmarks that supports both API-based models (OpenAI, OpenRouter) and local models via OpenAI-compatible APIs.

## Overview

This framework provides two evaluation scripts:
- **`unified_benchmark.py`**: Original script supporting OpenAI and OpenRouter APIs
- **`enhanced_benchmark.py`**: Extended script adding local model support while maintaining full compatibility

Both scripts use identical prompts and evaluation criteria, ensuring consistent results across different model deployment methods.

## Project Structure

```
project_root/                          # Can be any path (e.g., /mnt/LIVEVQA, /home/user/vqa)
├── unified_benchmark.py               # Original API-only evaluation script
├── Evaluation/                        # Enhanced evaluation modules
│   ├── enhanced_benchmark.py          # Main enhanced script with local support
│   ├── local_config.py               # Local model configuration management
│   ├── local_client.py               # Local model client implementation
│   ├── unified_client.py             # Unified client interface (API + local)
│   ├── local_qa_handler.py           # QA processing with local model support
│   └── README.md                     # This documentation
├── News/                              # News benchmark dataset
│   ├── benchmark.json                # Questions and ground truth data
│   └── News_imgs/                    # News images
├── Paper/                             # Paper benchmark dataset
│   ├── benchmark.json
│   └── Paper_images/                 # Paper images
└── Video/                             # Video benchmark dataset
    ├── benchmark.json
    └── Video_images/                 # Video frame images
```

## Installation & Setup

### Prerequisites

```bash
pip install openai tqdm pathlib
```

### Environment Variables

Set API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
# OR for OpenRouter
export OPENROUTER_API_KEY="your-openrouter-key"
```

### Local Model Server Setup

For local models, ensure your OpenAI-compatible server is running:

**vLLM (Recommended)**:
```bash
vllm serve llava-hf/llava-v1.6-34b-hf --port 8000 --trust-remote-code
```

**Ollama**:
```bash
ollama pull llava:34b
ollama serve  # Default: http://localhost:11434
```

**text-generation-webui**:
```bash
python server.py --api --model your-model --port 8000
```

## Usage Guide

### Method 1: Original Script (API Only)

Run from project root directory:

```bash
# OpenAI GPT-4 Vision
python unified_benchmark.py -b Video -q gpt-4-vision-preview

# OpenRouter with Claude
python unified_benchmark.py -b Paper -q anthropic/claude-3-sonnet --api-provider openrouter

# Custom settings
python unified_benchmark.py -b News -q gpt-4o -w 10 -k your-api-key
```

### Method 2: Enhanced Script (API + Local Support)

Run from Evaluation directory:

```bash
cd Evaluation

# Local model for QA, API for judge
python enhanced_benchmark.py -b Video \
    --qa-model llava-v1.6-34b --qa-provider local \
    --judge-model gpt-4o-mini --judge-provider openai \
    --local-base-url http://localhost:8000/v1

# Full local setup
python enhanced_benchmark.py -b Paper \
    --qa-model qwen2-vl-72b --qa-provider local \
    --judge-model qwen2-7b-instruct --judge-provider local \
    --local-base-url http://localhost:8000/v1

# Custom project root
python enhanced_benchmark.py -b Video -q gpt-4o \
    --project-root /custom/path/to/vqa/project
```

## Command Line Reference

### Common Arguments (Both Scripts)
- `-b, --benchmark`: Dataset (News, Paper, Video) **[Required]**
- `-q, --qa-model`: QA model name **[Required for enhanced_benchmark.py]**
- `-j, --judge-model`: Judge model (default: gpt-4o-mini)
- `-w, --workers`: Worker threads (default: 5)
- `-k, --api-key`: API key override

### unified_benchmark.py Specific
- `--api-provider`: API provider (openai, openrouter)
- `--http-referer`: OpenRouter HTTP referer
- `--site-title`: OpenRouter site title

### enhanced_benchmark.py Specific
- `--qa-provider`: QA provider (openai, openrouter, local)
- `--judge-provider`: Judge provider (openai, openrouter, local)
- `--local-base-url`: Local server base URL
- `--judge-base-url`: Separate judge server URL
- `--api-delay`: API call delay in seconds (default: 1.0)
- `--project-root`: Project root directory (auto-detected)

## Configuration Examples

### Local Model Configuration

Edit `Evaluation/local_config.py` to add your models:

```python
LOCAL_MODELS = {
    "llava-v1.6-34b": LocalModelConfig(
        name="llava-v1.6-34b",
        base_url="http://localhost:8000/v1",
        temperature=0.2,
        max_tokens=2048
    ),
    "my-custom-model": LocalModelConfig(
        name="my-custom-model",
        base_url="http://localhost:8001/v1",
        temperature=0.1,
        max_tokens=1024
    )
}
```

### Mixed Provider Scenarios

```bash
# High-performance local QA + reliable API judge
python enhanced_benchmark.py -b Video \
    --qa-model llava-v1.6-34b --qa-provider local \
    --judge-model gpt-4o-mini --judge-provider openai

# Cost-effective OpenRouter QA + local judge
python enhanced_benchmark.py -b Paper \
    --qa-model anthropic/claude-3-haiku --qa-provider openrouter \
    --judge-model qwen2-7b-instruct --judge-provider local

# Multiple local servers
python enhanced_benchmark.py -b News \
    --qa-model llava-34b --qa-provider local --local-base-url http://gpu1:8000/v1 \
    --judge-model qwen2-7b --judge-provider local --judge-base-url http://gpu2:8001/v1
```

## Output Files & Results

### File Naming Convention

Results are saved in the benchmark directory with this pattern:
```
{benchmark}/
├── {model_name}_{provider}_results.json      # Detailed results
├── {model_name}_{provider}_summary.txt       # Statistical summary
└── benchmark.json                            # Original dataset
```

Example:
```
Video/
├── llava-v1.6-34b_local_results.json
├── llava-v1.6-34b_local_summary.txt
├── gpt-4-vision-preview_results.json
└── gpt-4-vision-preview_summary.txt
```

### Result Structure

**JSON Results** (`*_results.json`):
```json
{
  "id": "video_001",
  "original_img_filename": "frame_001.jpg",
  "source": "news_video",
  "time": "2024-01-15",
  "level1": {
    "question": "What is the main subject in the image?",
    "question_type": "object_identification",
    "img_filename_from_json": "frame_001.jpg",
    "img_path_used": "/path/to/Video/Video_images/frame_001.jpg",
    "ground_truth_list": ["person", "human", "man"],
    "model_qa_output": {
      "answer": "A person standing in the frame",
      "confidence_score": 85.0
    },
    "judge_evaluation": {
      "grade": "A",
      "reasoning": "AI Grader result: A",
      "status_for_stats": "CORRECT"
    }
  },
  "level2": {
    // Similar structure for level 2 questions
  }
}
```

**Summary Statistics** (`*_summary.txt`):
```
--- llava-v1.6-34b @ Video Benchmark ---
QA Provider: local
Judge Model: gpt-4o-mini (openai)
Timestamp: 2024-01-15 14:30:22

Level1:
  Total: 150
  Correct (A): 120
  Incorrect (B): 25
  Not Attempted (C): 5
  Errors: 0
  Accuracy (A/(A+B)): 82.76%
  Overall Accuracy (A/(A+B+C)): 80.00%

Level2:
  Total: 150
  Correct (A): 95
  Incorrect (B): 40
  Not Attempted (C): 15
  Errors: 0
  Accuracy (A/(A+B)): 70.37%
  Overall Accuracy (A/(A+B+C)): 63.33%

Overall:
  Total Questions: 300
  Correct: 215
  Incorrect: 65
  Not Attempted: 20
  Accuracy: 76.79%
```

## Evaluation Process

### 1. Question Answering Phase
- Images are base64-encoded and sent with questions
- Uses identical prompt template: `ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE`
- Expected JSON response: `{"answer": "text", "confidence_score": number}`
- Supports fallback parsing for malformed JSON

### 2. Answer Judging Phase
- Uses `GRADER_TEMPLATE` for consistent AI-based evaluation
- Compares model answers against ground truth lists
- Returns grades: A (CORRECT), B (INCORRECT), C (NOT_ATTEMPTED)
- Handles multiple valid answers per question

### 3. Statistical Analysis
- Calculates accuracy metrics: A/(A+B) and A/(A+B+C)
- Breaks down by question type and difficulty level
- Generates comprehensive reports

## Path Handling & Portability

### Automatic Path Detection
The framework automatically detects the project structure:
1. When run from `Evaluation/`, finds parent as project root
2. Validates required directories (News/, Paper/, Video/)
3. Checks for benchmark.json and image directories

### Manual Path Specification
```bash
# Specify custom project root
python enhanced_benchmark.py -b Video -q model \
    --project-root /custom/path/to/vqa/data

# Run from anywhere
python /path/to/project/Evaluation/enhanced_benchmark.py \
    -b Video -q model --project-root /path/to/project
```

## Resume & Checkpoint Features

Both scripts support automatic resume functionality:
- Tracks processed items by ID in output JSON files
- Skips already-processed items on restart
- Safely handles concurrent execution and interruptions
- Incrementally saves results to prevent data loss

Example resume scenario:
```bash
# Initial run (interrupted after 50/200 items)
python enhanced_benchmark.py -b Video -q llava-34b --qa-provider local

# Resume automatically continues from item 51
python enhanced_benchmark.py -b Video -q llava-34b --qa-provider local
# Output: "Resuming: 50 processed, 150 remaining"
```

## Performance Optimization

### Threading & Concurrency
- Default: 5 worker threads (adjustable with `-w`)
- Thread-safe client management with thread-local storage
- Safe incremental result saving with file locks

### API Rate Limiting
- Configurable delays between API calls (`--api-delay`)
- Separate delay handling for QA and judge models
- Built-in retry logic for transient failures

### Local Model Optimization
- Connection pooling and health checks
- Configurable timeouts and parameters per model
- Memory-efficient image encoding

## Troubleshooting

### Common Issues

**1. Path/Structure Errors**
```
FileNotFoundError: Benchmark directory not found
```
- Ensure you're in the correct directory
- Use `--project-root` to specify path explicitly
- Verify benchmark directories exist (News/, Paper/, Video/)

**2. Local Model Connection**
```
Local model connection test failed
```
- Check if server is running: `curl http://localhost:8000/v1/models`
- Verify correct port and URL in `--local-base-url`
- Check server logs for errors

**3. API Authentication**
```
API key not properly configured
```
- Set environment variable: `export OPENAI_API_KEY="..."`
- Or use command line: `-k your-api-key`

**4. JSON Parsing Issues**
```
Fallback JSON parsing
```
- This is normal - the system handles malformed responses automatically
- Check model output quality if this occurs frequently

### Debug Tips

```bash
# Enable debug logging
export PYTHONPATH="/path/to/project:$PYTHONPATH"
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"

# Test single item processing
python enhanced_benchmark.py -b Video -q model --workers 1

# Verify model availability
python -c "
from Evaluation.unified_client import is_model_available
print(is_model_available('llava-34b', 'local', 'http://localhost:8000/v1'))
"
```

## Model Recommendations

### QA Models
- **High Accuracy**: GPT-4o, Claude-3.5 Sonnet, GPT-4 Vision
- **Cost-Effective**: GPT-4o-mini, Claude-3 Haiku
- **Local High-Performance**: LLaVA-1.6-34B, Qwen2-VL-72B, InternVL2-76B
- **Local Efficient**: LLaVA-1.5-13B, Qwen2-VL-7B

### Judge Models
- **Recommended**: GPT-4o-mini (optimal balance of speed, accuracy, cost)
- **High Accuracy**: GPT-4o, Claude-3.5 Sonnet
- **Local Options**: Qwen2-7B-Instruct, Llama-3-8B-Instruct

## Advanced Usage

### Batch Processing Multiple Models
```bash
#!/bin/bash
# Script to evaluate multiple models
for model in "gpt-4o" "claude-3-sonnet" "llava-34b"; do
    for benchmark in "News" "Paper" "Video"; do
        if [[ $model == "llava-34b" ]]; then
            python enhanced_benchmark.py -b $benchmark -q $model --qa-provider local
        else
            python enhanced_benchmark.py -b $benchmark -q $model --qa-provider openai
        fi
    done
done
```

### Custom Evaluation Scenarios
```bash
# Research scenario: Compare API vs Local for same model family
python enhanced_benchmark.py -b Video -q gpt-4o --qa-provider openai
python enhanced_benchmark.py -b Video -q llava-34b --qa-provider local

# Production scenario: Fast local QA + reliable cloud judge
python enhanced_benchmark.py -b Video \
    --qa-model qwen2-vl-7b --qa-provider local \
    --judge-model gpt-4o-mini --judge-provider openai \
    --workers 10 --api-delay 0.5
```

## Contributing & Extending

### Adding New Model Providers
1. Extend `unified_client.py` with new client class
2. Add configuration options in `local_config.py`
3. Update argument parsing in `enhanced_benchmark.py`
4. Add provider-specific logic in `local_qa_handler.py`

### Custom Prompt Templates
Both scripts use shared templates from `unified_benchmark.py`:
- `ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE`: QA prompt
- `GRADER_TEMPLATE`: Judge evaluation prompt

Modify these templates to customize evaluation behavior while maintaining consistency across provider types.
