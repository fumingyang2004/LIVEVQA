# LIVEVQA

## 🚀 Environment

You can create a virtual environment and install the required packages using the following commands:

```bash
conda create -n livevqa python=3.9.0 -y
conda activate livevqa
pip install -r requirements.txt
```

## 📊 Benchmark

Please refer to the `liveVQA_benchmarks/README.md` for detailed information.

---

## 📰 NEWS

### News Collector

This module can help you collect news from BBC, CNN, Forbes, AP and Variety.

#### How to start?

Before collecting news, you need to do settings in `collectors/config.py`.

After simple settings, you can run the following command to collect news articles:

```bash
cd LIVEVQA
python run.py
```

Every time you run the command, it will collect news articles and save them in `hot_topics_{timestamp}.json`.

### Image Filter

This module can rank and filter irrelevant images from the collected news articles.

#### How to start?

You should set your api key and base path in `ranking/config.py`. After that, you can run the following command to filter images:

```bash
cd ranking
python Model_ranking.py
```

Every time you run the command, it will read the latest `hot_topics_{timestamp}.json` and filter images. The filtered file will be saved in `modified_topics_{timestamp}.json`.

### Level 1 QAs Generation and Filter

This module can generate and filter Level 1 QAs from the filtered news articles.

#### How to start?

You should set your api key and base path in `qa_makers/config.py` & `qa_Filter/config.py`. After that, you can run the following commands to generate Level 1 QAs:

**Generate Level 1 QAs:**
```bash
cd qa_makers
python main.py
```

Every time you run the command, it will read the latest `modified_topics_{timestamp}.json` and generate QAs. The output file will be saved in `l1_topics_{timestamp}.json`.

**Filter Level 1 QAs:**
```bash
cd qa_Filter
python main.py
```

Every time you run the command, it will read the latest `l1_topics_{timestamp}.json` and filter QAs. The filtered file will be saved in `l1_filtered_topics_{timestamp}.json`.

### Level 2 QAs Generation

This module can generate Level 2 QAs from the filtered Level 1 QAs.

#### How to start?

You should set your api key and base path in `qa_makers_mh/config.py`. After that, you can run the following command to generate Level 2 QAs:

```bash
cd qa_makers_mh
python main.py
```

Every time you run the command, it will read the latest `l1_filtered_topics_{timestamp}.json` and generate Level 2 QAs. The output file will be saved in `l23_topics_{timestamp}.json`.

### Automatic Pipeline

If you want to run the whole pipeline automatically, you can set your base path in `start.py` and run the following command:

```bash
python start.py
```

This will automatically:
1. Collect news
2. Filter images
3. Generate Level 1 QAs
4. Filter Level 1 QAs
5. Generate Level 2 QAs

The final output will be saved in `l23_topics_{timestamp}.json`.

---

## 🎥 VIDEOS

### Video Collector

This module can help you collect videos from YouTube.

#### How to start?

Before collecting videos, you need to:

1. **Configure settings** in `video_code/video_pipeline.sh`
2. **Download and configure** the following repositories according to their instructions:
   - https://github.com/zcczhang/UVD
   - https://github.com/opendatalab/DocLayout-YOLO
3. **Modify** the `demo.py` files in both folders based on the implementations in `uvd.py` and `doclayout.py`

#### ⚠️ Important: Torch Installation

Torch version may conflict with the CUDA version. We recommend checking your CUDA version:

```bash
nvcc --version
nvidia-smi
```

Then install the corresponding torch version:

**For CUDA 12.4:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For CUDA 11.8:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Running the Pipeline

After configuration, run the following command to collect YouTube videos:

```bash
cd LIVEVQA/video_code
bash video_pipeline.sh
```

> **💡 Tips:** Make sure to install both `ffprobe` and `ffmpeg`, otherwise the pipeline will fail with errors.

#### Pipeline Process

This module includes:
1. Downloading videos
2. Splitting videos by text
3. Extracting keyframes
4. Deduplication
5. Selecting final pictures

Finally, it processes a JSON file named `modified_{timestamp}.json`, and the QA generation follows the same process as NEWS.

> **📝 Note:** We made a small modification to `qa_makers/main.py` — before generating QAs, the module now evaluates whether the associated text is meaningful enough for QA generation. Therefore, to generate QAs from videos, you should use the QA generation code provided in the `video_code` directory. Other components remain unchanged.

---

## 📚 ARXIV

This section helps you collect ArXiv data.

```bash
cd arxiv
```

### Download Papers

First, configure settings in `arxiv/config.py`. Specifically, change `BASE_DIR` to the directory where you want to save the downloaded papers. Then run:

```bash
python direct_download.py --yearmonth 2504 --start-id 1 --end-id 100 --concurrent 5 --processes 4
```

You can see crawled data in `data/raw`.

### Preprocess Papers

Process the downloaded papers to extract images and associations:

```bash
python get_article.py --dir /path/to/html/files --workers 4
```

Then you can see the processed data in `data/processed`.

Set environment variable `OPENAI_API_KEY` to your OpenAI API key. Then run the following command to select the best images from the processed papers:

```bash
python select_best_images.py --input_dir /path/to/processed/jsons --workers 4 --start_index 0 --end_index 100
```

### Generate QAs

When synthesizing QAs about the authors, we put all authors from all papers in `authors.json`.

**Generate Level 1 QAs:**
```bash
python construct_level1.py -i /path/to/processed/jsons -o /path/to/output/level1.jsonl --workers 4
```

**Generate Level 2 QAs:**
```bash
python construct_level2.py -i /path/to/output/level1.jsonl -o /path/to/output/level2.jsonl --processes 4
```