# LIVEVQA

## Benchmark

Please refer to the `liveVQA_benchmarks/README.md` for detailed information.

## NEWS

### News Collector

This module can help you collect news from BBC, CNN, Forbes, AP and Variety.

#### How to start?

Before collecting news, you need to do settings in `collectors/config.py`.

After simple settings, you can run the following command to collect news articles:

```bash
cd LIVEVQA
conda create -n news python=3.10.16 -y
conda activate news
pip install -r requirements.txt
python run.py
```

Every time your run the command, it will collect news articles and save them in `hot_topics_{timestamp}.json`.

### Image Filter

This module can rank and filter irrelevant images from the collected news articles.

#### How to start?

You should set your api key and base path in 'ranking/config.py'. After that, you can run the following command to filter images:

```bash
cd ranking
python Model_ranking.py
```

Every time your run the command, it will read the latest `hot_topics_{timestamp}.json` and filter images. The filtered file will be saved in `modified_topics_{timestamp}.json`.

### Level 1 QAs Generation and Filter

This module can generate and filter Level 1 QAs from the filtered news articles.

### How to start?

You should set your api key and base path in 'qa_makers/config.py' & 'qa_Filter/config.py'. After that, you can run the following command to generate Level 1 QAs:

```bash
cd qa_makers
python main.py
```

Every time your run the command, it will read the latest `modified_topics_{timestamp}.json` and filter images. The filtered file will be saved in `l1_topics_{timestamp}.json`.

```bash
cd qa_Filter
python main.py
```

Every time your run the command, it will read the latest `l1_topics_{timestamp}.json` and filter images. The filtered file will be saved in `l1_filtered_topics_{timestamp}.json`.

### Level 2 QAs Generation

This module can generate Level 2 QAs from the filtered Level 1 QAs.

#### How to start?

You should set your api key and base path in 'qa_makers_mh/config.py' & 'qa_Filter/config.py'. After that, you can run the following command to generate Level 2 QAs:

```bash
cd qa_makers_mh
python main.py
```
Every time your run the command, it will read the latest `l1_filtered_topics_{timestamp}.json` and filter images. The filtered file will be saved in `l23_topics_{timestamp}.json`.

### Automatic Pipeline

If you want to run the whole pipeline automatically, you can set your base path in `start.py` and run the following command:

```bash
python start.py
```

This will automatically collect news, filter images, generate Level 1 QAs, filter Level 1 QAs, and generate Level 2 QAs. The final output will be saved in `l23_topics_{timestamp}.json`.


## VIDEOS

### Video Collector

This module can help you collect videos from YouTube.

#### How to start?

Before collecting videos, you need to do settings in `video_code/video_pipeline.sh`, and make sure to download and configure the following repositories according to their instructions: 
https://github.com/zcczhang/UVD
https://github.com/opendatalab/DocLayout-YOLO
You should also modify the `demo.py` files in both folders based on the implementations in `uvd.py` and `doclayout.py`.

After simple settings, you can run the following command to collect YouTube videos:


```bash
cd LIVEVQA
conda create -n video python=3.9.0 -y
conda activate video
pip install -r requirements_video.txt
cd video_code
bash video_pipeline.sh
```

**Tips:** When setting up the environment, make sure to install both ffprobe and ffmpeg, otherwise the pipeline will fail with errors.

This module includes downloading the videos, spliting videos by text, Extracting keyframes, deduplication, selecting the final pictures. Finally, it would process a json named `modified_{timestamp}.json`, and the QA generation is the same as NEWS.

**Note:** We made a small modification to `qa_makers/main.py` — before generating QAs, the module now evaluates whether the associated text is meaningful enough for QA generation. Therefore, to generate QAs from videos, you should use the QA generation code provided in the `video_code` directory. Other components remain unchanged.


## ARXIV

This part can help you collect arxiv data

```bash
cd arxiv
pip install -r requirements_arxiv.txt
```

### Download papers

First you need to do settings in `arxiv/config.py`. Specifically, you need to change `BASE_DIR` to the directory where you want to save the downloaded papers. Then you can run the following command to download papers:

```bash
python direct_download.py --yearmonth 2504 --start-id 1 --end-id 100 --concurrent 5 --processes 4
```

You can see crawled data in `data/raw`.

### Preprocess papers

Process the downloaded papers to extract images and associations. You can run the following command:

```bash
python get_article.py --dir /path/to/html/files --workers 4
```

Then you can see the processed data in `data/processed`.

Set environment `OPENAI_API_KEY` to your OpenAI API key. Then run the following command to select the best images from the processed papers:

```bash
python select_best_images.py --input_dir /path/to/processed/jsons --workers 4 --start_index 0 --end_index 100
```

### Generate QAs

When synthesizing QAs about the authors, we just put all authors in all papers in `authors.json`.

You can run the following command to generate Level 1 QAs:

```bash
python construct_level1.py -i /path/to/processed/jsons -o /path/to/output/level1.jsonl --workers 4
```

You can run the following command to generate Level 2 QAs:

```bash
python construct_level2.py -i /path/to/output/level1.jsonl -o /path/to/output/level2.jsonl --processes 4
```