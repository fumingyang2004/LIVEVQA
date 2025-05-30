import logging
import os
import json
import argparse
import glob
from tqdm import tqdm
import time
from openai import OpenAI
import multiprocessing
from functools import partial
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"log_image_selection_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Set OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Create Manager to manage shared objects
manager = multiprocessing.Manager()

# Use manager to create shared dict
token_usage = manager.dict({
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
})

# Add lock to protect multi-process token stats update
token_lock = manager.Lock()

def update_token_usage(usage_data):
    """Update token usage statistics"""
    with token_lock:
        token_usage["prompt_tokens"] += usage_data.prompt_tokens
        token_usage["completion_tokens"] += usage_data.completion_tokens
        token_usage["total_tokens"] += usage_data.total_tokens

def load_paper_data(json_path):
    """Load paper JSON data"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.warning(f"Failed to load {json_path}: {e}")
        return None

def extract_paper_info(paper_data):
    """Extract paper info, including title, abstract, and figure info"""
    title = paper_data.get("title", "")
    abstract = paper_data.get("abstract", "")
    figures = paper_data.get("figures", [])
    
    # Extract each figure and its context
    figure_info = []
    for i, fig in enumerate(figures):
        if "image_url" not in fig or "caption" not in fig:
            continue
            
        image_url = fig["image_url"]
        caption = fig["caption"]
        context = fig["context"]
        
        figure_info.append({
            "index": i,
            "image_local_path": fig["image_local_path"],
            "image_url": image_url,
            "caption": caption,
            "context": context
        })
    
    return {
        "title": title,
        "abstract": abstract,
        "figures": figure_info
    }

def rank_figures(title, abstract, figures, paper_id):
    """Use GPT-4 to rank all figures in the paper and recommend the number to select"""
    try:
        system_prompt = """Objective: Analyze the provided paper abstract and image captions to identify and rank figures that best serve as memorable "paper identifiers." A "paper identifier" image is one that is visually distinct, memorable, and highly specific to this paper's unique contributions, making it unlikely to appear in other papers. Seeing this image should ideally make someone think of *this specific paper*.

Inputs You Will Receive:

1.  Paper Abstract: A concise summary of the paper's research, methodology, and findings.
2.  Image Captions: A list of captions, each corresponding to an image within the paper.

Your Task:

1.  Understand the Core Contributions:
    * Read the Abstract to identify the primary contributions, methodologies, specific datasets, key theoretical concepts, or highly distinct results presented in the paper. What makes this paper stand out?

2.  Evaluate Each Image Caption for Memorability and Uniqueness:
    * For every image caption provided:
        * Assess how well the image (as described by its caption) visually represents the unique and memorable aspects identified from the abstract.
        * Strongly prioritize images described as:
            * Framework/Architectural Diagrams
            * Conceptual Diagrams
            * Striking or Unexpected Visualizations/Illustrations of Key Findings except for statistical visualizations
            * Highly Distinctive Scientific Illustrations
            * Flowcharts or Block Diagrams
        * Avoid giving high ranks to (these are typically *not* memorable identifiers):
            * Any Bars, Plots, Graphs, Maps and statistical visualizations are USELESS, you MUST NOT select them
            * Images of People, Animals, or Objects that are not unique to the paper
            * Image with rich text like summaries, challenges, conclusions or limitations
            * Tables, Equations, Algorithm Boxes/Pseudocode presented as images

3.  Rank All Figures:
    * Create a ranking for *all* provided figures based on their potential as memorable identifiers. The figure deemed the most unique and memorable identifier should be ranked first.
    * For each figure, provide a brief reason for its rank, specifically addressing its uniqueness, memorability, and connection to the paper's core novelties.

4.  Recommend a Selection Count:
    * Based on your ranking, decide on a `recommended_count` of figures (typically 1, 0-3) that you believe are the most effective and sufficient set of memorable identifiers for this paper.
    * If all the images are not unique or memorable like statistical visualizations, be brave to recommend 0 images.

5.  Explain Recommendation for Count:
    * Provide a `selection_reason` briefly explaining why you recommend selecting this particular number of figures.

Output Format:

Produce a single JSON object with the following structure:

Return a JSON object with the ranking information:
{
  "ranking": [
    {"index": figure_index_start_from_1, "reason": "Brief explanation of why this figure ranks here"}
  ],
  "recommended_count": number_of_figures_to_select,
  "selection_reason": "Brief explanation of why you recommend selecting this many figures"
}

The "ranking" array should contain ALL figures sorted by their value as paper identifiers, with the most valuable figure first.
"""

        # Prepare figure descriptions
        figure_descriptions = []
        for i, fig in enumerate(figures):
            if i >= 20:         # Limit to 20 images max
                break
            figure_descriptions.append(f"Figure {i+1}: {fig['caption']}"\
                                    #    "\nContext: {fig.get('context', 'No context provided')}"
                                       )
        
        figure_text = "\n\n".join(figure_descriptions)

        user_prompt = f"""Paper Title: {title}
Paper Abstract: {abstract}

Figures to rank:
{figure_text}
"""

        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Update token usage statistics
        update_token_usage(response.usage)
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure result contains ranking
        if "ranking" not in result:
            logging.warning(f"Paper {paper_id}: API response doesn't contain proper ranking format")
            # Create default ranking by index
            result = {"ranking": [{"index": i+1, "reason": "Default ranking"} for i in range(len(figures))]}
        
        # Get recommended number of images to select
        recommended_count = min(result.get("recommended_count", 3), 3)  # Default 3, max 3
        selection_reason = result.get("selection_reason", "Default selection based on ranking")
        
        logging.info(f"Paper {paper_id}: Successfully ranked {len(result['ranking'])} figures, recommended to select {recommended_count}")
        
        # Convert ranking result to figure list with rank
        ranked_figures = []
        for rank, item in enumerate(result["ranking"]):
            if not isinstance(item, dict) or "index" not in item:
                continue
                
            fig_idx = int(item["index"]) - 1  # Convert 1-based index to 0-based
            if 0 <= fig_idx < len(figures):
                fig = figures[fig_idx]
                ranked_figures.append({
                    "image_url": fig["image_url"],
                    "context": fig["context"],
                    "image_path": fig["image_local_path"],
                    "caption": fig["caption"],
                    "rank": rank + 1,  # Rank starts from 1
                    "reason": item.get("reason", f"Rank {rank+1}"),
                    "index": fig["index"]
                })
        
        return ranked_figures, recommended_count, selection_reason
        
    except Exception as e:
        logging.warning(f"Figure ranking failed: {e}")
        # Return original figure list, use default ranking, select 3 by default
        default_figures = [{
            "image_url": fig["image_url"],
            "caption": fig["caption"],
            "rank": i + 1,  # Default ranking
            "reason": f"Ranking failed: {str(e)}",
            "index": fig["index"]
        } for i, fig in enumerate(figures)]
        return default_figures, min(3, len(figures)), "Default selection due to ranking failure"
        
def process_paper(json_path, output_dir):
    """Process a single paper and select images"""
    try:
        # Load paper data
        paper_data = load_paper_data(json_path)
        if not paper_data:
            return None
        
        paper_id = paper_data.get("paper_id", os.path.basename(os.path.dirname(json_path)))
        logging.info(f"Processing paper: {paper_id}")
        
        # Extract paper info
        paper_info = extract_paper_info(paper_data)
        if not paper_info["figures"]:
            logging.warning(f"Paper {paper_id} has no figures")
            return None
        
        if paper_info['abstract'] == "":
            logging.warning(f"Paper {paper_id} has no abstract")
            return None
        
        # Rank all figures at once
        logging.info(f"Paper {paper_id}: Start ranking {len(paper_info['figures'])} figures and recommending selection count")
        ranked_figures, recommended_count, selection_reason = rank_figures(
            paper_info["title"], 
            paper_info["abstract"], 
            paper_info["figures"],
            paper_id
        )
        
        # Select images according to recommended count
        selected_figures = ranked_figures[:min(recommended_count, len(ranked_figures))]
        
        logging.info(f"Paper {paper_id}: Selected {len(selected_figures)} top-ranked figures from {len(ranked_figures)}")
        logging.info(f"Selection reason: {selection_reason}")
        
        # Prepare result
        result = {
            "paper_id": paper_id,
            "title": paper_info["title"],
            "all_figures": ranked_figures,
            "selected_figures": selected_figures,
            "recommended_count": recommended_count,
            "selection_reason": selection_reason
        }
        
        if len(selected_figures) == 0:
            logging.warning(f"Paper {paper_id} did not select any figures")
            return result
        
        # Save result
        output_path = os.path.join(Path(json_path).parent, f"{paper_id}_selected_images.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Paper {paper_id} processed, selected {len(selected_figures)} figures")
        return result
        
    except Exception as e:
        logging.warning(f"Error processing paper {os.path.basename(json_path)}: {e}")
        return None

def process_paper_wrapper(json_path):
    """Wrapper function for multiprocessing"""
    return process_paper(json_path, None)

def main():
    parser = argparse.ArgumentParser(description="Select the most representative images in papers")
    parser.add_argument("--input_dir", type=str, default='data/processed/json/2025-05-01', help="Directory containing associations.json files")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory to output results")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of worker processes for parallel processing")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of folders to process")
    parser.add_argument("--end_index", type=int, default=100, help="End index of folders to process")
    parser.add_argument("--force", action="store_true", help="Force reprocess files with existing results")
    args = parser.parse_args()
    
    # Collect directory paths containing associations.json
    json_files = []
    skipped_files = []
    
    # Traverse directories to find associations.json and processed *_selected_images.json
    for root, dirs, files in os.walk(args.input_dir):
        if "associations.json" in files:
            json_path = os.path.join(root, "associations.json")
            dir_name = os.path.dirname(json_path)
            
            # Check if corresponding selected_images.json exists
            if not args.force:
                # Get all *_selected_images.json files
                selected_files = glob.glob(os.path.join(root, "*_selected_images.json"))
                if selected_files:
                    # Extract paper_id for logging
                    try:
                        paper_id = os.path.basename(os.path.dirname(json_path))
                        skipped_files.append((json_path, paper_id))
                        continue  # Skip already processed files
                    except:
                        pass
            
            # If no selected_images.json found or --force is used
            json_files.append(json_path)
    
    # Sort by lexicographical order
    sorted_json_files = sorted(json_files)
    
    # Select files to process by index range
    start_idx = min(args.start_index, len(sorted_json_files))
    end_idx = min(args.end_index, len(sorted_json_files))
    selected_json_files = sorted_json_files[start_idx:end_idx]
    
    # Output skipped files info
    if skipped_files:
        logging.info(f"Skipped {len(skipped_files)} already processed files: {', '.join([pid for _, pid in skipped_files[:5]])}" + 
                   (f" etc..." if len(skipped_files) > 5 else ""))
    
    logging.info(f"After sorting, processing files in index range [{start_idx}:{end_idx}], total {len(selected_json_files)}")
    
    if len(selected_json_files) == 0:
        logging.info("No files to process, exiting")
        return
    
    start_time = time.time()
    
    # Multiprocessing
    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            process_func = partial(process_paper_wrapper)
            args_list = selected_json_files
            
            results = list(tqdm(
                pool.imap(process_func, args_list),
                total=len(args_list),
                desc="Processing papers"
            ))
    else:
        # Single process
        results = []
        for json_path in tqdm(selected_json_files, desc="Processing papers"):
            result = process_paper(json_path, None)
            results.append(result)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Summarize results 
    selected_folder_count = len(sorted_json_files[start_idx:end_idx])
    
    summary = {
        "total_papers_in_range": selected_folder_count,
        "skipped_papers": len(skipped_files),
        "processed_papers": len([r for r in results if r is not None]),
        "failed_papers": len([r for r in results if r is None]),
        "total_selected_images": sum(len(r.get("selected_figures", [])) for r in results if r is not None),
        "processing_time_seconds": processing_time,
        "token_usage": token_usage
    }
    
    usage_dict = dict(token_usage)
    logging.info(f"Token usage: prompt:{usage_dict['prompt_tokens']} completion:{usage_dict['completion_tokens']} total:{usage_dict['total_tokens']}")
    # logging.info(f"Estimated cost: ${cost['total_cost']:.4f} (prompt:${cost['prompt_cost']:.4f} completion:${cost['completion_cost']:.4f})")
    logging.info(f"Processing time: {processing_time:.2f} seconds")
    
    if args.output_dir:
        summary_path = os.path.join(args.output_dir, "selection_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Processing complete. Skipped {summary['skipped_papers']} already processed papers, newly processed {summary['processed_papers']} papers, selected {summary['total_selected_images']} images")
    
if __name__ == "__main__":
    main()