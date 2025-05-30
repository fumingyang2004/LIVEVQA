from bs4 import BeautifulSoup
from bs4.element import Tag
import os
import json
import re
from urllib.parse import urljoin, urlparse
import multiprocessing
import argparse
from tqdm import tqdm  # For progress bar display

def extract_image_paragraph_pairs(html_path):
    """
    Extract images, paragraphs, and association info from HTML file, format consistent with download_html1.py
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Change parser to lxml, consistent with download_html1.py
        soup = BeautifulSoup(content, 'lxml')
    
    # Get paper ID and date
    paper_id = os.path.basename(html_path).replace('.html', '')
    paper_date = os.path.basename(os.path.dirname(html_path))
    base_url = f"https://arxiv.org/abs/{paper_id}"
    
    # Define base path for local image storage
    image_base_path = f"data/raw/images/{paper_date}/{paper_id}"
    
    json_path = html_path.replace('/html/', '/json/').replace('.html', '.json')
    
    # Try to read data from JSON file
    json_data = {}
    if os.path.exists(json_path):
        try:
            # print(f"Trying to read JSON data: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
        except Exception as e:
            print(f"Failed to read JSON data: {json_path}, error: {e}")
    
    # --- Extract paragraphs ---
    paragraphs = []
    for p in soup.find_all('p'):
        text = p.get_text(strip=True)
        if text:  # No length filter, keep consistent with download_html1.py
            paragraphs.append(text)
    
    # --- Extract title ---
    # First get from JSON, if not present then get from HTML
    title = ""
    if 'title' in json_data and json_data['title']:
        title = json_data['title']
    else:
        title_elem = soup.find('h1')
        if title_elem:
            title = title_elem.get_text(strip=True)
    
    # --- Extract abstract ---
    abstract = ""
    if 'abstract' in json_data and json_data['abstract']:
        abstract = json_data['abstract']
    else:
        abstract_elem = soup.find('blockquote', class_='abstract') or soup.find('div', class_='abstract')
        if abstract_elem:
            abstract = abstract_elem.get_text(strip=True).replace("Abstract: ", "").replace("Abstract ", "")
    
    # --- Extract authors info ---
    authors = []
    if 'authors' in json_data and isinstance(json_data['authors'], list):
        authors = json_data['authors']
    else:
        print(f"No author info found in JSON, using empty list by default")
        
    if abstract == '' or len(authors) == 0:
        raise ValueError(f"Abstract or author info missing, cannot process file: {html_path}")
    
    # --- Extract images and associations ---
    # Prepare all <p> tag list
    all_p_tags = [p for p in soup.find_all('p') if p.get_text(strip=True)]
    
    # Get image URLs
    image_urls = extract_images_from_html(soup, paper_id, base_url)
    
    # Build figures list, structure consistent with download_html1.py
    figures = []
    for image_url in image_urls:
        # Find corresponding image element
        img_elements = soup.find_all('img', src=lambda src: src and image_url in src)
        if not img_elements:
            continue
        
        img = img_elements[0]
        
        # Extract image caption - consistent with download_html1.py
        caption = extract_image_caption(img)
        
        # Find context paragraphs before and after image - same logic as download_html1.py
        context_paragraphs = find_context_paragraphs_like_download_html1(soup, img, all_p_tags)
        
        # Get image filename
        img_filename = os.path.basename(image_url)
        
        # Build local image path
        image_local_path = os.path.join(image_base_path, img_filename)
        
        # Build structure consistent with download_html1.py output
        figure_data = {
            "image_url": image_url,
            "image_local_path": image_local_path,  # Set local image path
            "caption": caption,
            "context": context_paragraphs
        }
        figures.append(figure_data)
    
    # Build final output result
    result = {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "paper_id": paper_id,
        "date": paper_date,
        "paragraphs": paragraphs,
        "figures": figures,
        "success": True
    }
    
    return result

def extract_image_caption(img):
    """Extract image caption according to download_html1.py logic"""
    caption = ""
    
    # First look for figcaption in figure tag
    figure_parent = img.find_parent('figure')
    if figure_parent:
        figcaption = figure_parent.find('figcaption')
        if figcaption:
            caption = figcaption.get_text(strip=True)
        # If no figcaption, try img's alt or title attribute
        elif img.get('alt'):
            caption = img.get('alt')
        elif img.get('title'):
            caption = img.get('title')
            
    big_picture = figure_parent.find_parent('figure')
    if big_picture:
        fig_caption = big_picture.find('figcaption', recursive=False)
        if fig_caption:
            caption = caption+'\n' + fig_caption.get_text(strip=True)
    
    # If still no caption found, check surrounding elements
    if not caption:
        # Check next sibling elements
        next_elem = img.find_next_sibling(['div', 'span', 'p'])
        if next_elem and (next_elem.get('class') and any('caption' in c.lower() for c in next_elem.get('class', [])) 
                         or 'caption' in next_elem.get('id', '').lower()):
            caption = next_elem.get_text(strip=True)
    
    return caption

def extract_images_from_html(soup, paper_id, base_url):
    """Extract image URLs exactly as in download_html1.py"""
    image_urls = []
    
    # Patterns to skip
    skip_patterns = [
        'arxiv-logo',
        'cornell-reduced',
        'icon',
        'logo',
        'static.arxiv.org/images',
        'static.arxiv.org/icons',
        'inline_',
        'data:image/',
        'favicon',
    ]
    
    def should_skip_url(url):
        """Check if URL should be skipped by pattern"""
        return any(pattern in str(url).lower() for pattern in skip_patterns)
    
    # 1. First look for images in figure tags
    for figure in soup.find_all('figure'):
        img = figure.find('img')
        if img and img.get('src'):
            if img.get('alt') == '[Uncaptioned image]':
                continue
            src = img.get('src')
            if not should_skip_url(src):
                image_urls.append(src)
        
        # Check for image references in figcaption
        figcaption = figure.find('figcaption')
        if figcaption:
            img_ref = figcaption.find('img')
            if img_ref and img_ref.get('src'):
                src = img_ref.get('src')
                if not should_skip_url(src):
                    image_urls.append(src)
    
    # 2. Find images with specific class names
    figure_classes = ['ltx_figure', 'ltx_figure_panel', 'ltx_graphics', 'figure', 'figure-panel']
    for img in soup.find_all('img', class_=lambda c: c and any(cls in str(c) for cls in figure_classes)):
        if img.get('alt') == '[Uncaptioned image]':
            continue
        src = img.get('src')
        if src and not should_skip_url(src):
            image_urls.append(src)
    
    # 3. Find images with specific IDs
    figure_id_patterns = ['fig', 'figure', 'F', 'S']
    for img in soup.find_all('img', id=lambda i: i and any(pattern in str(i).lower() for pattern in figure_id_patterns)):
        if img.get('alt') == '[Uncaptioned image]':
            continue
        src = img.get('src')
        if src and not should_skip_url(src):
            image_urls.append(src)
    
    # 4. Find arXiv-specific image formats (x1.png, x2.png, etc.)
    for img in soup.find_all('img'):
        if img.get('alt') == '[Uncaptioned image]':
            continue
        src = img.get('src')
        if src and not should_skip_url(src):
            # Check if arXiv image format
            if re.match(r'^x\d+\.(png|jpg|jpeg|gif)$', os.path.basename(src), re.IGNORECASE):
                image_urls.append(src)
    
    # 5. If no images found above, get all non-skipped images
    if not image_urls:
        for img in soup.find_all('img'):
            if img.get('alt') == '[Uncaptioned image]':
                continue
            src = img.get('src')
            if src and not should_skip_url(src):
                image_urls.append(src)
    
    # Remove duplicates but keep order
    seen = set()
    image_urls = [url for url in image_urls if not (url in seen or seen.add(url))]
    
    return image_urls

def find_context_paragraphs_like_download_html1(soup, img, all_p_tags):
    """Find context paragraphs around image according to download_html1.py logic"""
    body = soup.body if soup.body else soup
    
    # Extract all tags (including paragraphs and images)
    flat_tags = [tag for tag in body.descendants 
                if isinstance(tag, Tag) and tag.name in ('p', 'figure', 'img')]
    
    # Find current image's position in tag list
    img_idx = -1
    try:
        img_idx = flat_tags.index(img)
    except ValueError:
        # If not found, try matching by src
        img_src = img.get('src', '')
        for i, tag in enumerate(flat_tags):
            if tag.name == 'img' and tag.get('src', '') == img_src:
                img_idx = i
                break
    
    if img_idx == -1:
        return []
    
    # Collect previous and next paragraphs
    before = []
    after = []
    
    # Look backwards for up to 2 previous paragraphs
    i = img_idx - 1
    while i >= 0 and len(before) < 2:
        tag = flat_tags[i]
        if tag.name == 'p' and tag.get_text(strip=True):
            text = tag.get_text(strip=True)
            if text:
                before.insert(0, text)
        i -= 1
    
    # Look forwards for up to 2 next paragraphs
    i = img_idx + 1
    while i < len(flat_tags) and len(after) < 2:
        tag = flat_tags[i]
        if tag.name == 'p' and tag.get_text(strip=True):
            text = tag.get_text(strip=True)
            if text:
                after.append(text)
        i += 1
    
    # Merge before and after text as context
    return before + after

def process_single_file(html_file_info):
    """
    Process a single HTML file and extract data
    
    Args:
        html_file_info: tuple (html_path, html_dir, output_base_dir), includes HTML file path, directory path, and output base directory
    
    Returns:
        Dictionary with processing result info
    """
    html_path, html_dir, output_base_dir = html_file_info
    html_file = os.path.basename(html_path)
    result = {
        "file": html_file,
        "success": False,
        "figures_count": 0,
        "paragraphs_count": 0
    }
    
    try:
        # Extract data
        extracted_data = extract_image_paragraph_pairs(html_path)
        # If no figures, skip
        if len(extracted_data['figures']) == 0:
            result["sucess"] = 0
            return result
        
        paper_id = os.path.splitext(html_file)[0]
        paper_date = os.path.basename(html_dir)
        
        # Determine output path - use provided output base directory
        output_dir = os.path.join(output_base_dir, "json", paper_date, paper_id)
        os.makedirs(output_dir, exist_ok=True)
        associations_path = os.path.join(output_dir, "associations.json")
        
        # Output result
        with open(associations_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        # Create a symlink next to the original file pointing to associations.json for easy access
        json_link_path = os.path.join(html_dir, f"{paper_id}.json")
        if os.path.exists(json_link_path):
            os.remove(json_link_path)  # Remove if exists
        
        # Create relative symlink so it's still valid if directory is moved
        relative_path = os.path.relpath(associations_path, html_dir)
        os.symlink(relative_path, json_link_path)
        
        result["success"] = True
        result["figures_count"] = len(extracted_data['figures'])
        result["paragraphs_count"] = len(extracted_data['paragraphs'])
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

def main():
    """
    Main function: process all HTML files in directory in parallel, extract content and save as JSON
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parallel extraction of images and paragraphs from arXiv papers')
    parser.add_argument('--dir', type=str, required=True,
                      help='Directory containing HTML files')
    parser.add_argument('--output', type=str, default=None,
                      help='Output base directory, defaults to processed directory under parent of input directory')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                      help='Number of parallel processes, defaults to CPU core count')
    parser.add_argument('--force', action='store_true',
                      help='Force reprocess all files even if already processed')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit number of files to process, default is all')
    args = parser.parse_args()
    
    html_dir = args.dir
    os.makedirs(html_dir, exist_ok=True)
    paper_date = os.path.basename(html_dir)
    
    # Determine output base directory
    if args.output:
        output_base_dir = args.output
    else:
        # Default to processed directory under parent of input directory
        # e.g. /media/sata3/cdp/livevqa-arxiv/data/raw/html/2025-05-01 -> /media/sata3/cdp/livevqa-arxiv/data/processed
        raw_dir = os.path.dirname(html_dir)  # /media/sata3/cdp/livevqa-arxiv/data/raw/html
        data_dir = os.path.dirname(os.path.dirname(raw_dir))  # /media/sata3/cdp/livevqa-arxiv/data
        output_base_dir = os.path.join(data_dir, "processed")
    
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Output base directory: {output_base_dir}")
    
    # Get all HTML files
    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    
    # Filter out already processed files
    if not args.force:
        unprocessed_files = []
        skipped_files = []
        
        for html_file in html_files:
            paper_id = os.path.splitext(html_file)[0]
            output_dir = os.path.join(output_base_dir, "json", paper_date, paper_id)
            associations_path = os.path.join(output_dir, "associations.json")
            
            if os.path.exists(associations_path):
                skipped_files.append(html_file)
            else:
                unprocessed_files.append(html_file)
        
        print(f"Found {len(html_files)} HTML files, {len(skipped_files)} already processed (will skip), {len(unprocessed_files)} to process")
        html_files = unprocessed_files
    else:
        print(f"Found {len(html_files)} HTML files, will force reprocess all files")
    
    if not html_files:
        print("No HTML files to process, exiting")
        return
    
    html_file_paths = [(os.path.join(html_dir, f), html_dir, output_base_dir) for f in html_files]
    html_file_paths = sorted(html_file_paths, key=lambda x: x[0])  # Sort by filename
    
    # Limit number of files to process
    if args.limit and args.limit > 0:
        html_file_paths = html_file_paths[:args.limit]
        print(f"Will only process first {args.limit} files")
    
    print(f"Using {args.workers} processes for parallel processing")
    
    # Use process pool to process all files in parallel
    with multiprocessing.Pool(processes=args.workers) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_single_file, html_file_paths),
            total=len(html_file_paths),
            desc="Processing HTML files"
        ))
    
    # Summarize results
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_figures = sum(r["figures_count"] for r in results if r["success"])
    total_paragraphs = sum(r["paragraphs_count"] for r in results if r["success"])
    
    print(f"Processing complete: {successful} succeeded, {failed} failed")
    print(f"Total extracted: {total_figures} images, {total_paragraphs} paragraphs")
    
    # If there are failed files, print error info
    if failed > 0:
        print("\nFailed files:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['file']}: {r.get('error', 'Unknown error')}")
                
if __name__ == "__main__":
    main()