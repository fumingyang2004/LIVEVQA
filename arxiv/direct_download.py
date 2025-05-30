import argparse
import asyncio
import json
import multiprocessing
import os
import random
import re
import sys
import time
import ssl
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from functools import partial

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add parent directory to path to import config and utils
sys.path.append(str(Path(__file__).parent))
from config import (
    RAW_JSON_DIR, RAW_HTML_DIR, RAW_IMAGES_DIR,
    REQUEST_TIMEOUT, DELAY_BETWEEN_REQUESTS
)
from utils import setup_logger, sanitize_filename

logger = setup_logger(__name__)

# Define a list of realistic User-Agents to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.47'
]

def get_random_headers():
    """Generate random headers to appear more like a real browser"""
    user_agent = random.choice(USER_AGENTS)
    
    # List of possible accept languages
    accept_languages = [
        'en-US,en;q=0.9',
        'en-US,en;q=0.8',
        'en-GB,en;q=0.9,en-US;q=0.8',
        'en-CA,en;q=0.9,fr-CA;q=0.8',
        'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
    ]
    
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': random.choice(accept_languages),
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    
    # Randomly decide whether to include some headers to add variability
    if random.random() > 0.5:
        headers['DNT'] = '1'
    
    return headers

async def random_delay():
    """Add random delay between requests to appear more human-like"""
    # Random delay between 2-5 seconds
    delay = DELAY_BETWEEN_REQUESTS + random.uniform(2, 5)
    await asyncio.sleep(delay)

def parse_args():
    parser = argparse.ArgumentParser(description='Directly download arXiv papers by year-month and ID range.')
    parser.add_argument('--yearmonth', type=str, required=True,
                        help='Year and month in format YYMM (e.g., 2401 for January 2024)')
    parser.add_argument('--start-id', type=int, required=True,
                        help='Starting ID number')
    parser.add_argument('--end-id', type=int, required=True,
                        help='Ending ID number')
    parser.add_argument('--concurrent', type=int, default=5, 
                        help='Number of concurrent downloads per process (default: 5)')
    parser.add_argument('--processes', type=int, default=4,
                        help='Number of processes to use (default: 4)')
    
    return parser.parse_args()

async def download_html(session, paper_id, yearmonth, semaphore):
    """
    Download HTML content for a paper.
    
    Args:
        session (aiohttp.ClientSession): Async HTTP session
        paper_id (str): Paper ID in format YYMM.NNNNN
        yearmonth (str): Year and month in format YYMM
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent downloads
    
    Returns:
        tuple: (paper_id, success_status, html_content, publication_date)
    """
    # Format publication date from yearmonth (YYMM to YYYY-MM)
    publication_date = yearmonth
    
    # Create output directory
    date_dir = RAW_HTML_DIR / publication_date
    os.makedirs(date_dir, exist_ok=True)
    
    html_file = date_dir / f"{paper_id}.html"
    
    # Check if file already exists
    if html_file.exists():
        logger.info(f"HTML for paper {paper_id} already exists at {html_file}")
        return paper_id, True, None, publication_date
    
    # Construct URLs for arxiv
    abs_url = f"https://arxiv.org/abs/{paper_id}"
    html_url = f"https://arxiv.org/html/{paper_id}"
    
    async with semaphore:
        try:
            # Add a random delay before each request to avoid CAPTCHA
            await random_delay()
            
            # Try different URL patterns
            url_patterns = [
                html_url,
                html_url.replace('/html/', '/format/'),
                f"https://arxiv.org/format/{paper_id}",
                # Try to use PDF URL if HTML is blocked
                f"https://arxiv.org/pdf/{paper_id}.pdf"
            ]
            
            html_content = None
            
            # First try to get the abstract page to see if the paper exists
            headers = get_random_headers()
            async with session.get(abs_url, headers=headers, ssl=False) as response:
                if response.status != 200:
                    logger.warning(f"Paper {paper_id} does not exist (HTTP {response.status})")
                    return paper_id, False, None, publication_date
                
                # Extract publication date from abstract page if needed
                abs_content = await response.text()
                soup = BeautifulSoup(abs_content, 'html.parser')
                
                # If we got the abstract, try to download the HTML content
                for url in url_patterns:
                    try:
                        # Use fresh random headers for each URL pattern
                        headers = get_random_headers()
                        
                        async with session.get(
                            url, 
                            headers=headers,
                            allow_redirects=True,
                            ssl=False
                        ) as response:
                            if response.status == 200:
                                html_content = await response.text()
                                
                                # Check for CAPTCHA in response
                                if 'recaptcha' in html_content.lower() or 'captcha' in html_content.lower():
                                    logger.warning(f"CAPTCHA detected when downloading HTML for {paper_id} from {url}")
                                    await asyncio.sleep(30 + random.uniform(10, 30))
                                    continue
                                
                                # Basic validation of HTML content
                                if len(html_content) < 100 or not html_content.strip():
                                    logger.warning(f"Retrieved empty or too short HTML for {paper_id} from {url}")
                                    continue
                                    
                                if not any(marker in html_content.lower() for marker in ['<html', '<body', 'arxiv']):
                                    logger.warning(f"Retrieved non-HTML content for {paper_id} from {url}")
                                    continue
                                
                                # Save HTML content
                                async with aiofiles.open(html_file, 'w', encoding='utf-8') as f:
                                    await f.write(html_content)
                                
                                logger.info(f"Successfully downloaded HTML for {paper_id} from {url}")
                                
                                # Add json metadata from abstract page
                                json_dir = RAW_JSON_DIR / publication_date
                                os.makedirs(json_dir, exist_ok=True)
                                
                                # Extract information from HTML
                                title_tag = soup.find('h1', class_='title')
                                title = title_tag.text.replace('Title:', '').strip() if title_tag else ""
                                
                                abstract_tag = soup.find('blockquote', class_='abstract')
                                abstract = abstract_tag.text.replace('Abstract:', '').strip() if abstract_tag else ""
                                
                                authors_tag = soup.find('div', class_='authors')
                                authors = []
                                if authors_tag:
                                    author_links = authors_tag.find_all('a')
                                    authors = [a.text.strip() for a in author_links]
                                
                                categories_tag = soup.find('div', class_='subjects')
                                categories = []
                                primary_category = ""
                                if categories_tag:
                                    category_text = categories_tag.text.strip()
                                    category_matches = re.findall(r'([a-zA-Z\-\.]+)', category_text)
                                    categories = [c.strip() for c in category_matches if c.strip()]
                                    primary_category = categories[0] if categories else ""
                                
                                # Create paper dictionary
                                paper_dict = {
                                    "id": paper_id,
                                    "title": title,
                                    "abstract": abstract,
                                    "authors": authors,
                                    "first_author": authors[0] if authors else "",
                                    "all_authors": ", ".join(authors),
                                    "categories": categories,
                                    "date": publication_date,
                                    "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                                    "abstract_url": abs_url,
                                    "html_url": html_url,
                                    "primary_category": primary_category
                                }
                                
                                # Save JSON metadata
                                json_file = json_dir / f"{paper_id}.json"
                                async with aiofiles.open(json_file, 'w', encoding='utf-8') as f:
                                    await f.write(json.dumps(paper_dict, ensure_ascii=False, indent=2))
                                
                                # Processing images will be done after HTML download
                                return paper_id, True, html_content, publication_date
                            else:
                                logger.warning(f"Failed to download HTML for {paper_id} from {url} (HTTP {response.status})")
                    
                    except Exception as e:
                        logger.error(f"Error downloading HTML for {paper_id} from {url}: {str(e)}")
                    
                    # Add a small delay between URL attempts
                    await asyncio.sleep(1 + random.uniform(0.5, 2.0))
            
            logger.error(f"Failed to download HTML for {paper_id} after trying all URL patterns")
            return paper_id, False, None, publication_date
        
        except Exception as e:
            logger.error(f"Unexpected error downloading HTML for {paper_id}: {str(e)}")
            return paper_id, False, None, publication_date

def extract_images_from_html(html_content, paper_id, base_url):
    """
    Extract image URLs from HTML content, focusing on actual paper figures.
    
    Args:
        html_content (str): HTML content of the paper
        paper_id (str): Paper ID
        base_url (str): Base URL for resolving relative URLs
    
    Returns:
        list: List of image URLs
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    image_urls = []
    
    # Skip patterns for unwanted images
    skip_patterns = [
        'arxiv-logo', 'cornell-reduced', 'icon', 'logo',
        'static.arxiv.org/images', 'static.arxiv.org/icons',
        'inline_', 'data:image/'
    ]
    
    def should_skip_url(url):
        """Check if URL should be skipped based on patterns"""
        return any(pattern in url.lower() for pattern in skip_patterns)
    
    # 1. First priority: Look for images in figure tags
    for figure in soup.find_all('figure'):
        img = figure.find('img')
        if img and img.get('src'):
            src = img.get('src')
            if not should_skip_url(src):
                image_urls.append(src)
    
    # 2. Second priority: Look for images with specific classes that indicate figures
    figure_classes = ['ltx_figure', 'ltx_figure_panel', 'ltx_graphics', 'figure', 'figure-panel']
    for img in soup.find_all('img', class_=lambda c: c and any(cls in str(c) for cls in figure_classes)):
        src = img.get('src')
        if src and not should_skip_url(src):
            image_urls.append(src)
    
    # 3. Third priority: Look for arXiv specific figure formats (x1.png, x2.png, etc.)
    for img in soup.find_all('img'):
        src = img.get('src')
        if src and not should_skip_url(src):
            # Check if it's an arXiv figure format
            if re.match(r'^x\d+\.(png|jpg|jpeg|gif)$', os.path.basename(src), re.IGNORECASE):
                image_urls.append(src)
    
    # Remove duplicates while preserving order
    seen = set()
    image_urls = [url for url in image_urls if not (url in seen or seen.add(url))]
    
    return image_urls

async def download_image(session, image_url, paper_id, publication_date, base_url, semaphore):
    """
    Download an image.
    
    Args:
        session (aiohttp.ClientSession): Async HTTP session
        image_url (str): URL of the image
        paper_id (str): Paper ID
        publication_date (str): Paper publication date in YYYY-MM format
        base_url (str): Base URL for resolving relative paths
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent downloads
    
    Returns:
        tuple: (image_url, success_status, local_path)
    """
    # Skip data URLs and inline images
    if image_url.startswith('data:image/') or 'inline_' in image_url:
        return image_url, False, None
        
    # Skip logos and icons
    skip_patterns = [
        'arxiv-logo', 'cornell-reduced', 'icon', 'logo',
        'static.arxiv.org/images', 'static.arxiv.org/icons'
    ]
    if any(pattern in image_url.lower() for pattern in skip_patterns):
        return image_url, False, None

    # Create output directory
    date_dir = RAW_IMAGES_DIR / publication_date / paper_id
    os.makedirs(date_dir, exist_ok=True)
    
    # Get filename
    image_filename = os.path.basename(urlparse(image_url).path)
    if not image_filename:
        image_filename = f"image_{hash(image_url) % 10000}.png"
    
    # Ensure file extension
    if not any(image_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg']):
        image_filename += '.png'
    
    image_file = date_dir / sanitize_filename(image_filename)
    
    # Skip if exists
    if image_file.exists():
        return image_url, True, str(image_file.relative_to(RAW_IMAGES_DIR))
    
    async with semaphore:
        # Add a random delay between image requests
        await random_delay()
        
        # Construct primary URL
        primary_url = image_url
        if not image_url.startswith(('http://', 'https://')):
            paper_url_base = base_url
            if not paper_url_base.endswith('/'):
                paper_url_base += '/'
            primary_url = urljoin(paper_url_base, image_url)

        try:
            headers = get_random_headers()
            headers['Referer'] = base_url
            
            async with session.get(
                primary_url,
                headers=headers,
                ssl=False
            ) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if content_type.startswith('text/html'):
                        logger.warning(f"Received HTML instead of image for URL: {primary_url}")
                        return image_url, False, None
                    
                    image_data = await response.read()
                    if len(image_data) < 100:
                        logger.warning(f"Image too small ({len(image_data)} bytes), potentially invalid: {primary_url}")
                        return image_url, False, None
                    
                    async with aiofiles.open(image_file, 'wb') as f:
                        await f.write(image_data)
                    
                    return image_url, True, str(image_file.relative_to(RAW_IMAGES_DIR))
                
                elif response.status == 404 and re.match(r'^x\d+\.(png|jpg|jpeg|gif)$', image_filename, re.IGNORECASE):
                    # Try alternative paths for arxiv-style image names
                    paper_id_match = re.search(r'/(?:html|abs)/([^/]+)', base_url)
                    if paper_id_match:
                        paper_id_with_version = paper_id_match.group(1)
                        alt_url = f"https://arxiv.org/html/{paper_id_with_version}/figures/{image_filename}"
                        
                        await asyncio.sleep(1)
                        
                        alt_headers = get_random_headers()
                        alt_headers['Referer'] = base_url
                        
                        async with session.get(alt_url, headers=alt_headers, ssl=False) as alt_response:
                            if alt_response.status == 200:
                                alt_data = await alt_response.read()
                                
                                async with aiofiles.open(image_file, 'wb') as f:
                                    await f.write(alt_data)
                                
                                return image_url, True, str(image_file.relative_to(RAW_IMAGES_DIR))
                
                logger.warning(f"Failed to download image: {primary_url} (HTTP {response.status})")
                return image_url, False, None

        except Exception as e:
            logger.error(f"Error downloading image {primary_url}: {str(e)}")
            return image_url, False, None

async def process_paper(session, paper_id, yearmonth, dl_semaphore, img_semaphore):
    """
    Process a paper: download HTML and images.
    
    Args:
        session (aiohttp.ClientSession): Async HTTP session
        paper_id (str): Paper ID in format YYMM.NNNNN
        yearmonth (str): Year-month in format YYMM
        dl_semaphore (asyncio.Semaphore): Semaphore for HTML downloads
        img_semaphore (asyncio.Semaphore): Semaphore for image downloads
    
    Returns:
        dict: Paper data with download results
    """
    # Download HTML and get publication date
    html_id, html_success, html_content, publication_date = await download_html(session, paper_id, yearmonth, dl_semaphore)
    
    if not html_success:
        return {
            "id": paper_id,
            "success": False,
            "html_downloaded": False,
            "image_results": []
        }
    
    # If HTML was already downloaded previously, we don't have the content
    if html_content is None:
        return {
            "id": paper_id,
            "success": True,
            "html_downloaded": True,
            "image_results": []  # Skip image downloading for previously downloaded papers
        }
    
    # Extract and download images
    base_url = f"https://arxiv.org/html/{paper_id}"
    image_urls = extract_images_from_html(html_content, paper_id, base_url)
    
    # Download images
    image_tasks = [
        download_image(session, image_url, paper_id, publication_date, base_url, img_semaphore)
        for image_url in image_urls
    ]
    
    image_results = []
    if image_tasks:
        image_results = await asyncio.gather(*image_tasks)
    
    return {
        "id": paper_id,
        "success": True,
        "html_downloaded": True,
        "image_results": image_results
    }

async def process_batch_async(yearmonth, paper_ids, concurrent_limit=5):
    """
    Process a batch of paper IDs asynchronously.
    
    Args:
        yearmonth (str): Year-month in format YYMM
        paper_ids (list): List of paper IDs to process
        concurrent_limit (int): Concurrent download limit
    
    Returns:
        list: Results of processing
    """
    # Create semaphores to limit concurrent downloads
    dl_semaphore = asyncio.Semaphore(concurrent_limit)
    img_semaphore = asyncio.Semaphore(concurrent_limit)
    
    # Configure client session with improved SSL handling
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    conn = aiohttp.TCPConnector(
        ssl=ssl_context,  # Use custom SSL context instead of False
        limit=concurrent_limit,
        force_close=True,  # Avoid connection reuse issues
        enable_cleanup_closed=True,  # Clean up closed connections
        ttl_dns_cache=300  # DNS cache TTL in seconds
    )
    
    session_timeout = aiohttp.ClientTimeout(
        total=120,      # Total timeout
        connect=30,     # Connection timeout
        sock_read=60,   # Socket read timeout
        sock_connect=30 # Socket connect timeout
    )
    
    cookie_jar = aiohttp.CookieJar(unsafe=True)
    
    async with aiohttp.ClientSession(
        connector=conn,
        timeout=session_timeout,
        cookie_jar=cookie_jar,
        raise_for_status=False,  # Don't raise exceptions for HTTP status
        trust_env=True  # Use environment variables for proxy etc.
    ) as session:
        # Do a warm-up request
        try:
            logger.info(f"[Process {os.getpid()}] Performing warm-up request to arxiv.org...")
            async with session.get("https://arxiv.org", 
                                  headers=get_random_headers()) as response:
                if response.status == 200:
                    logger.info(f"[Process {os.getpid()}] Warm-up request successful")
                    await asyncio.sleep(3)
                else:
                    logger.warning(f"[Process {os.getpid()}] Warm-up request failed: HTTP {response.status}")
        except Exception as e:
            logger.error(f"[Process {os.getpid()}] Error during warm-up request: {str(e)}")
        
        total_papers = len(paper_ids)
        logger.info(f"[Process {os.getpid()}] Processing {total_papers} papers")
        
        # Process papers in smaller batches to avoid overloading the server
        small_batch_size = min(10, total_papers)
        all_results = []
        
        for i in range(0, total_papers, small_batch_size):
            small_batch = paper_ids[i:i+small_batch_size]
            logger.info(f"[Process {os.getpid()}] Processing small batch {i//small_batch_size + 1}/{(total_papers-1)//small_batch_size + 1} ({len(small_batch)} papers)")
            
            tasks = [
                process_paper(session, paper_id, yearmonth, dl_semaphore, img_semaphore)
                for paper_id in small_batch
            ]
            
            batch_results = []
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[Proc {os.getpid()}] Batch {i//small_batch_size + 1}"):
                try:
                    result = await task
                    batch_results.append(result)
                    all_results.append(result)
                except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
                    # Log the error and add a failed result
                    logger.error(f"Connection error: {str(e)}")
                    # Add failed result if needed
                
            # Take a break between small batches
            if i + small_batch_size < total_papers:
                batch_break = 5 + random.uniform(2, 8)
                logger.info(f"[Process {os.getpid()}] Taking a break of {batch_break:.2f}s between small batches")
                await asyncio.sleep(batch_break)
    
    # Return results
    return all_results

def process_batch(yearmonth, paper_ids, concurrent_limit=5):
    """
    Process a batch of paper IDs (wrapper for running in a process).
    
    Args:
        yearmonth (str): Year-month in format YYMM
        paper_ids (list): List of paper IDs to process
        concurrent_limit (int): Concurrent download limit
    
    Returns:
        list: Results of processing
    """
    # Configure process-specific logger
    logger = setup_logger(f"process_{os.getpid()}")
    logger.info(f"Starting process {os.getpid()} with {len(paper_ids)} papers")
    
    # Run the async function
    try:
        return asyncio.run(process_batch_async(yearmonth, paper_ids, concurrent_limit))
    except Exception as e:
        logger.error(f"Error in process {os.getpid()}: {str(e)}")
        return []

def main():
    args = parse_args()
    
    # Validate arguments
    if not re.match(r'^\d{4}$', args.yearmonth):
        logger.error("Year-month must be in format YYMM (e.g., 2401 for January 2024)")
        return 1
    
    if args.start_id <= 0 or args.end_id <= 0:
        logger.error("Start ID and End ID must be positive integers")
        return 1
    
    if args.end_id < args.start_id:
        logger.error("End ID must be greater than or equal to Start ID")
        return 1
    
    # Create necessary directories
    os.makedirs(RAW_JSON_DIR, exist_ok=True)
    os.makedirs(RAW_HTML_DIR, exist_ok=True)
    os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
    
    # Generate list of paper IDs
    paper_ids = [f"{args.yearmonth}.{i:05d}" for i in range(args.start_id, args.end_id + 1)]
    total_papers = len(paper_ids)
    
    # Use multi-processing if specified and we have enough papers
    if args.processes > 1 and total_papers >= args.processes:
        logger.info(f"Using {args.processes} processes for {total_papers} papers")
        
        # Split paper IDs into batches for each process
        process_batches = []
        papers_per_process = total_papers // args.processes
        remainder = total_papers % args.processes
        
        start_idx = 0
        for i in range(args.processes):
            # Distribute the remainder papers across processes
            batch_size = papers_per_process + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size
            
            process_batches.append(paper_ids[start_idx:end_idx])
            start_idx = end_idx
        
        # Create a process pool and distribute the work
        with multiprocessing.Pool(processes=args.processes) as pool:
            # Prepare the function with partial application
            process_func = partial(process_batch, args.yearmonth, concurrent_limit=args.concurrent)
            
            # Run the processes and get results
            all_results = []
            for batch_results in pool.imap_unordered(process_func, process_batches):
                all_results.extend(batch_results)
        
        # Summarize results
        successful_papers = sum(1 for r in all_results if r.get("success", False))
        total_images = sum(len(r.get("image_results", [])) for r in all_results)
        successful_images = sum(sum(1 for img in r.get("image_results", []) if img[1]) for r in all_results)
        
        logger.info(f"Downloaded HTML for {successful_papers}/{total_papers} papers")
        logger.info(f"Downloaded {successful_images}/{total_images} images")
        
    else:
        # Run in a single process (uses async for concurrency)
        if args.processes > 1:
            logger.info(f"Not enough papers ({total_papers}) to use {args.processes} processes, using single process")
        else:
            logger.info(f"Using single process for {total_papers} papers")
        
        # Run the async function directly
        asyncio.run(process_batch_async(args.yearmonth, paper_ids, args.concurrent))
    
    return 0

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows compatibility
    sys.exit(main())