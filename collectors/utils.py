import os
import re
import time
import random
import urllib.parse
import hashlib
import requests
from datetime import datetime, timedelta
from functools import wraps
from bs4 import BeautifulSoup

# Import Levenshtein library for calculating edit distance
try:
    import Levenshtein
    has_levenshtein = True
except ImportError:
    has_levenshtein = False
    print("Hint: python-Levenshtein library not installed, using simplified string comparison method")

def get_current_timestamp():
    """Gets the current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def scrape_safely(func):
    """Crawler safety decorator, handles exceptions and limits run frequency"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            return []
    return wrapper

def is_article_url(url):
    """Determines if a URL is a specific article link, excluding homepages and list pages"""
    if not url or url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.css', '.js')):
        return False
        
    # Exclude some common non-article paths
    exclude_patterns = ['/search/', '/tag/', '/author/', '/about/', '/contact/', 
                        '/privacy/', '/terms/', '/help/', '/sitemap/', '/404/']
    if any(pattern in url for pattern in exclude_patterns):
        return False
    
    # Check for possible article formats
    # 1. Contains date format /2023/01/15/
    if re.search(r'/\d{4}/\d{1,2}/\d{1,2}/', url):
        return True
        
    # 2. Common article paths
    article_patterns = ['/article/', '/story/', '/news/', '/post/', 
                            '/feature/', '/video/', '/content/', '/watch/']
    if any(pattern in url for pattern in article_patterns):
        return True
        
    # 3. Article formats for specific common news websites
    if ('cnn.com' in url and 'index.html' in url) or \
       ('bbc.com' in url and re.search(r'/news/[a-z]+-\d+$', url)) or \
       ('forbes.com' in url and '/sites/' in url):
        return True
        
    # Complexity check, too short URLs are usually homepages or category pages
    path_parts = urllib.parse.urlsplit(url).path.strip('/').split('/')
    if len(path_parts) >= 2 and any(len(part) > 5 for part in path_parts):
        return True
        
    return False

def is_recent_article(article_url, article_soup):
    """Determines if the article was recently published"""
    # Try to extract date from URL
    date_match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', article_url)
    if date_match:
        year, month, day = map(int, date_match.groups())
        if year == datetime.now().year:
            return True
            
    # Look for date information in page elements
    date_indicators = []
    
    # Find common date elements
    date_elements = article_soup.select('time, .date, .timestamp, [datetime], [pubdate]')
    for elem in date_elements:
        date_text = elem.text.strip().lower() if elem.text else ""
        date_attr = elem.get('datetime', '') or elem.get('content', '')
        
        if date_text:
            date_indicators.append(date_text)
        if date_attr:
            date_indicators.append(date_attr)
            
    # Check for "today", "yesterday", "hours ago" etc. indicators
    recent_patterns = ['today', 'yesterday', 'hour ago', 'hours ago', 
                       'minute ago', 'minutes ago', datetime.now().strftime('%Y')]
    
    for indicator in date_indicators:
        if any(pattern in indicator.lower() for pattern in recent_patterns):
            return True
            
    # If unable to determine, default to recent article
    return True

def safe_select(soup, selector):
    """Safely uses CSS selector, avoids exceptions caused by syntax errors"""
    try:
        return soup.select(selector)
    except Exception:
        return []  # Return empty list if selector has syntax error

def safe_select_one(soup, selector):
    """Safely uses CSS selector to get a single element, avoids exceptions caused by syntax errors"""
    try:
        return soup.select_one(selector)
    except Exception:
        return None  # Return None if selector has syntax error

def enhance_image_detection(soup, url):
    """Enhanced image detection method, checks if page contains content images"""
    # Check Open Graph image tags
    og_image = soup.find('meta', property='og:image') or soup.find('meta', attrs={'name': 'og:image'})
    if og_image and og_image.get('content'):
        return True
        
    # Check Twitter Card image tags
    twitter_image = soup.find('meta', property='twitter:image') or soup.find('meta', attrs={'name': 'twitter:image'})
    if twitter_image and twitter_image.get('content'):
        return True
        
    # Check regular image tags
    images = soup.select('article img, .article-content img, .entry-content img, .post-content img')
    for img in images:
        src = img.get('src', '')
        if src and not ('icon' in src.lower() or 'logo' in src.lower() or 'avatar' in src.lower()):
            # Exclude small images
            width = img.get('width', '0')
            height = img.get('height', '0')
            
            try:
                w = int(width) if width and width.isdigit() else 0
                h = int(height) if height and height.isdigit() else 0
                
                if w > 100 or h > 100:  # Only consider sufficiently large images
                    return True
            except ValueError:
                # If dimensions cannot be obtained, assume there is an image by default
                return True
                
    # Look for any non-small-icon images on the webpage
    all_imgs = soup.find_all('img')
    for img in all_imgs:
        src = img.get('src', '')
        if not src or src.startswith('data:'):
            continue
            
        # Exclude common small icons and decorative images
        if 'icon' in src.lower() or 'logo' in src.lower() or 'avatar' in src.lower() or 'button' in src.lower():
            continue
            
        # Check image dimensions
        width = img.get('width', '0')
        height = img.get('height', '0')
        
        try:
            w = int(width) if width and width.isdigit() else 0
            h = int(height) if height and height.isdigit() else 0
            
            if w > 200 or h > 150:  # More lenient size requirement
                return True
        except ValueError:
            pass
            
    # No sufficiently large images found
    return False

def get_random_headers(user_agents):
    """Gets request headers with a random user agent"""
    user_agent = random.choice(user_agents) if user_agents else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    return {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

def get_title_keywords(title, stopwords):
    """Extracts keywords from a title"""
    if not title:
        return set()
        
    # Clean and extract keywords
    clean_title = re.sub(r'[^\w\s]', '', title.lower())
    words = clean_title.split()
    
    # Keywords after filtering stopwords
    return set(word for word in words if word not in stopwords and len(word) > 2)

def calculate_title_similarity(title1, title2):
    """Calculates the similarity between two titles"""
    # Use edit distance or word set similarity
    if has_levenshtein:
        distance = Levenshtein.distance(title1.lower(), title2.lower())
        max_len = max(len(title1), len(title2))
        return 1 - (distance / max_len) if max_len > 0 else 0
    else:
        # Word set similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        if not words1 or not words2:
            return 0
        return len(words1.intersection(words2)) / len(words1.union(words2))

def download_image(img_url, save_dir, source_name=None):
    """
    Downloads an image and saves it to the specified directory
    
    Args:
        img_url: URL of the image
        save_dir: Directory to save to
        source_name: Name of the source, used for generating filename
        
    Returns:
        str: Path to the saved image, None if failed
    """
    try:
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate unique filename
        img_hash = hashlib.md5(img_url.encode()).hexdigest()
        img_ext = os.path.splitext(urllib.parse.urlparse(img_url).path)[1]
        if not img_ext or len(img_ext) > 5:  # If extension is abnormal, use default extension
            img_ext = '.jpg'
        
        # Add source prefix
        prefix = f"{source_name}_" if source_name else ""
        filename = f"{prefix}{img_hash}{img_ext}"
        filepath = os.path.join(save_dir, filename)
        
        # If file already exists, return path directly
        if os.path.exists(filepath):
            return filepath
        
        # Add random request headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': urllib.parse.urljoin(img_url, '/'),
            'DNT': '1',
        }
        
        # Download image
        response = requests.get(img_url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()
        
        # Save image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Successfully downloaded image: {img_url} -> {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to download image {img_url}: {e}")
        return None

def extract_article_text(soup):
    """
    Extracts full article text from a webpage
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        str: Extracted article content
    """
    # Common article content selectors
    content_selectors = [
        'article p', '.article-body p', '.Article p', '.story-body p',
        'article .content p', '.article__body p', '.article-content p',
        '.story-content p', '.entry-content p', '.post-content p',
        '.main-content p', '.story__body p', '.news-article-content p',
        '.body-content p', '.article__content p', '.content-body p',
        '.RichTextArticleBody p', '.article-container p', '.story__content p',
        '.body-copy p', '.article-text p', '.article_content p',
        '.wire-story-body p', '.editorial p', '.article-main p',
        '.article-wrapper p', '.text-wrapper p', '.post__body p',
        '.article-wrapper article p', '.article-text article p'
    ]
    
    # Try different selectors to extract content
    paragraphs = []
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            paragraphs = [p.get_text().strip() for p in elements if p.get_text().strip()]
            if len(''.join(paragraphs)) > 100:  # If enough content, use current selector
                break
    
    # If specific selectors didn't find content, try a more general method
    if not paragraphs:
        # Try to find all paragraphs, but exclude navigation, footer, etc. areas
        exclude_selectors = [
            'header', 'footer', 'nav', '.nav', '.navigation', '.menu',
            '.sidebar', '.related', '.comments', '.share', '.social',
            '.advertisement', '.ads', '.ad-container', '#comments',
            '.newsletter', '.subscribe', '.subscription'
        ]
        
        # Create exclusion selectors
        exclude_elements = []
        for selector in exclude_selectors:
            exclude_elements.extend(soup.select(selector))
        
        # Get all paragraphs, excluding those within excluded elements
        all_p = soup.find_all('p')
        for p in all_p:
            # Check if this paragraph is within any excluded element
            is_excluded = False
            for exclude_elem in exclude_elements:
                if p in exclude_elem.find_all('p'):
                    is_excluded = True
                    break
            
            # If not in exclusion list and content is not empty, add to results
            if not is_excluded and p.get_text().strip():
                paragraphs.append(p.get_text().strip())
    
    # Clean text, remove extra whitespace
    cleaned_paragraphs = []
    for p in paragraphs:
        text = p.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace characters with a single space
        if text:
            cleaned_paragraphs.append(text)
    
    # Combine all paragraphs
    article_text = '\n\n'.join(cleaned_paragraphs)
    
    # If content is too short, try to extract general content
    if len(article_text) < 100:
        # Try to extract text within article or main tags
        for container in soup.select('article, main, .article, .main-content, .story, .post'):
            text = container.get_text().strip()
            text = re.sub(r'\s+', ' ', text)
            if len(text) > len(article_text):
                article_text = text
    
    return article_text

def extract_best_image(soup, url):
    """
    Extracts the best image URL from an article
    
    Args:
        soup: BeautifulSoup object
        url: Article URL, used for converting relative paths
        
    Returns:
        str: Best image URL, None if not found
    """
    # 1. Try to get Open Graph image (usually high-quality images for social media sharing)
    og_image = soup.find('meta', property='og:image')
    if og_image and og_image.get('content'):
        return urllib.parse.urljoin(url, og_image.get('content'))
    
    # 2. Try to get Twitter Card image
    twitter_image = soup.find('meta', {'name': 'twitter:image'}) or soup.find('meta', property='twitter:image')
    if twitter_image and twitter_image.get('content'):
        return urllib.parse.urljoin(url, twitter_image.get('content'))
    
    # 3. Try to get main article image
    image_selectors = [
        'article .image img', '.article-image img', '.article__image img',
        '.main-image img', '.featured-image img', '.article img:first-of-type',
        '.article-featured-image img', '.entry-featured-image img',
        'article header img', '.post-thumbnail img', '.article-header img',
        '.main-content img:first-of-type', '.hero-image img',
        '.article-hero-image img', '.lead-image img', '.article-head img',
        '.article-lead-image img', '.story-image img', '.responsive-image img'
    ]
    
    for selector in image_selectors:
        img = soup.select_one(selector)
        if img and img.get('src'):
            # Return the first image found
            return urllib.parse.urljoin(url, img.get('src'))
    
    # 4. Look for any sufficiently large image
    all_images = soup.find_all('img')
    for img in all_images:
        src = img.get('src', '')
        if not src or src.startswith('data:'):
            continue
            
        # Exclude small icons or buttons
        width = img.get('width', '0')
        height = img.get('height', '0')
        
        try:
            # Try to convert width and height to numbers for comparison
            w = int(width) if width and width.isdigit() else 0
            h = int(height) if height and height.isdigit() else 0
            
            if w >= 300 or h >= 200:  # Assume larger images are content images
                return urllib.parse.urljoin(url, src)
        except ValueError:
            pass
            
    # 5. If no sufficiently large image found, return the first non-icon image
    for img in all_images:
        src = img.get('src', '')
        # Exclude some obvious small icons
        if not src or src.startswith('data:') or 'icon' in src.lower() or 'logo' in src.lower():
            continue
            
        return urllib.parse.urljoin(url, src)
    
    # No valid image found
    return None
