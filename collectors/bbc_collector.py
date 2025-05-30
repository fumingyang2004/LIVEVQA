import re
import time
import random
import requests
import urllib.parse
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from collectors.utils import (
    scrape_safely, is_article_url, get_random_headers,
    extract_article_text
)
from collectors.utils_images import (
    extract_article_images, filter_images, insert_image_tags
)
from collectors.config import BBC_SECTIONS, MAX_ARTICLES_PER_SOURCE

def is_valid_bbc_article_url(url):
    """Checks if the URL is a valid BBC article link format"""
    if not url:
        return False
    
    # Exclude video, tag, and other non-article pages
    exclude_patterns = [
        '/videos/', '/video/', '/tags/', '/tag/', '/topics/', '/topic/',
        '/search/', '/login', '/account/', '/iplayer/', '/tv/', '/radio/',
        '/weather/', '/contact/', '/about/', '/help/'
    ]
    
    if any(pattern in url.lower() for pattern in exclude_patterns):
        return False
    
    # Identify BBC news article URL patterns
    # 1. Standard news format: /news/article-id
    if re.search(r'/news/[a-z]+-\d+$', url):
        return True
        
    # 2. Article pages containing "articles"
    if '/articles/' in url:
        return True
        
    # 3. Sports news format
    if '/sport/' in url and re.search(r'/[a-z]+-\d+$', url):
        return True
    
    # 4. New format articles with subdirectories and numeric IDs
    if re.search(r'/news/[a-zA-Z-]+-\d+$', url) or re.search(r'/sport/[a-zA-Z-]+-\d+$', url):
        return True
        
    # 5. Older special section articles
    if any(path in url for path in ['/culture/', '/travel/', '/future/']) and re.search(r'/[a-z]+-\d+$', url):
        return True
    
    # Exclude if none of the common article formats match
    return False

def extract_bbc_links(soup, base_url):
    """Extracts article links from BBC pages"""
    article_links = set()
    
    # Common article link selectors
    article_selectors = [
        'a.gs-c-promo-heading', 'a.title-link',
        '.media__link', '.media__content a',
        '.lakefront__link', '.eagle-item__link',
        '.most-popular-list__link', '.featured-post-link',
        '.top-stories a', '.story-topper a',
        'h2 a', 'h3 a', '.article_link a',
        'a[href*="/news/"]', 'a[href*="/sport/"]',
        'a[href*="/culture/"]', 'a[href*="/future/"]',
        'a[href*="/travel/"]', 'a[href*="/articles/"]'
    ]
    
    # Use selectors to find article links
    for selector in article_selectors:
        try:
            for link in soup.select(selector):
                href = link.get('href', '')
                if not href or href.startswith('#'):
                    continue
                
                # Ensure it's a full URL
                full_url = href
                if not href.startswith('http'):
                    full_url = urllib.parse.urljoin(base_url, href)
                
                # Ensure it's a BBC domain
                if 'bbc.com' in full_url or 'bbc.co.uk' in full_url:
                    article_links.add(full_url)
        except Exception:
            continue
    
    # If not enough links found via selectors, use a more general method
    if len(article_links) < 10:
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if not href or href.startswith('#'):
                continue
                
            full_url = href
            if not href.startswith('http'):
                full_url = urllib.parse.urljoin(base_url, href)
                
            if ('bbc.com' in full_url or 'bbc.co.uk' in full_url) and any(pattern in full_url for pattern in ['/news/', '/sport/', '/culture/', '/travel/', '/future/']):
                article_links.add(full_url)
    
    # Filter valid article links
    valid_links = [link for link in article_links if is_valid_bbc_article_url(link)]
    print(f"Initial link set: {len(article_links)}, valid article links: {len(valid_links)}")
    
    return valid_links

def is_today_article(article_url, article_soup):
    """Checks if the article was published today"""
    # Check URL date
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Look for date information in the page
    date_elements = article_soup.select('time, .date-time, .date, [data-datetime]')
    for date_elem in date_elements:
        # Extract date information from tag attributes
        date_text = date_elem.text.strip()
        date_attr = date_elem.get('datetime', '') or date_elem.get('data-datetime', '')
        
        # Check if text contains "today", "hours ago" etc. indicators
        if any(marker in date_text.lower() for marker in ['today', 'hours ago', 'hour ago', 'minutes ago', 'minute ago']):
            return True
            
        # Check if attribute contains today's date
        if today in date_attr or yesterday in date_attr:
            return True
    
    # If no explicit "today" indicator, default to false
    return False

def get_bbc_page_urls(base_url, pages=2):
    """Generates BBC pagination URLs"""
    urls = [base_url]
    
    # Most BBC pages don't use pagination, only a few use the page query parameter
    if any(section in base_url for section in ['/most-read/', '/most-popular/']):
        for i in range(2, pages + 1):
            if '?' in base_url:
                urls.append(f"{base_url}&page={i}")
            else:
                urls.append(f"{base_url}?page={i}")
    
    # Add subcategory to extend crawling scope
    if '/news/' in base_url:
        if base_url.endswith('/news'):
            # For main news page, add news from specific regions
            regions = ['world', 'uk', 'europe']
            for region in regions[:1]:  # Limit to adding only 'world'
                urls.append(f"{base_url}/{region}")
                
    elif '/sport/' in base_url:
        if base_url.endswith('/sport'):
            # For main sports page, add specific sports
            sports = ['football', 'cricket', 'tennis']
            for sport in sports[:1]:  # Limit to adding only 'football'
                urls.append(f"{base_url}/{sport}")
    
    return urls

@scrape_safely
def scrape_bbc_news(user_agents, topics_collector=None):
    """Scrapes various sections of BBC News
    
    Args:
        user_agents: List of user agents
        topics_collector: Hot topics collector instance, for real-time deduplication
    """
    all_topics = []
    processed_sections = 0
    
    # Generate multiple URLs for each category to expand crawling scope
    all_section_urls = []
    for section in BBC_SECTIONS:
        page_urls = get_bbc_page_urls(section['url'], pages=2)
        for page_url in page_urls:
            all_section_urls.append({
                'url': page_url,
                'source': section['source'],
                'category': section['category']
            })
    
    print(f"BBC: Preparing to scrape {len(all_section_urls)} pages")
    
    for section in all_section_urls:
        url = section['url']
        source = section['source']
        category = section['category']
        
        processed_sections += 1
        print(f"[{processed_sections}/{len(all_section_urls)}] Scraping {source} - {url}")
        
        try:
            # Get page content
            headers = get_random_headers(user_agents)
            response = requests.get(url, headers=headers, timeout=20)
            
            if response.status_code != 200:
                print(f"Access failed {url}: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links
            article_links = extract_bbc_links(soup, url)
            print(f"Found {len(article_links)} potential article links in {source}")
            
            # Randomly shuffle link order to increase diversity
            random.shuffle(article_links)
            
            # Process each article
            processed = 0
            for article_url in article_links:
                if processed >= MAX_ARTICLES_PER_SOURCE:
                    break
                    
                try:
                    # Random delay to avoid too frequent requests
                    time.sleep(random.uniform(0.5, 1.0))
                    
                    article_headers = get_random_headers(user_agents)
                    article_res = requests.get(article_url, headers=article_headers, timeout=15)
                    
                    if article_res.status_code != 200:
                        print(f"Failed to access article: {article_res.status_code} - {article_url}")
                        continue
                        
                    article_soup = BeautifulSoup(article_res.content, 'html.parser')
                    
                    # Extract title
                    title_selectors = [
                        'h1', 'h1.story-headline', 'h1.article-headline',
                        'h1.vxp-media__headline', 'h1.lx-stream-page__header-text',
                        '.article-headline', '.story-body__h1'
                    ]
                    
                    title_elem = None
                    for selector in title_selectors:
                        title_elem = article_soup.select_one(selector)
                        if title_elem and len(title_elem.text.strip()) > 5:
                            break
                            
                    # If title not found via selectors, try meta tags
                    if not title_elem:
                        meta_title = article_soup.find('meta', property='og:title')
                        if meta_title and meta_title.get('content'):
                            title = meta_title.get('content').strip()
                        else:
                            continue
                    else:
                        title = title_elem.text.strip()
                    
                    # Skip titles that are too short or meaningless
                    if len(title) < 20 or title in ["Headlines", "BBC News", "BBC Sport"]:
                        continue
                    
                    # Extract article content
                    article_text = extract_article_text(article_soup)
                    
                    # Extract all images and captions - new multi-image processing
                    img_urls, img_captions = extract_article_images(article_soup, article_url)
                    
                    # Insert image tags into text
                    if img_urls:
                        article_text = insert_image_tags(article_text, len(img_urls))
                    
                    has_image = len(img_urls) > 0
                    
                    # Assume all BBC articles are recent by default
                    is_recent = is_today_article(article_url, article_soup)
                    
                    # Use current category
                    curr_category = category
                    
                    # If collector instance is provided, use real-time deduplication
                    if topics_collector:
                        # Build article object - new format
                        article_data = {
                            'topic': title,
                            'text': article_text,
                            'img_urls': img_urls,
                            'captions': img_captions,
                            'source': source,
                            'category': curr_category,
                            'timestamp': datetime.now().isoformat(),
                            'url': article_url,
                            'has_image': has_image,
                            'is_recent': is_recent
                        }
                        
                        # Use real-time deduplication to add article
                        if topics_collector.add_topic_realtime(article_data):
                            processed += 1
                            print(f"Fetched {source} article ({processed}/{MAX_ARTICLES_PER_SOURCE}): {title[:40]}... [Images:{len(img_urls)}]")
                        # Otherwise skip this article
                    else:
                        # Add using new format
                        all_topics.append({
                            'topic': title,
                            'text': article_text,
                            'img_urls': img_urls,
                            'captions': img_captions,
                            'source': source,
                            'category': curr_category,
                            'timestamp': datetime.now().isoformat(),
                            'url': article_url,
                            'has_image': has_image,
                            'is_recent': is_recent
                        })
                        processed += 1
                        print(f"Fetched {source} article ({processed}/{MAX_ARTICLES_PER_SOURCE}): {title[:40]}... [Images:{len(img_urls)}]")
                        
                except Exception as e:
                    print(f"Error processing article: {article_url} - {str(e)}")
                    
            print(f"Fetched {processed} articles from {source}")
            
            # Delay between pages to avoid being blocked
            time.sleep(random.uniform(1.0, 2.0))
            
        except Exception as e:
            print(f"Failed to scrape {source}: {str(e)}")
    
    # When using real-time deduplication, return topics directly from the collector
    if topics_collector:
        valid_articles = topics_collector.all_topics
        print(f"BBC fetched a total of {len(valid_articles)} valid articles")
        return valid_articles
    else:
        # Original filtering logic
        all_topics.sort(key=lambda x: (0 if x.get('is_recent') else 1, 0 if x.get('has_image') else 1))
        valid_articles = [article for article in all_topics if is_article_url(article.get('url', ''))]
        print(f"BBC fetched a total of {len(valid_articles)} valid articles")
        return valid_articles
