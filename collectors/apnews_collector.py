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
from collectors.config import APNEWS_SECTIONS, MAX_ARTICLES_PER_SOURCE

def is_valid_apnews_url(url):
    """Checks if the URL is a valid AP News article link format"""
    if not url or 'apnews.com' not in url:
        return False
    
    # Exclude non-article pages
    exclude_patterns = [
        '/about/', '/contact/', '/tag/', '/author/', 
        '/search/', '/video/', '/videos/', '/gallery/',
        '/privacy/', '/terms/', '/hub/', '/latest-news/',
        '/top-news/', '/photos/'
    ]
    
    if any(pattern in url.lower() for pattern in exclude_patterns):
        return False
    
    # AP News article URLs typically contain /article/ path and a hash ID
    # Example: https://apnews.com/article/nasa-stuck-astronauts-butch-suni-834c3dd3bd50dc6be8e9ed1812ac7839
    if '/article/' in url and url.count('/') >= 3 and re.search(r'[a-f0-9]{20,}$', url):
        return True
        
    # Older articles might use a different format but still contain /article/
    if '/article/' in url and url.count('/') >= 3:
        return True
    
    return False

def extract_apnews_links(soup, base_url):
    """Extracts article links from AP News page"""
    article_links = set()
    
    # Try various selectors to find article links
    article_selectors = [
        'a.PagePromo-title', 'a.PageList-item-title', 
        'a.Link', '.CardHeadline a', '.FeedCard a',
        '.PagePromo-title a', '.PagePromo a',
        'h2 a', 'h3 a', '.bbtopHeadline a',
        'a[href*="/article/"]',  # Key path matching
        'a[data-key]',  # AP News articles often have data-key attribute
        '.PagePromo-content a', '.Hub-feed a',
        '.TopicsHub-cards a', '.Link--article',
        '.FeedCard-title a', '.StoryBlock a'
    ]
    
    # Try specific selectors
    for selector in article_selectors:
        try:
            for link in soup.select(selector):
                href = link.get('href', '')
                if not href or href.startswith('#') or 'javascript:' in href:
                    continue
                    
                # Ensure it's a full URL
                full_url = href
                if not href.startswith('http'):
                    full_url = urllib.parse.urljoin(base_url, href)
                
                if 'apnews.com' in full_url:
                    article_links.add(full_url)
        except Exception:
            continue
    
    # If specific selectors didn't find enough links, use a general method
    if len(article_links) < 10:
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if not href or href.startswith('#') or 'javascript:' in href:
                continue
                
            full_url = href
            if not href.startswith('http'):
                full_url = urllib.parse.urljoin(base_url, href)
            
            if 'apnews.com' in full_url and '/article/' in full_url:
                article_links.add(full_url)
    
    # Filter valid article links
    valid_links = []
    for link in article_links:
        if is_valid_apnews_url(link):
            valid_links.append(link)
    
    print(f"Initial link set: {len(article_links)}, valid article links: {len(valid_links)}")
    return valid_links

def is_today_article(article_url, article_soup):
    """Checks if the AP News article was published today"""
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Extract date information from the page
    date_elements = article_soup.select('.Timestamp, .timestamp, time, [data-source="Updated"], [data-source="Published"]')
    for date_elem in date_elements:
        date_text = date_elem.text.strip().lower()
        date_attr = date_elem.get('datetime', '')
        
        # Check if text contains "today", "hours ago" etc. indicators
        today_indicators = ['today', 'hours ago', 'hour ago', 'minutes ago', 'minute ago', 'just now']
        if any(indicator in date_text for indicator in today_indicators):
            return True
            
        # Check if attribute contains today's date
        if date_attr and (today in date_attr or yesterday in date_attr):
            return True
    
    # Check metadata
    meta_date = article_soup.find('meta', {'property': 'article:published_time'})
    if meta_date and meta_date.get('content'):
        date_content = meta_date.get('content')
        if today in date_content or yesterday in date_content:
            return True
    
    # If it's an old article but recently updated, it might still be relevant for today
    if article_soup.find(string=re.compile(r'updated\s+today|updated\s+\d+\s+hours?\s+ago', re.I)):
        return True
    
    # Default assumption for recent articles
    return True

@scrape_safely
def scrape_apnews_news(user_agents, topics_collector=None):
    """Scrapes news from various sections of AP News
    
    Args:
        user_agents: List of user agents
        topics_collector: Hot topics collector instance, for real-time deduplication
    """
    all_topics = []
    processed_sections = 0
    
    print(f"AP News: Preparing to scrape {len(APNEWS_SECTIONS)} pages")
    
    for section in APNEWS_SECTIONS:
        url = section['url']
        source = section['source']
        category = section['category']
        
        processed_sections += 1
        print(f"[{processed_sections}/{len(APNEWS_SECTIONS)}] Scraping {source} - {url}")
        
        try:
            # Use random user agent to avoid being blocked
            headers = get_random_headers(user_agents)
            headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Cache-Control': 'max-age=0',
            })
            
            # Add random delay to simulate human behavior
            time.sleep(random.uniform(1.0, 2.0))
            
            response = requests.get(url, headers=headers, timeout=20)
            
            if response.status_code != 200:
                print(f"Access failed {url}: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links
            article_links = extract_apnews_links(soup, url)
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
                    time.sleep(random.uniform(0.5, 1.5))
                    
                    # Use a new random user agent for each article
                    article_headers = get_random_headers(user_agents)
                    article_headers.update({
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Referer': url,
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'same-origin'
                    })
                    
                    article_res = requests.get(article_url, headers=article_headers, timeout=15)
                    
                    if article_res.status_code != 200:
                        print(f"Failed to access article: {article_res.status_code} - {article_url}")
                        continue
                        
                    article_soup = BeautifulSoup(article_res.content, 'html.parser')
                    
                    # Extract title - AP News article titles are usually in h1 tags
                    title_selectors = [
                        'h1.Page-headline', 'h1.article-headline', 'h1[data-key="headline"]',
                        '.Article-headline', '.Article-content h1', 'h1.Headline',
                        'h1'  # Last resort, try any h1 tag
                    ]
                    
                    title_elem = None
                    for selector in title_selectors:
                        title_elem = article_soup.select_one(selector)
                        if title_elem and len(title_elem.text.strip()) > 5:
                            break
                    
                    # If still no title found, try to get <title> tag or metadata
                    if not title_elem:
                        meta_title = article_soup.find('meta', property='og:title')
                        if meta_title and meta_title.get('content'):
                            title = meta_title.get('content').strip()
                        else:
                            # If title detection fails, skip this article
                            print(f"Unable to extract title: {article_url}")
                            continue
                    else:
                        title = title_elem.text.strip()
                    
                    # Title length and quality check
                    if len(title) < 15:
                        print(f"Title too short: {title}")
                        continue
                    
                    # Extract article content
                    article_text = extract_article_text(article_soup)
                    
                    # Extract all images and captions - use multi-image processing function
                    img_urls, img_captions = extract_article_images(article_soup, article_url)
                    
                    # Insert image tags into text
                    if img_urls:
                        article_text = insert_image_tags(article_text, len(img_urls))
                    
                    has_image = len(img_urls) > 0
                    
                    # Check if it's a recent article
                    is_recent = is_today_article(article_url, article_soup)
                    
                    # Use current category
                    curr_category = category
                    
                    # If collector instance is provided, use real-time deduplication
                    if topics_collector:
                        # Build article object - using new format
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
        print(f"AP News fetched a total of {len(valid_articles)} valid articles")
        return valid_articles
    else:
        # Original filtering logic
        all_topics.sort(key=lambda x: (0 if x.get('is_recent') else 1, 0 if x.get('has_image') else 1))
        valid_articles = [article for article in all_topics if is_article_url(article.get('url', ''))]
        print(f"AP News fetched a total of {len(valid_articles)} valid articles")
        return valid_articles
