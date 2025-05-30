import re
import time
import random
import requests
import json
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
from collectors.config import FORBES_SECTIONS, MAX_ARTICLES_PER_SOURCE

def is_valid_forbes_url(url):
    """Checks if the URL is a valid Forbes article link format"""
    if not url or 'forbes.com' not in url:
        return False
    
    exclude_patterns = [
        '/signup/', '/login/', '/subscription/', '/about/', '/tags/', '/topics/', 
        '/search/', '/video/', '/videos/', '/newsletters/', '/lists/index',
        '/marketplace/', '/advisor/', '/sites/advice/', '/sites/advisorindex/',
        '/tableau/', '/offers/', '/forbes-live/', '/podcasts/', '/sites/forbespr/',
        '/dashboard/', '/connect/', '/profile/', '/privacy/', '/terms/', 
        '/follow/', '/leadership/'
    ]
    
    if any(pattern in url.lower() for pattern in exclude_patterns):
        return False
    
    if url.strip('/').endswith('forbes.com'):
        return False
    
    category_index_patterns = [
        r'forbes\.com/[a-z-]+/?$',  
    ]
    
    for pattern in category_index_patterns:
        if re.match(pattern, url.lower()):
            if '/sites/' not in url and not re.search(r'/\d{4}/', url):
                return False
    
    if re.search(r'forbes\.com/sites/[^/]+/\d{4}/\d{1,2}/\d{1,2}/[^/]+', url):
        return True
        
    if re.search(r'forbes\.com/sites/[^/]+/[^/]+/?$', url) and '/sites/' in url:
        return True
    
    if '/sites/' in url and url.count('/') >= 5:
        return True
        
    if re.search(r'/page/\d+/?$', url):
        return False
    
    return False

def extract_forbes_links(soup, base_url):
    """Extracts article links from Forbes pages"""
    article_links = set()
    
    article_selectors = [
        'article a', '.stream-item a', '.card a', '.feature__title a',
        'a.stream-item__title', 'h3 a', 'h2 a',
        'a.card__title-link', 'a[href*="/sites/"]',
        '.fbs-video__info a',  
        '.channel-river__item a', '.hero__hed a',
        'a.happening__title', '.editor-picks__title a',
        'a.forbes-river__card__title-link',
        'a[href*="/2025/"]', 'a[href*="/2024/"]', 'a[href*="/2023/"]', # Specific year patterns
        '.fbs-cnct__figure a', '.stream-item__image-link',
        '.headline a', '.title a', '.article-title a'
    ]
    
    for selector in article_selectors:
        try:
            for link in soup.select(selector):
                href = link.get('href', '')
                if not href or href.startswith('#') or 'javascript:' in href:
                    continue
                    
                full_url = href
                if not href.startswith('http'):
                    full_url = urllib.parse.urljoin(base_url, href)
                
                if '?' in full_url:
                    full_url = full_url.split('?')[0]
                
                if 'forbes.com' in full_url:
                    article_links.add(full_url)
        except Exception:
            continue
    
    if len(article_links) < 15:  
        print(f"Too few valid links ({len(article_links)}), attempting to expand scraping methods...")
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if not href or href.startswith('#') or 'javascript:' in href:
                continue
                
            full_url = href
            if not href.startswith('http'):
                full_url = urllib.parse.urljoin(base_url, href)
            
            if '?' in full_url:
                full_url = full_url.split('?')[0]
            
            if 'forbes.com' in full_url and ('/sites/' in full_url or re.search(r'/\d{4}/\d{2}/', full_url)):
                article_links.add(full_url)
    
    valid_links = []
    for link in article_links:
        if is_valid_forbes_url(link):
            valid_links.append(link)
    
    if len(valid_links) < 5:
        print(f"Too few valid links ({len(valid_links)}), attempting to broaden capture methods...")
        for link in article_links:
            if 'forbes.com' in link and link not in valid_links:
                if '/sites/' in link and len(link.split('/')) >= 6:
                    valid_links.append(link)
    
    print(f"Initial link set: {len(article_links)}, valid article links: {len(valid_links)}")
    return valid_links

def is_today_article(article_url, article_soup):
    """Checks if the Forbes article was published today"""
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    date_match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', article_url)
    if date_match:
        year, month, day = map(int, date_match.groups())
        article_date = f"{year}-{month:02d}-{day:02d}"
        if article_date == today or article_date == yesterday:
            return True
    
    date_elements = article_soup.select('time, .article-timestamp, [itemprop="datePublished"], [itemprop="dateModified"]')
    for date_elem in date_elements:
        date_text = date_elem.text.strip().lower()
        date_attr = date_elem.get('datetime', '') or date_elem.get('content', '')
        
        today_indicators = ['today', 'hours ago', 'hour ago', 'minutes ago', 'minute ago', 'just now']
        if any(indicator in date_text for indicator in today_indicators):
            return True
            
        if today in date_attr or yesterday in date_attr:
            return True
    
    og_date = article_soup.find('meta', property='article:published_time')
    if og_date and og_date.get('content'):
        content = og_date.get('content')
        if today in content or yesterday in content:
            return True
    
    return False

@scrape_safely
def scrape_forbes_news(user_agents, topics_collector=None):
    """Scrapes news from various sections of Forbes
    
    Args:
        user_agents: List of user agents
        topics_collector: Hot topics collector instance, for real-time deduplication
    """
    all_topics = []
    processed_sections = 0
    
    print(f"Forbes: Preparing to scrape {len(FORBES_SECTIONS)} pages")
    
    for section in FORBES_SECTIONS:
        url = section['url']
        source = section['source']
        category = section['category']
        
        processed_sections += 1
        print(f"[{processed_sections}/{len(FORBES_SECTIONS)}] Scraping {source} - {url}")
        
        try:
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
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            })
            
            time.sleep(random.uniform(1.0, 3.0))
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"Access failed {url}: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_links = extract_forbes_links(soup, url)
            print(f"Found {len(article_links)} potential article links in {source}")
            
            random.shuffle(article_links)
            
            processed = 0
            for article_url in article_links:
                if processed >= MAX_ARTICLES_PER_SOURCE:
                    break
                
                if not is_valid_forbes_url(article_url):
                    print(f"Skipping non-article link: {article_url}")
                    continue
                
                try:
                    time.sleep(random.uniform(1.0, 2.0))
                    
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
                    
                    article_res = requests.get(article_url, headers=article_headers, timeout=20)
                    
                    if article_res.status_code == 404:
                        print(f"Article not found (404): {article_url}")
                        continue
                    elif article_res.status_code in [301, 302, 307, 308]:
                        print(f"Article redirected: {article_url}")
                        continue
                    elif article_res.status_code != 200:
                        print(f"Failed to access article: {article_res.status_code} - {article_url}")
                        continue
                        
                    article_soup = BeautifulSoup(article_res.content, 'html.parser')
                    
                    title_selectors = [
                        'h1.article-headline', 'h1.fs-headline', 'h1.speakable-headline',
                        'h1.heading--tape', 'h1[data-ga-track="Title"]', 'h1.entry-title',
                        'article h1', '.article-header h1', '.article-title',
                        'h1.article-title', 'h1'
                    ]
                    
                    title_elem = None
                    for selector in title_selectors:
                        title_elem = article_soup.select_one(selector)
                        if title_elem and len(title_elem.text.strip()) > 5:
                            break
                    
                    if not title_elem:
                        meta_title = article_soup.find('meta', property='og:title')
                        if meta_title and meta_title.get('content'):
                            title = meta_title.get('content').strip()
                        else:
                            title_tag = article_soup.find('title')
                            if title_tag and title_tag.text:
                                title_text = title_tag.text.strip()
                                if ' | ' in title_text:
                                    title = title_text.split(' | ')[0].strip()
                                else:
                                    title = title_text
                            else:
                                print(f"Unable to extract title: {article_url}")
                                continue
                    else:
                        title = title_elem.text.strip()
                    
                    if len(title) < 20:
                        print(f"Title too short: {title}")
                        continue
                    
                    article_text = extract_article_text(article_soup)
                    
                    img_urls, img_captions = extract_article_images(article_soup, article_url)
                    
                    if img_urls:
                        article_text = insert_image_tags(article_text, len(img_urls))
                    
                    has_image = len(img_urls) > 0
                    
                    is_recent = is_today_article(article_url, article_soup)
                    
                    curr_category = category
                    
                    if topics_collector:
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
                        
                        if topics_collector.add_topic_realtime(article_data):
                            processed += 1
                            print(f"Fetched {source} article ({processed}/{MAX_ARTICLES_PER_SOURCE}): {title[:40]}... [Images:{len(img_urls)}]")
                    else:
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
            
            time.sleep(random.uniform(2.0, 4.0))
            
        except Exception as e:
            print(f"Failed to scrape {source}: {str(e)}")
    
    if topics_collector:
        valid_articles = topics_collector.all_topics
        print(f"Forbes fetched a total of {len(valid_articles)} valid articles")
        return valid_articles
    else:
        all_topics.sort(key=lambda x: (0 if x.get('is_recent') else 1, 0 if x.get('has_image') else 1))
        valid_articles = [article for article in all_topics if is_article_url(article.get('url', ''))]
        print(f"Forbes fetched a total of {len(valid_articles)} valid articles")
        return valid_articles
