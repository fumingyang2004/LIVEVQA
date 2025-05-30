import re
import time
import random
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from collectors.utils import (
    scrape_safely, is_article_url, get_random_headers,
    extract_article_text
)
from collectors.utils_images import (
    extract_article_images, filter_images, insert_image_tags
)
from collectors.config import VARIETY_SOURCES, MAX_ARTICLES_PER_SOURCE

@scrape_safely
def scrape_variety_entertainment(user_agents, topics_collector=None):
    """Scrapes entertainment news from the Variety website
    
    Args:
        user_agents: List of user agents
        topics_collector: Hot topics collector instance, for real-time deduplication
    """
    print(f"Fetching latest high-quality entertainment news from Variety...")
    
    # Extend article selectors, add more types of link selectors
    article_selectors = [
        '.c-title a', '.c-entry-box--compact__title a', 'article.c-entry h2 a',
        '.c-entry-box--compact a', 'a[href*="/2024/"]', 'a[href*="/2025/"]',
        '.l-main article a', '.c-entry-box--compact__image-wrapper a',
        # Add more selectors
        'a[href*="/2023/"]', 'a[href*="/2022/"]',  # Add support for earlier years
        '.o-story__block a', '.article-permalink', '.feed__title a',
        '.c-entry-box a', '.article-card a', '.l-homepageLatestNews a'
    ]
    
    all_articles = []
    
    # Extend Variety sources, add more category pages and archive pages
    extended_sources = VARIETY_SOURCES + [
        # Add archive pages by year to increase crawling scope
        {'url': 'https://variety.com/2024/', 'source': 'Variety 2024'},
        {'url': 'https://variety.com/2023/', 'source': 'Variety 2023'},
        {'url': 'https://variety.com/2022/', 'source': 'Variety 2022'},
        # Add more content category pages
        {'url': 'https://variety.com/v/awards/', 'source': 'Variety Awards'},
        {'url': 'https://variety.com/v/biz/', 'source': 'Variety Business'},
        {'url': 'https://variety.com/t/box-office/', 'source': 'Variety Box Office'},
        {'url': 'https://variety.com/t/streaming/', 'source': 'Variety Streaming'},
        # Increase depth, crawl pages 2 and 3
        {'url': 'https://variety.com/page/2/', 'source': 'Variety Page 2'},
        {'url': 'https://variety.com/page/3/', 'source': 'Variety Page 3'},
        {'url': 'https://variety.com/v/film/page/2/', 'source': 'Variety Film Page 2'},
        {'url': 'https://variety.com/v/tv/page/2/', 'source': 'Variety TV Page 2'}
    ]
    
    # Get articles from multiple pages
    for source_config in extended_sources:
        url = source_config['url']
        source_name = source_config['source']
        
        try:
            print(f"Fetching articles from {url}...")
            headers = get_random_headers(user_agents)
            response = requests.get(url, headers=headers, timeout=20)
            
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = []
            for selector in article_selectors:
                for link in soup.select(selector):
                    href = link.get('href', '')
                    if href and re.search(r'variety\.com/\d{4}/', href):
                        article_links.append(href)
            
            # Deduplicate and display progress
            article_links = list(set(article_links))
            print(f"Found {len(article_links)} potential article links in {url}")
            
            # Randomly shuffle article links for diversity
            random.shuffle(article_links)
            
            # Process found article links
            processed = 0
            for link in article_links[:MAX_ARTICLES_PER_SOURCE]:
                try:
                    # Use random delay to reduce server load
                    time.sleep(random.uniform(0.2, 0.5))
                    article_res = requests.get(link, headers=get_random_headers(user_agents), timeout=15)
                    
                    if article_res.status_code != 200:
                        continue
                        
                    article_soup = BeautifulSoup(article_res.content, 'html.parser')
                    
                    # Try various title selectors
                    title_elem = article_soup.select_one('h1') or article_soup.select_one('.entry-title') or article_soup.select_one('.c-title')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    
                    # Extract article content
                    article_text = extract_article_text(article_soup)
                    
                    # Extract all images and captions - use multi-image processing function
                    img_urls, img_captions = extract_article_images(article_soup, link)
                    
                    # Insert image tags into text
                    if img_urls:
                        article_text = insert_image_tags(article_text, len(img_urls))
                    
                    has_image = len(img_urls) > 0
                    
                    if title:
                        # If collector instance is provided, use real-time deduplication
                        if topics_collector:
                            article_data = {
                                'topic': title,
                                'text': article_text,
                                'img_urls': img_urls,
                                'captions': img_captions,
                                'source': source_name,
                                'category': 'entertainment',
                                'timestamp': datetime.now().isoformat(),
                                'url': link,
                                'has_image': has_image,
                                'is_recent': True  # Variety usually has recent articles
                            }
                            
                            # Use real-time deduplication to add article
                            if topics_collector.add_topic_realtime(article_data):
                                processed += 1
                                print(f"Fetched Variety article ({processed}): {title[:40]}... [Images:{len(img_urls)}]")
                        else:
                            # Add using new format
                            all_articles.append({
                                'topic': title,
                                'text': article_text,
                                'img_urls': img_urls,
                                'captions': img_captions,
                                'source': source_name,
                                'category': 'entertainment',
                                'timestamp': datetime.now().isoformat(),
                                'url': link,
                                'has_image': has_image,
                                'is_recent': True
                            })
                            processed += 1
                            print(f"Fetched Variety article ({processed}): {title[:40]}... [Images:{len(img_urls)}]")
                    
                except Exception as e:
                    print(f"Error processing Variety article: {e}")
            
            print(f"Fetched {processed} articles from {source_name}")
            # Add a small delay between sources to avoid too frequent requests
            time.sleep(random.uniform(1.0, 2.0))
                    
        except Exception as e:
            print(f"Error fetching articles from {source_name}: {e}")
    
    # When using real-time deduplication, return topics directly from the collector
    if topics_collector:
        valid_articles = topics_collector.all_topics
        print(f"Variety fetched a total of {len(valid_articles)} valid articles, with {sum(1 for a in valid_articles if a.get('is_recent', False))} recent articles")
        return valid_articles
    else:
        # Prioritize recent articles
        all_articles.sort(key=lambda x: (0 if x.get('is_recent') else 1, 0 if x.get('has_image') else 1))
        valid_articles = [article for article in all_articles if is_article_url(article.get('url', ''))]
        print(f"Variety fetched a total of {len(valid_articles)} valid articles, with {sum(1 for a in valid_articles if a.get('is_recent', False))} recent articles")
        return valid_articles
