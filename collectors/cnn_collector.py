import time
import random
import requests
import urllib.parse
import re
from datetime import datetime
from bs4 import BeautifulSoup
from collectors.utils import (
    scrape_safely, is_recent_article, 
    is_article_url, get_random_headers, extract_article_text
)
from collectors.utils_images import (
    extract_article_images, filter_images, insert_image_tags
)
from collectors.config import CNN_SECTIONS, MAX_ARTICLES_PER_SOURCE

@scrape_safely
def scrape_cnn_news(user_agents, topics_collector=None):
    """Scrapes news from various sections of CNN
    
    Args:
        user_agents: List of user agents
        topics_collector: Hot topics collector instance, for real-time deduplication
    """
    all_topics = []
    
    for section in CNN_SECTIONS:
        url = section['url']
        source = section['source']
        category = section['category']
        
        try:
            print(f"Scraping CNN {category} news...")
            response = requests.get(url, headers=get_random_headers(user_agents), timeout=20)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            article_urls = set()
            
            # 1. Get links containing the current year (ensure latest)
            current_year = datetime.now().year
            for link in soup.select(f'a[href*="/{current_year}/"]'):
                href = link.get('href', '')
                if href and not href.startswith('#'):
                    href = urllib.parse.urljoin(url, href) if not href.startswith('http') else href
                    if 'cnn.com' in href:
                        article_urls.add(href)
            
            # 2. Get links containing index.html
            for link in soup.select('a[href*="index.html"]'):
                href = link.get('href', '')
                if href and not href.startswith('#'):
                    href = urllib.parse.urljoin(url, href) if not href.startswith('http') else href
                    if 'cnn.com' in href:
                        article_urls.add(href)
            
            # 3. Look for article links within regions
            for container in soup.select('.container__item, .card, .cd__content, article, .zn__containers'):
                for link in container.select('a[href]'):
                    href = link.get('href', '')
                    if not href or href.startswith('#'): 
                        continue
                    
                    href = urllib.parse.urljoin(url, href) if not href.startswith('http') else href
                    if 'cnn.com' in href:
                        article_urls.add(href)
            
            # 4. Look for headline or featured articles
            for link in soup.select('.container_lead-plus-headlines a, .cd__headline a'):
                href = link.get('href', '')
                if href and not href.startswith('#'):
                    href = urllib.parse.urljoin(url, href) if not href.startswith('http') else href
                    if 'cnn.com' in href:
                        article_urls.add(href)
            
            # Process discovered URLs
            processed = 0
            for article_url in article_urls:
                if not is_article_url(article_url) or processed >= MAX_ARTICLES_PER_SOURCE:
                    continue
                
                try:
                    article_res = requests.get(article_url, headers=get_random_headers(user_agents), timeout=15)
                    if article_res.status_code != 200:
                        continue
                        
                    article_soup = BeautifulSoup(article_res.content, 'html.parser')
                    
                    # Extract title
                    title_elem = article_soup.select_one('h1.pg-headline, .Article__title, .article-title, .headline') or article_soup.select_one('h1')
                    if not title_elem or len(title_elem.text.strip()) < 5:
                        continue
                    
                    # Modify title processing to keep only the first line and remove all extra content
                    title_text = title_elem.text.strip()
                    # Keep only the first line as the title
                    title = title_text.split('\n')[0].strip()
                    
                    # Extract article content
                    article_text = extract_article_text(article_soup)
                    
                    # Extract all images and captions - use multi-image processing function
                    img_urls, img_captions = extract_article_images(article_soup, article_url)
                    
                    # Insert image tags into text
                    if img_urls:
                        article_text = insert_image_tags(article_text, len(img_urls))
                    
                    has_image = len(img_urls) > 0
                    
                    is_recent = is_recent_article(article_url, article_soup)
                    
                    # Use category as defined in the section directly
                    curr_category = category
                    
                    # If collector instance is provided, use real-time deduplication
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
                        
                        # Use real-time deduplication to add article
                        if topics_collector.add_topic_realtime(article_data):
                            processed += 1
                            print(f"Fetched CNN article ({processed}): {title[:40]}... [Images:{len(img_urls)}]")
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
                        print(f"Fetched CNN article ({processed}): {title[:40]}... [Images:{len(img_urls)}]")
                    
                    time.sleep(random.uniform(0.3, 0.6))
                except Exception as e:
                    print(f"Error processing CNN article: {e}")
            
            print(f"Fetched {processed} articles from CNN {category}")
            time.sleep(random.uniform(1.0, 1.5))
        except Exception as e:
            print(f"Failed to scrape CNN {category}: {e}")
    
    # When using real-time deduplication, return topics directly from the collector
    if topics_collector:
        valid_articles = topics_collector.all_topics
        print(f"CNN fetched a total of {len(valid_articles)} valid articles")
        return valid_articles
    else:
        # Original filtering logic
        all_topics.sort(key=lambda x: (0 if x.get('is_recent') else 1, 0 if x.get('has_image') else 1))
        valid_articles = [article for article in all_topics if is_article_url(article.get('url', ''))]
        print(f"CNN fetched a total of {len(valid_articles)} valid articles")
        return valid_articles
