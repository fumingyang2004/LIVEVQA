"""URL-related utility functions"""
import re
import urllib.parse

def is_article_url(url):
    """Determines if a URL is a specific article rather than a section page"""
    if not url: return False
    
    # Handle by website category
    if 'cnn.com' in url:
        return is_cnn_article_url(url)
    elif 'bbc.com' in url or 'bbc.co.uk' in url:
        return is_bbc_article_url(url)
    elif 'variety.com' in url:
        return is_variety_article_url(url)
    elif 'yahoo.com' in url:
        return is_yahoo_article_url(url)
    elif 'forbes.com' in url:
        return is_forbes_article_url(url)
    elif 'apnews.com' in url:
        return is_apnews_article_url(url)
    
    # Generic article URL pattern check
    article_patterns = ['/article/', '/story/', '/news/', '/index.html', '.html', '.htm', '/posts/', '/articles/']
    non_article_patterns = ['/tag/', '/author/', '/about/', '/contact/', '/search', '/category/', 
                            '/gallery/', '/video/', '/live/', 'facebook.com', 'twitter.com']
    
    return (any(pattern in url.lower() for pattern in article_patterns) and 
            not any(pattern in url.lower() for pattern in non_article_patterns))

def get_canonical_url(url):
    """Processes URL, removes tracking parameters, etc., to get the canonical URL"""
    if not url:
        return url
        
    # Remove query parameters from URL
    if '?' in url:
        url = url.split('?')[0]
        
    return url

def join_url(base_url, path):
    """Safely joins URLs"""
    if not path:
        return base_url
        
    if path.startswith('http'):
        return path
        
    return urllib.parse.urljoin(base_url, path)

# Create dedicated URL checking functions for each website
def is_cnn_article_url(url):
    """Determines if it's a CNN article URL"""
    # Special handling for CNN article URLs
    if 'cnn.com' in url:
        # Exclude main CNN category pages
        main_sections = ['https://edition.cnn.com', 'https://www.cnn.com', 
                         'https://edition.cnn.com/politics', 'https://edition.cnn.com/business',
                         'https://edition.cnn.com/health', 'https://edition.cnn.com/sport']
        if url in main_sections:
            return False
        
        # Identify valid CNN URL patterns
        if re.search(r'cnn\.com/\d{4}/\d{2}/\d{2}/', url) or '/index.html' in url:
            return True
            
        # Exclude video and other pages
        if any(pattern in url.lower() for pattern in ['/videos/', '/gallery/', '/live-news/']):
            return False
            
        # URLs containing subcategories
        return any(section in url for section in ['/politics/', '/business/', '/health/', '/sport/'])
    return False

def is_bbc_article_url(url):
    """Determines if it's a BBC article URL"""
    if 'bbc.com' in url and ('/news/articles/' in url or '/sport/articles/' in url): 
        return True
    return False

def is_forbes_article_url(url):
    """Determines if it's a Forbes article URL"""
    if not url or 'forbes.com' not in url:
        return False
    
    # Exclude pages that are clearly not articles
    exclude_patterns = [
        '/signup/', '/login/', '/subscription/', '/about/', '/tags/', '/topics/', 
        '/search/', '/video/', '/videos/', '/newsletters/', '/lists/index',
        '/marketplace/', '/advisor/', '/sites/advice/', '/sites/advisorindex/',
        '/tableau/', '/offers/', '/forbes-live/', '/podcasts/', '/sites/forbespr/',
        # Add more exclusion patterns
        '/dashboard/', '/connect/', '/profile/', '/privacy/', '/terms/', 
        '/follow/', '/leadership/'
    ]
    
    if any(pattern in url.lower() for pattern in exclude_patterns):
        return False
    
    # Index page check, avoid crawling category index pages
    if url.strip('/').endswith('forbes.com'):
        return False
    
    # Check common category index pages
    category_index_patterns = [
        r'forbes\.com/[a-z-]+/?$',  # e.g., forbes.com/business/
    ]
    
    for pattern in category_index_patterns:
        if re.match(pattern, url.lower()):
            # Must contain 'sites' or 'year' to be considered an article
            if '/sites/' not in url and not re.search(r'/\d{4}/', url):
                return False
    
    # 1. Forbes article URLs are typically characterized by date and article title
    # Example: https://www.forbes.com/sites/johnauthor/2023/05/10/article-title/
    if re.search(r'forbes\.com/sites/[^/]+/\d{4}/\d{1,2}/\d{1,2}/[^/]+', url):
        return True
        
    # 2. Another Forbes article format (articles without date)
    # Example: https://www.forbes.com/sites/johnauthor/article-title/
    if re.search(r'forbes\.com/sites/[^/]+/[^/]+/?$', url) and '/sites/' in url:
        return True
    
    # 3. Some special article URL formats
    if '/sites/' in url and url.count('/') >= 5:
        return True
        
    # 4. Exclude pagination paths
    if re.search(r'/page/\d+/?$', url):
        return False
    
    return False

def is_yahoo_article_url(url):
    """Determines if it's a Yahoo article URL"""
    # Special handling for Yahoo article URLs - more lenient judgment
    if 'yahoo.com' in url:
        # Explicitly exclude non-article pages
        if any(pattern in url for pattern in ['/category/', '/tag/', '/author/', '/search/', 'yahoo.com/news/?', 'yahoo.com/lifestyle/?']):
            return False
            
        # Standard Yahoo article URL
        if re.search(r'yahoo\.com/news/[^/]+-\d+\.html$', url):
            return True
            
        # Extended Yahoo article URL format
        if re.search(r'yahoo\.com/(news|lifestyle|entertainment)/[^/]+', url) and url.endswith('.html'):
            return True
            
        # Finance special handling
        if 'finance.yahoo.com' in url and '/news/' in url:
            return True
            
        # More lenient format matching, but exclude homepage and category pages
        if url.endswith('.html') and not url.endswith('index.html'):
            if re.search(r'yahoo\.com/[^/]+/[^/]+', url):  # At least two path segments
                return True
    return False

def is_variety_article_url(url):
    """Determines if it's a Variety article URL"""
    # Special handling for Variety article URLs - more lenient judgment
    if 'variety.com' in url:
        # Exclude category and tag pages
        if re.search(r'variety\.com/([vclt])/[^/]+/?$', url) or '/page/' in url:
            return False
        # Match common year format
        if re.search(r'variety\.com/\d{4}/', url):
            return True
        # Try to match other article formats
        return bool(re.search(r'variety\.com/[^/]+/\d+', url))
    return False

# Add AP News URL checking function
def is_apnews_article_url(url):
    """Determines if it's an AP News article URL"""
    if not url or 'apnews.com' not in url:
        return False
    
    # Exclude non-article pages
    exclude_patterns = [
        '/about/', '/contact/', '/tag/', '/author/', 
        '/search/', '/video/', '/videos/', '/gallery/'
    ]
    
    if any(pattern in url.lower() for pattern in exclude_patterns):
        return False
    
    # AP News article URLs typically contain /article/ path and a hash ID
    if '/article/' in url and url.count('/') >= 3:
        return True
    
    return False
