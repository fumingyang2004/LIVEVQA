"""HTML parsing utility functions"""

def safe_select(soup, selector):
    """Safely uses CSS selector"""
    try:
        return soup.select(selector)
    except Exception:
        return []

def safe_select_one(soup, selector):
    """Safely uses CSS selector to get a single element"""
    try:
        return soup.select_one(selector)
    except Exception:
        return None

def enhance_image_detection(article_soup, url):
    """Enhanced image detection logic"""
    # Call specific image detection based on website
    if 'cnn.com' in url:
        return detect_cnn_images(article_soup)
    elif 'yahoo.com' in url:
        return detect_yahoo_images(article_soup)
    elif 'variety.com' in url:
        return detect_variety_images(article_soup)
    elif 'bbc.com' in url or 'bbc.co.uk' in url:
        return detect_bbc_images(article_soup)
    elif 'forbes.com' in url:
        return detect_forbes_images(article_soup)
    elif 'apnews.com' in url:
        return detect_apnews_images(article_soup)
    
    # Generic image detection
    return detect_generic_images(article_soup)

def detect_generic_images(article_soup):
    """Generic image detection"""
    # Common image selectors
    image_selectors = [
        'article img', 'figure img', '.main-image', 
        '.article-image', '.story-img', '.image-container img'
    ]
    for selector in image_selectors:
        if article_soup.select_one(selector):
            return True
            
    # Check all images, exclude icons
    for img in article_soup.select('img'):
        if img.get('src') and not any(x in img.get('src', '').lower() for x in ['icon', 'logo', 'avatar', 'button']):
            return True
    
    return False

# Create dedicated image detection functions for each website
def detect_cnn_images(article_soup):
    """CNN image detection"""
    selectors = [
        '.media__image', '.image--media', 'img.media__image', 
        '.el__image--fullwidth', 'img.img-responsive',
        '.el__hero--standard-image', '.el-art-full__image',
        'picture img', '.image__picture img', '.el__gallery-image'
    ]
    for selector in selectors:
        if article_soup.select_one(selector):
            return True
            
    all_images = article_soup.select('img')
    for img in all_images:
        if img.get('width') and img.get('height'):
            try:
                width = int(img['width'])
                height = int(img['height'])
                if width > 200 and height > 200:
                    return True
            except (ValueError, TypeError):
                pass
        src = img.get('src', '')
        img_class = img.get('class', [])
        if any(term in src for term in ['photo', 'image', 'picture', '-super-', 'hero']):
            return True
        if any(cls and any(term in cls for term in ['photo', 'image', 'hero', 'main']) for cls in img_class):
            return True
    return False

def detect_yahoo_images(article_soup):
    """Yahoo image detection"""
    selectors = [
        '.caas-img', 'img.caas-img', '.caas-carousel',
        '.caas-cover-wrap', '.caas-figure'
    ]
    for selector in selectors:
        if safe_select_one(article_soup, selector):
            return True
            
    content_area = safe_select_one(article_soup, '.caas-body')
    if content_area and safe_select(content_area, 'img'):
        return True
        
    for img in article_soup.find_all('img'):
        src = img.get('src', '')
        if src and '/media/' in src and not any(x in src.lower() for x in ['icon', 'logo', 'avatar', 'button']):
            return True
    return False

def detect_bbc_images(article_soup):
    """BBC image detection"""
    selectors = [
        '.gs-o-media__img', '.lx-media__img', '.qa-story-image',
        '.qa-post-body-image', '.qa-post-body img', '.lx-stream-post__image img'
    ]
    for selector in selectors:
        if article_soup.select_one(selector):
            return True
            
    all_images = article_soup.select('img')
    for img in all_images:
        if img.get('width') and img.get('height'):
            try:
                width = int(img['width'])
                height = int(img['height'])
                if width > 200 and height > 200:
                    return True
            except (ValueError, TypeError):
                pass
        src = img.get('src', '')
        img_class = img.get('class', [])
        if any(term in src for term in ['photo', 'image', 'picture', '-super-', 'hero']):
            return True
        if any(cls and any(term in cls for term in ['photo', 'image', 'hero', 'main']) for cls in img_class):
            return True
    return False

def detect_forbes_images(article_soup):
    """Forbes image detection"""
    selectors = [
        '.article-image', '.fs-headline-image', '.fs-article-image',
        '.fs-article__image', '.fs-article__image img', '.fs-article__image-container img'
    ]
    for selector in selectors:
        if article_soup.select_one(selector):
            return True
            
    all_images = article_soup.select('img')
    for img in all_images:
        if img.get('width') and img.get('height'):
            try:
                width = int(img['width'])
                height = int(img['height'])
                if width > 200 and height > 200:
                    return True
            except (ValueError, TypeError):
                pass
        src = img.get('src', '')
        img_class = img.get('class', [])
        if any(term in src for term in ['photo', 'image', 'picture', '-super-', 'hero']):
            return True
        if any(cls and any(term in cls for term in ['photo', 'image', 'hero', 'main']) for cls in img_class):
            return True
    return False

def detect_apnews_images(article_soup):
    """AP News image detection"""
    selectors = [
        '.Article-visual img', '.RichTextBody img', '.Article-media img',
        '.LeadMediaItem img', '.FeedCard-media img', '.Figure img',
        '.LeadMedia-media img', '.LeadMediaContent img', '.ContentMedia-media img',
        '.Article-visualContent img', '.Primary-visualContent img'
    ]
    
    for selector in selectors:
        if article_soup.select_one(selector):
            return True
    
    # Check all images, exclude small icons
    all_images = article_soup.select('img')
    for img in all_images:
        if img.get('data-key') == 'card-image' or img.get('data-key') == 'lead-image':
            return True
            
        # Check dimensions
        if img.get('width') and img.get('height'):
            try:
                width = int(img['width'])
                height = int(img['height'])
                if width > 200 and height > 200:
                    return True
            except (ValueError, TypeError):
                pass
                
        # Check image URL and class
        src = img.get('src', '')
        img_class = ' '.join(img.get('class', []))
        if 'images.apnews.com' in src and not any(term in src.lower() for term in ['icon', 'logo', 'button']):
            return True
        if any(term in img_class.lower() for term in ['photo', 'image', 'hero', 'lead']):
            return True
            
    return False

def extract_title(article_soup, site_type=None):
    """Extracts article title"""
    title_selectors = {
        'cnn': ['h1.pg-headline', '.Article__title', '.article-title', '.headline'],
        'yahoo': ['h1', '.caas-title', '.canvas-header', '.headline', '[data-test-locator="headline"]'],
        'bbc': ['h1', 'h1.story-headline', 'h1.article-headline', 'h1.vxp-media__headline', 'h1.lx-stream-page__header-text', '.article-headline', '.story-body__h1'],
        'forbes': ['h1.article-headline', 'h1.fs-headline', 'h1.speakable-headline', 'h1.heading--tape', 'h1[data-ga-track="Title"]', 'h1.entry-title', 'article h1', '.article-header h1', '.article-title', 'h1.article-title', 'h1']
    }
    
    if site_type in title_selectors:
        for selector in title_selectors[site_type]:
            title_elem = article_soup.select_one(selector)
            if title_elem and len(title_elem.text.strip()) > 5:
                return title_elem.text.strip()
    
    # Generic title selector
    title_elem = article_soup.select_one('h1')
    if title_elem and len(title_elem.text.strip()) > 5:
        return title_elem.text.strip()
    
    return None
