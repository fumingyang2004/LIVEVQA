"""Image processing related utility functions"""
import os
import re
import hashlib
import urllib.parse
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import warnings

# Attempt to import computer vision libraries, use fallback methods if unavailable
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.transform import resize
    has_skimage = True
except ImportError:
    has_skimage = False
    warnings.warn("skimage library not found, using simplified image comparison method. Please install: pip install scikit-image")

try:
    import cv2
    has_opencv = True
except ImportError:
    has_opencv = False
    warnings.warn("OpenCV library not found, using simplified image comparison method. Please install: pip install opencv-python")

def extract_article_images(soup, article_url):
    """
    Extracts all images and their corresponding captions from an article
    
    Args:
        soup: BeautifulSoup object
        article_url: URL of the article
        
    Returns:
        tuple: (list of image URLs, list of image captions)
    """
    img_urls = []
    img_captions = []
    
    # 1. Extract Open Graph and Twitter Card images
    og_image = soup.find('meta', property='og:image')
    if og_image and og_image.get('content'):
        img_urls.append(urllib.parse.urljoin(article_url, og_image.get('content')))
        img_captions.append(None)
        
    twitter_image = soup.find('meta', {'name': 'twitter:image'}) or soup.find('meta', property='twitter:image')
    if twitter_image and twitter_image.get('content'):
        twitter_img_url = urllib.parse.urljoin(article_url, twitter_image.get('content'))
        # Check if it duplicates the OG image
        if twitter_img_url not in img_urls:
            img_urls.append(twitter_img_url)
            img_captions.append(None)
    
    # 2. Extract images within image containers
    figure_selectors = [
        'figure', '.image-and-copyright-container', '.article-body__image-container',
        '.responsive-image-container', '.media-with-caption', '.image--inline', 
        '.image', '.inline-img-container', '.media-landscape', '.article__body-image'
    ]
    
    for selector in figure_selectors:
        figures = soup.select(selector)
        for figure in figures:
            # Find image
            img = figure.find('img')
            if not img or not img.get('src'):
                continue
                
            # Process image URL
            img_src = img.get('src')
            if img_src.startswith('data:'):
                continue  # Skip base64 encoded images
                
            # Handle relative URLs
            img_url = urllib.parse.urljoin(article_url, img_src)
            
            # Find corresponding caption
            caption = None
            caption_selectors = [
                'figcaption', '.caption', '.media-caption', '.image-caption',
                '.caption-text', '.wp-caption-text', '.caption__text',
                '.image__caption', '.media__caption', '.article-caption'
            ]
            
            for caption_selector in caption_selectors:
                caption_elem = figure.select_one(caption_selector)
                if caption_elem and caption_elem.get_text().strip():
                    caption = caption_elem.get_text().strip()
                    break
            
            # Avoid adding duplicate images
            if img_url not in img_urls:
                img_urls.append(img_url)
                img_captions.append(caption)
    
    # 3. Extract all images embedded within the article content
    content_selectors = [
        'article', '.article-body', '.article__body', '.story-body',
        '.entry-content', '.post-content', '.content', '.main-content'
    ]
    
    for selector in content_selectors:
        content = soup.select_one(selector)
        if content:
            images = content.find_all('img')
            for img in images:
                if not img.get('src'):
                    continue
                    
                # Exclude small icons and decorative images
                if 'icon' in img.get('src', '').lower() or 'logo' in img.get('src', '').lower():
                    continue
                    
                img_src = img.get('src')
                if img_src.startswith('data:'):
                    continue
                    
                img_url = urllib.parse.urljoin(article_url, img_src)
                
                # Avoid duplicates
                if img_url not in img_urls:
                    img_urls.append(img_url)
                    
                    # Try to find adjacent caption
                    caption = None
                    next_elem = img.find_next_sibling()
                    if next_elem and next_elem.name in ['figcaption', 'small', 'span', 'div']:
                        for cls in ['caption', 'wp-caption-text', 'image-caption']:
                            if cls in next_elem.get('class', []):
                                caption = next_elem.get_text().strip()
                                break
                        
                        if not caption and len(next_elem.get_text().strip()) < 100:
                            caption = next_elem.get_text().strip()
                    
                    img_captions.append(caption)
    
    # Filter and sort images
    filtered_urls, filtered_captions = filter_images(img_urls, img_captions)
    
    return filtered_urls, filtered_captions

def filter_images(img_urls, img_captions):
    """
    Filters the list of images according to new rules:
    1. Sorts by image area in descending order
    2. Keeps a maximum of 4 images
    3. Deletes images with an area less than half of the largest image
    
    Args:
        img_urls: List of image URLs
        img_captions: List of image captions
        
    Returns:
        tuple: (filtered list of image URLs, corresponding list of captions)
    """
    if not img_urls:
        return [], []
    
    # Store processed image URLs to avoid duplicates
    processed_urls = set()
    
    # Temporary storage for image information (URL, caption, area)
    image_info = []
    
    for i, url in enumerate(img_urls):
        # Skip already processed URLs
        if url in processed_urls:
            continue
            
        processed_urls.add(url)
        
        try:
            # Get image dimensions
            response = requests.get(url, stream=True, timeout=5)
            if response.status_code != 200:
                continue
                
            # Use PIL to open the image and get dimensions
            img = Image.open(BytesIO(response.content))
            width, height = img.size
            area = width * height
            
            # Store image information
            caption = img_captions[i] if i < len(img_captions) else "null"
            image_info.append({
                'url': url,
                'caption': caption if caption else "null",
                'area': area,
                'width': width,
                'height': height,
                'img': img,  # Save image object for potential visual comparison
                'content': response.content  # Save original content to avoid re-downloading
            })
            
        except Exception as e:
            print(f"Error getting image dimensions {url}: {e}")
            continue
    
    # If no images were successfully processed
    if not image_info:
        return [], []
    
    # Image deduplication - using visual similarity
    unique_images = deduplicate_images(image_info)
    
    # Sort by area in descending order
    unique_images.sort(key=lambda x: x['area'], reverse=True)
    
    # Get the area of the largest image
    max_area = unique_images[0]['area']
    half_max_area = max_area * 0.3
    
    # Apply filtering rules
    filtered_info = []
    for img in unique_images:
        # Rule 3: Only keep images with an area not less than half of the largest image
        if img['area'] >= half_max_area:
            filtered_info.append(img)
    
    # Rule 2: Keep a maximum of 4 images
    if len(filtered_info) > 4:
        filtered_info = filtered_info[:4]
    
    # Extract filtered URLs and captions
    filtered_urls = [img['url'] for img in filtered_info]
    filtered_captions = [img['caption'] for img in filtered_info]
    
    # Clean up no longer needed image objects
    for img in image_info:
        if 'img' in img:
            try:
                img['img'].close()
            except:
                pass
        img.pop('img', None)
        img.pop('content', None)
    
    return filtered_urls, filtered_captions

def deduplicate_images(image_info):
    """
    Deduplicates images using computer vision algorithms
    
    Args:
        image_info: List of dictionaries containing image information
        
    Returns:
        list: List of deduplicated image information
    """
    # No need to deduplicate if less than 2 images
    if len(image_info) < 2:
        return image_info
    
    # Mark found duplicate images
    duplicates = set()
    num_images = len(image_info)
    
    # If OpenCV or scikit-image is available, use visual similarity comparison
    if has_opencv or has_skimage:
        # Compare all images pairwise
        for i in range(num_images):
            if i in duplicates:
                continue
                
            for j in range(i + 1, num_images):
                if j in duplicates:
                    continue
                    
                # Calculate similarity
                similarity = calculate_image_similarity(image_info[i], image_info[j])
                
                # If similarity is above threshold, mark as duplicate
                if similarity > 0.85:  # Threshold can be adjusted
                    # Keep the larger image, discard the smaller one
                    if image_info[i]['area'] >= image_info[j]['area']:
                        duplicates.add(j)
                    else:
                        duplicates.add(i)
                        break  # If i is marked as duplicate, no need to continue comparing
    else:
        # Fallback to simple deduplication based on URL and size
        for i in range(num_images):
            if i in duplicates:
                continue
                
            for j in range(i + 1, num_images):
                if j in duplicates:
                    continue
                    
                # If URLs contain the same image ID or filename part
                url_i = image_info[i]['url'].split('/')[-1]
                url_j = image_info[j]['url'].split('/')[-1]
                
                # Find common image ID or name part
                common_parts = False
                if len(url_i) > 10 and len(url_j) > 10:
                    for part in url_i.split('-'):
                        if len(part) > 8 and part in url_j:
                            common_parts = True
                            break
                
                # If sizes are similar and URLs have common parts, it might be different versions of the same image
                if common_parts and abs(image_info[i]['width'] - image_info[j]['width']) < 100:
                    # Keep the larger image
                    if image_info[i]['area'] >= image_info[j]['area']:
                        duplicates.add(j)
                    else:
                        duplicates.add(i)
                        break
    
    # Return images not marked as duplicate
    return [img for i, img in enumerate(image_info) if i not in duplicates]

def calculate_image_similarity(img_info1, img_info2):
    """
    Calculates the visual similarity between two images
    
    Args:
        img_info1: Information of the first image
        img_info2: Information of the second image
        
    Returns:
        float: Similarity score (0-1), 1 for identical
    """
    # Ensure image data is available
    if 'img' not in img_info1 or 'img' not in img_info2:
        return 0
    
    try:
        # Resize images to the same size for comparison
        target_size = (128, 128)  # Smaller size for faster calculation
        
        if has_opencv:
            # Calculate similarity using OpenCV
            img1 = np.array(img_info1['img'].resize(target_size).convert('L'))  # Convert to grayscale
            img2 = np.array(img_info2['img'].resize(target_size).convert('L'))  # Convert to grayscale
            
            # Use histogram comparison
            hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # If similarity is high, confirm with Structural Similarity Index
            if score > 0.8 and has_skimage:
                # Calculate SSIM
                ssim_score = ssim(img1, img2)
                score = (score + ssim_score) / 2  # Average of both methods
            
            return max(0, min(1, score))  # Ensure within 0-1 range
            
        elif has_skimage:
            # Use scikit-image's Structural Similarity Index
            img1 = np.array(img_info1['img'].resize(target_size).convert('L'))
            img2 = np.array(img_info2['img'].resize(target_size).convert('L'))
            
            return ssim(img1, img2)
        else:
            # No computer vision library available, use simple size comparison
            w1, h1 = img_info1['width'], img_info1['height']
            w2, h2 = img_info2['width'], img_info2['height']
            
            # Simple similarity calculation based on aspect ratio and area
            aspect_ratio1 = w1 / h1 if h1 > 0 else 0
            aspect_ratio2 = w2 / h2 if h2 > 0 else 0
            
            # Aspect ratio similarity
            ratio_similarity = 1 - min(abs(aspect_ratio1 - aspect_ratio2) / max(aspect_ratio1, aspect_ratio2), 1)
            
            # Size similarity
            size_similarity = min(img_info1['area'], img_info2['area']) / max(img_info1['area'], img_info2['area'])
            
            # Combine similarity scores
            return (ratio_similarity * 0.3 + size_similarity * 0.7) * 0.8  # Reduce overall score when using simple method
            
    except Exception as e:
        print(f"Error calculating image similarity: {e}")
        return 0

def download_multiple_images(img_urls, save_dir, source_name=None):
    """
    Downloads multiple images
    
    Args:
        img_urls: List of image URLs
        save_dir: Directory to save to
        source_name: Name of the source
        
    Returns:
        list: List of saved image paths, None at corresponding position if failed
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = []
    
    for img_url in img_urls:
        try:
            # Generate unique filename
            img_hash = hashlib.md5(img_url.encode()).hexdigest()
            img_ext = os.path.splitext(urllib.parse.urlparse(img_url).path)[1]
            if not img_ext or len(img_ext) > 5:
                img_ext = '.jpg'
            
            # Add source prefix
            prefix = f"{source_name}_" if source_name else ""
            filename = f"{prefix}{img_hash}{img_ext}"
            filepath = os.path.join(save_dir, filename)
            
            # If file already exists, return path directly
            if os.path.exists(filepath):
                saved_paths.append(filepath)
                continue
            
            # Add request headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
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
            
            saved_paths.append(filepath)
            print(f"Downloaded image: {img_url} -> {filepath}")
            
        except Exception as e:
            print(f"Failed to download image {img_url}: {e}")
            saved_paths.append(None)  # Placeholder for failed download
    
    return saved_paths

def insert_image_tags(text, num_images):
    """
    Inserts image tags into the text
    
    Args:
        text: Original text
        num_images: Number of images
        
    Returns:
        str: Text with image tags inserted
    """
    if not text or num_images <= 0:
        return text
    
    # Process by paragraph, insert one image every 3 paragraphs
    paragraphs = text.split('\n\n')
    if len(paragraphs) <= 1:
        # If not enough paragraphs, insert all image tags at the beginning
        tags = ' '.join([f"<img{i+1}>" for i in range(min(num_images, 9))])
        return f"{tags} {text}"
    
    # Calculate interval for inserting image tags
    interval = max(1, len(paragraphs) // (num_images + 1))
    
    # Insert image tags dispersed throughout the text
    result = []
    img_idx = 0
    
    for i, para in enumerate(paragraphs):
        result.append(para)
        
        # Insert image tag after appropriate paragraph
        if i > 0 and i % interval == 0 and img_idx < num_images:
            result.append(f"<img{img_idx+1}>")
            img_idx += 1
    
    # If there are remaining image tags, append them at the end
    while img_idx < min(num_images, 9):
        result.append(f"<img{img_idx+1}>")
        img_idx += 1
    
    return '\n\n'.join(result)

def compare_images_from_urls(url1, url2, threshold=0.85):
    """
    Compares the similarity of images from two URLs
    
    Args:
        url1: URL of the first image
        url2: URL of the second image
        threshold: Similarity threshold
        
    Returns:
        bool: True if image similarity exceeds the threshold
    """
    try:
        # Download images
        response1 = requests.get(url1, stream=True, timeout=5)
        response2 = requests.get(url2, stream=True, timeout=5)
        
        if response1.status_code != 200 or response2.status_code != 200:
            return False
            
        # Open images
        img1_info = {
            'img': Image.open(BytesIO(response1.content)),
            'content': response1.content,
            'width': 0,
            'height': 0,
            'area': 0
        }
        
        img2_info = {
            'img': Image.open(BytesIO(response2.content)),
            'content': response2.content,
            'width': 0,
            'height': 0,
            'area': 0
        }
        
        # Get dimensions
        img1_info['width'], img1_info['height'] = img1_info['img'].size
        img1_info['area'] = img1_info['width'] * img1_info['height']
        
        img2_info['width'], img2_info['height'] = img2_info['img'].size
        img2_info['area'] = img2_info['width'] * img2_info['height']
        
        # Calculate similarity
        similarity = calculate_image_similarity(img1_info, img2_info)
        
        # Release resources
        img1_info['img'].close()
        img2_info['img'].close()
        
        return similarity > threshold
        
    except Exception as e:
        print(f"Error comparing images: {e}")
        return False
