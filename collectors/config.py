from tkinter import W
import openai
import os

# Base configuration
WORKSPACE = "YOUR WORKSPACE NAME"
OPENAI_API_KEY = "YOUR OPENAI API KEY"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Data and image storage paths
DATA_DIR = os.path.join(WORKSPACE, "data")
IMG_DIR = os.path.join(DATA_DIR, "imgs")

# Define topic categories
CATEGORIES = {
    'real_events': ['news', 'sports', 'entertainment', 'health', 'business', 'politics', 'history'],
    'virtual_events': ['films', 'drama', 'anime'],
    'entities': ['location', 'people', 'time', 'technology', 'science', 'culture', 'environment']
}
ALL_CATEGORIES = [item for sublist in CATEGORIES.values() for item in sublist]

# Common stopwords
STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'on', 'in', 'at', 
             'to', 'for', 'with', 'by', 'about', 'as', 'of', 'this', 'that', 'these', 
             'those', 'it', 'its', 'from', 'has', 'have', 'had', 'was', 'were', 'be', 
             'been', 'being', 'will', 'would', 'should', 'could', 'may', 'might', 'must'}

# ============= Crawler Configuration =============
# To enable a crawler, ensure the corresponding section of the code is uncommented
# To disable a crawler, add a '#' comment symbol before the corresponding section

# Maximum number of articles to collect
# You can set this num to change the max number of articles collected per source
MAX_ARTICLES_PER_SOURCE = 300  # Collect up to 300 articles per source

# CNN Crawler Configuration - Enable/Disable
# You can change Ture/False to enable or disable the crawler, the default is True.
ENABLE_CNN = True
# ENABLE_CNN = False
CNN_SECTIONS = [
    {'url': 'https://edition.cnn.com/politics', 'source': 'CNN Politics', 'category': 'politics'},
    {'url': 'https://edition.cnn.com/business', 'source': 'CNN Business', 'category': 'business'},
    {'url': 'https://edition.cnn.com/health', 'source': 'CNN Health', 'category': 'health'},
    {'url': 'https://edition.cnn.com/sport', 'source': 'CNN Sport', 'category': 'sports'},
    # Add latest news page to ensure daily content is fetched
    {'url': 'https://edition.cnn.com/', 'source': 'CNN Latest', 'category': 'news'},
    {'url': 'https://edition.cnn.com/world', 'source': 'CNN World', 'category': 'news'}
] if ENABLE_CNN else []

# Variety Crawler Configuration - Enable/Disable
ENABLE_VARIETY = True
# ENABLE_VARIETY = False
VARIETY_SOURCES = [
    {'url': 'https://variety.com/', 'source': 'Variety'},
    {'url': 'https://variety.com/v/film/', 'source': 'Variety Film'},
    {'url': 'https://variety.com/v/tv/', 'source': 'Variety TV'},
    {'url': 'https://variety.com/v/music/', 'source': 'Variety Music'},
    {'url': 'https://variety.com/c/digital/', 'source': 'Variety Digital'},
    {'url': 'https://variety.com/t/documentaries-to-watch/', 'source': 'Variety Docs'},
    {'url': "https://variety.com/c/global/", 'source': 'Variety Global'}
] if ENABLE_VARIETY else []

# BBC Crawler Configuration - Enable/Disable
ENABLE_BBC = True
# ENABLE_BBC = False
BBC_SECTIONS = [
    {'url': 'https://www.bbc.com/news/world', 'source': 'BBC World', 'category': 'news'},
    {'url': 'https://www.bbc.com/news/uk', 'source': 'BBC UK', 'category': 'news'},
    {'url': 'https://www.bbc.com/news/business', 'source': 'BBC Business', 'category': 'business'},
    {'url': 'https://www.bbc.com/news/politics', 'source': 'BBC Politics', 'category': 'politics'},
    {'url': 'https://www.bbc.com/news/technology', 'source': 'BBC Technology', 'category': 'science'},
    {'url': 'https://www.bbc.com/news/science_and_environment', 'source': 'BBC Science', 'category': 'science'},
    {'url': 'https://www.bbc.com/news/health', 'source': 'BBC Health', 'category': 'health'},
    {'url': 'https://www.bbc.com/news/education', 'source': 'BBC Education', 'category': 'news'},
    {'url': 'https://www.bbc.com/sport', 'source': 'BBC Sport', 'category': 'sports'},
    {'url': 'https://www.bbc.com/sport/football', 'source': 'BBC Football', 'category': 'sports'},
    {'url': 'https://www.bbc.com/sport/olympics', 'source': 'BBC Olympics', 'category': 'sports'},
    {'url': 'https://www.bbc.com/culture', 'source': 'BBC Culture', 'category': 'entertainment'},
    {'url': 'https://www.bbc.com/culture/entertainment-news', 'source': 'BBC Entertainment', 'category': 'entertainment'},
    {'url': 'https://www.bbc.com/travel', 'source': 'BBC Travel', 'category': 'lifestyle'},
    {'url': 'https://www.bbc.com/future', 'source': 'BBC Future', 'category': 'science'}
] if ENABLE_BBC else []

# Forbes Crawler Configuration - Enable/Disable
ENABLE_FORBES = True
# ENABLE_FORBES = False
FORBES_SECTIONS = [
    {'url': 'https://www.forbes.com/business/', 'source': 'Forbes Business', 'category': 'business'},
    {'url': 'https://www.forbes.com/money/', 'source': 'Forbes Money', 'category': 'business'},
    {'url': 'https://www.forbes.com/innovation/', 'source': 'Forbes Innovation', 'category': 'science'},
    {'url': 'https://www.forbes.com/leadership/', 'source': 'Forbes Leadership', 'category': 'business'},
    {'url': 'https://www.forbes.com/lifestyle/', 'source': 'Forbes Lifestyle', 'category': 'lifestyle'},
    {'url': 'https://www.forbes.com/hollywood-entertainment/', 'source': 'Forbes Entertainment', 'category': 'entertainment'},
    {'url': 'https://www.forbes.com/consumer/', 'source': 'Forbes Consumer', 'category': 'business'},
    {'url': 'https://www.forbes.com/lists/', 'source': 'Forbes Lists', 'category': 'business'},
    {'url': 'https://www.forbes.com/worlds-billionaires/', 'source': 'Forbes Billionaires', 'category': 'business'},
    {'url': 'https://www.forbes.com/', 'source': 'Forbes Home', 'category': 'news'}
] if ENABLE_FORBES else []

# AP News Crawler Configuration - Enable/Disable
ENABLE_APNEWS = True
# ENABLE_APNEWS = False
APNEWS_SECTIONS = [
    {'url': 'https://apnews.com/politics', 'source': 'AP News Politics', 'category': 'politics'},
    {'url': 'https://apnews.com/entertainment', 'source': 'AP News Entertainment', 'category': 'entertainment'},
    {'url': 'https://apnews.com/business', 'source': 'AP News Business', 'category': 'business'},
    {'url': 'https://apnews.com/science', 'source': 'AP News Science', 'category': 'science'},
    # Add homepage for more comprehensive news
    {'url': 'https://apnews.com/', 'source': 'AP News', 'category': 'news'},
    {'url': 'https://apnews.com/world-news', 'source': 'AP News World', 'category': 'news'},
    {'url': 'https://apnews.com/us-news', 'source': 'AP News US', 'category': 'news'},
    {'url': 'https://apnews.com/health', 'source': 'AP News Health', 'category': 'health'}
] if ENABLE_APNEWS else []
