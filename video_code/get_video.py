import os
import argparse
import numpy as np
from datetime import datetime, timedelta
import glob
import traceback
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp
import time
import re
import isodate
from langdetect import detect, LangDetectException
import subprocess
import whisper  
import json  
import openai 
import logging  
import concurrent.futures  

def setup_logger(output_dir, name='youtube_crawler'):
    """Set up a logger and create a timestamped log file"""
    # 获取现有的记录器或创建新的
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    logger.info(f"Log file created: {log_file}")
    return logger

def load_known_video_ids(subdirectories_file='subdirectories.txt'):
    """
    Load known video IDs from subdirectories.txt
    
    Args:
        subdirectories_file: Path to subdirectories.txt
    
    Returns:
        set: Set of all known video IDs
    """
    known_ids = set()
    try:
        if os.path.exists(subdirectories_file):
            with open(subdirectories_file, 'r', encoding='utf-8') as file:
                # Use regex to extract possible video IDs
                for line in file:
                    # Skip headers and empty lines
                    if line.startswith("目录扫描结果") or line.startswith("目录") or not line.strip():
                        continue
                    
                    # Parse each line and look for video ID patterns (e.g., "1. YT5DIbzF3dE" or "123. -779Zc2xS10")
                    matches = re.search(r'\d+\.\s+([a-zA-Z0-9_-]+)', line.strip())
                    if matches:
                        video_id = matches.group(1)
                        # YouTube video IDs are usually at least 10 chars
                        if len(video_id) >= 10:
                            known_ids.add(video_id)
            
            print(f"Loaded {len(known_ids)} known video IDs from {subdirectories_file}")
        else:
            print(f"Warning: File not found {subdirectories_file}")
    except Exception as e:
        print(f"Error reading known video IDs: {e}")
    
    return known_ids

class YouTubeNewsCrawler:
    def __init__(self, api_keys, output_dir='downloaded_news', whisper_model='base',
                 openai_api_key=None, openai_model="gpt-4.1"):
        """Initialize YouTube API client, set output dir and Whisper model. Supports multiple API keys with automatic switching."""
        self.api_keys = api_keys
        self.api_key_index = 0
        
        self.logger = setup_logger(output_dir, f'youtube_crawler_{id(self)}')
        self.logger.info(f"Initialized YouTubeNewsCrawler: output_dir={output_dir}, Whisper model={whisper_model}")
        self._init_youtube_client()
        self.output_dir = output_dir
        self.whisper_model = whisper_model
        os.makedirs(self.output_dir, exist_ok=True)
        self.processed_ids_in_run = set()

        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

        self.known_video_ids = load_known_video_ids()
        self.logger.info(f"Loaded {len(self.known_video_ids)} known video IDs. These will be skipped.")

        self.max_workers = 16  # Adjustable depending on system

    def _init_youtube_client(self):
        current_key = self.api_keys[self.api_key_index]
        self.youtube = build('youtube', 'v3', developerKey=current_key)
        self.logger.info(f"Using API key: {current_key[:8]}... Initializing YouTube client")
        
    def check_video_has_subtitles(self, video_id):
        """
        Check if the video has English subtitles.
        Only accept original English subtitles, not auto-translated.
        """
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            ydl_opts = {
                'skip_download': True,
                'listsubtitles': True,
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                # Check manually added English subtitles
                subs = info.get('subtitles', {})
                has_manual_en_subs = 'en' in subs
                
                # Check auto-generated subtitles
                auto_subs = info.get('automatic_captions', {})
                
                # If there are auto subtitles, check for "en" (not "en-orig" or variants)
                # Original English auto subtitles are usually 'en' or 'en-orig'
                has_auto_en_subs = False
                if 'en' in auto_subs:
                    # Check if it's original English auto subtitles, not translated
                    has_auto_en_subs = True
                    
                    # Further check language keys
                    # If 'en-trans' or 'en.trans' exists, it's translated to English
                    for key in auto_subs.keys():
                        if 'en-trans' in key or 'en.trans' in key:
                            self.logger.info(f"  Detected translated English subtitles: {key}")
                            has_auto_en_subs = False
                            break
                
                # Log subtitle check results
                self.logger.info(f"  Video {video_id} subtitle check: manual English={has_manual_en_subs}, auto English={has_auto_en_subs}")
                
                # Also check main language of the video
                if 'lang' in info:
                    self.logger.info(f"  Video main language marked as: {info.get('lang')}")
                    if info.get('lang') and info.get('lang').lower() != 'en':
                        self.logger.warning(f"  Video main language is not English, but {info.get('lang')}")
                        
                # Print all available subtitle types for debugging
                self.logger.debug(f"  Available manual subtitles: {list(subs.keys())}")
                self.logger.debug(f"  Available auto subtitles: {list(auto_subs.keys())}")
                
                # At least one type of English subtitle
                return has_manual_en_subs or has_auto_en_subs
                
        except Exception as e:
            self.logger.error(f"Failed to check subtitles: {e}")
            return False
        
    def _parse_duration(self, duration_str):
        """Parse ISO 8601 duration string to seconds"""
        try:
            duration = isodate.parse_duration(duration_str)
            return duration.total_seconds()
        except Exception as e:
            self.logger.error(f"Error parsing duration '{duration_str}': {e}")
            return float('inf')

    def _is_reliable_english(self, text):
        """Enhanced English detection: use multiple detections and English word ratio for reliability"""
        if not text or len(text.strip()) < 20:
            return False
        words = text.split()
        if len(words) < 5:
            return False
        spaces_ratio = text.count(' ') / len(text) if len(text) > 0 else 0
        if spaces_ratio < 0.08:
             return False
        detections = []
        try:
            check_text = text[:500]  # 只检查前500个字符以提高性能
            for _ in range(2):
                lang = detect(check_text)
                detections.append(lang)
            return all(d == 'en' for d in detections)
        except LangDetectException:
            return False
        except Exception as e:
            return False

    def get_trending_news(self, date_str, max_total_results=1000000, retries=3, delay=5, max_duration_sec=600):
        """Get trending news videos for a given date, within duration and English title/description constraints"""
        all_search_items = []
        next_page_token = None
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        published_after = target_date.isoformat() + 'Z'
        published_before = (target_date + timedelta(days=1)).isoformat() + 'Z'
        results_per_page = 10  # 降低一次获取数量以便实时处理

        # 扩展视频分类列表，包含新闻(25)、娱乐(24)、教育(27)和科学技术(28)
        category_ids = ['25', '24', '27', '28']
        filtered_items = []

        self.logger.info(f"Starting video search for {date_str} (target: {max_total_results})...")

        # 对每个分类轮流搜索，避免单一分类搜索太多
        for category_id in category_ids:
            if len(filtered_items) >= max_total_results:
                self.logger.info(f"Reached target {max_total_results}, stopping search.")
                break

            self.logger.info(f"Searching category ID: {category_id}")
            category_next_page_token = None
            page_count = 0
            max_pages_per_category = 30  

            while page_count < max_pages_per_category and len(filtered_items) < max_total_results:
                page_count += 1
                attempt = 0
                success = False

                while attempt < retries and not success:
                    try:
                        self.logger.info(f"  Fetching page {page_count} (already filtered: {len(filtered_items)})...")
                        request = self.youtube.search().list(
                            part='snippet',
                            maxResults=results_per_page,
                            q='news OR education OR science OR technology',  # 扩展搜索关键词
                            type='video',
                            videoCategoryId=category_id,
                            order='relevance',  
                            publishedAfter=published_after,
                            publishedBefore=published_before,
                            relevanceLanguage='en',
                            pageToken=category_next_page_token
                        )
                        response = request.execute()
                        items = response.get('items', [])
                        category_next_page_token = response.get('nextPageToken')

                        if items:
                            self.logger.info(f"  Got {len(items)} items on this page, filtering for English...")
                            # Immediate language filtering
                            english_items = self._filter_english_items(items)
                            self.logger.info(f"  After language filter: {len(english_items)} items")

                            # Save metadata for each video passing language filter
                            for item in english_items:
                                video_id = item.get('id', {}).get('videoId')
                                title = item.get('snippet', {}).get('title', '')
                                video_folder_path = os.path.join(self.output_dir, video_id)
                                metadata_path = os.path.join(video_folder_path, f"{video_id}_metadata.json")
                                if os.path.exists(metadata_path):
                                    self.logger.info(f"  Metadata exists, skipping: {metadata_path}")
                                    continue
                                os.makedirs(video_folder_path, exist_ok=True)
                                self.save_enhanced_metadata(video_id, title, video_folder_path, date_str, item)

                            if english_items:
                                duration_filtered = self._filter_by_duration(english_items, max_duration_sec)
                                filtered_items.extend(duration_filtered)
                                self.logger.info(f"  After duration filter on this page: {len(duration_filtered)} kept, total {len(filtered_items)}")

                        success = True

                    except HttpError as e:
                        self.logger.error(f"  API search HTTP error (attempt {attempt + 1}/{retries}): {e}")
                        # Check for quota exceeded, switch API key
                        if e.resp.status == 403:
                            self.logger.warning(f"API Key {self.api_keys[self.api_key_index][:8]}... returned 403, switching to next key")
                            self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
                            self._init_youtube_client()
                            attempt += 1
                            time.sleep(delay)
                            continue
                        time.sleep(delay * (attempt + 1))
                        attempt += 1
                    except OSError as e:
                        self.logger.error(f"  Network error (attempt {attempt + 1}/{retries}): {e}")
                        time.sleep(delay * (attempt + 1))
                        attempt += 1
                    except Exception as e:
                        self.logger.error(f"  Unknown error getting video list (attempt {attempt + 1}/{retries}): {str(e)}")
                        time.sleep(delay * (attempt + 1))
                        attempt += 1

                if not success:
                    self.logger.warning(f"  Failed to get current page after {retries} attempts. Skipping to next category.")
                    break

                if not category_next_page_token or len(filtered_items) >= max_total_results:
                    self.logger.info("  Reached last page or target quantity.")
                    break

                # Short pause between pages
                time.sleep(1)

        self.logger.info(f"Search complete. Found {len(filtered_items)} videos matching criteria.")
        return filtered_items

    def _filter_english_items(self, items):
        """Filter items with English title/description and exclude known IDs"""
        english_items = []
        unique_ids = set()
        
        for item in items:
            video_id = item.get('id', {}).get('videoId')
            if not video_id or video_id in unique_ids:
                continue
                
            # Skip if known video ID
            if video_id in self.known_video_ids:
                self.logger.info(f"  Skipping known video ID: {video_id}")
                continue
                
            unique_ids.add(video_id)
            snippet = item.get('snippet', {})
            title = snippet.get('title', '')
            
            # Check if title is English (required)
            if not self._is_english_title(title):
                self.logger.info(f"  Title is not English, skipping video: {title}")
                continue
                
            description = snippet.get('description', '')
            channel_title = snippet.get('channelTitle', '')
            text_to_check = f"{title} {description} {channel_title}"
            
            if not text_to_check.strip():
                continue
                
            if self._is_reliable_english(text_to_check):
                english_items.append(item)
        
        return english_items
        
    def _filter_by_duration(self, items, max_duration_sec):
        """Filter video items by duration"""
        filtered_items = []
        video_ids = [item['id']['videoId'] for item in items if 'id' in item and 'videoId' in item['id']]

        if not video_ids:
            return []

        attempt = 0
        retries = 3
        delay = 5
        while attempt < retries:
            try:
                details_request = self.youtube.videos().list(
                    part='contentDetails,snippet,statistics',  
                    id=','.join(video_ids)
                )
                details_response = details_request.execute()
                details_map = {item['id']: item for item in details_response.get('items', [])}

                for item in items:
                    video_id = item.get('id', {}).get('videoId')
                    if video_id in details_map:
                        video_details = details_map[video_id]
                        duration_str = video_details.get('contentDetails', {}).get('duration')

                        if duration_str:
                            duration_sec = self._parse_duration(duration_str)
                            if duration_sec <= max_duration_sec:
                                item['video_details'] = video_details
                                filtered_items.append(item)
                break
            except HttpError as e:
                # Check for quota exceeded, auto switch API key
                if e.resp.status == 403 and 'quota' in str(e).lower():
                    self.logger.warning(f"API Key {self.api_keys[self.api_key_index][:8]}... quota exceeded, switching to next key")
                    self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
                    self._init_youtube_client()
                    attempt += 1
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"Error filtering by duration: {e}")
                    break
            except Exception as e:
                self.logger.error(f"Error filtering by duration: {e}")
                break

        return filtered_items

    def _check_video_exists_in_other_dir(self, video_id, other_dir=''):
        """Check if the video already exists in another directory"""
        if not os.path.exists(other_dir):
            return False
            
        video_dir = os.path.join(other_dir, video_id)
        if os.path.exists(video_dir):
            
            metadata_json = os.path.join(video_dir, f"{video_id}_metadata.json")
            if os.path.exists(metadata_json):
                return True
        
        return False

    def save_enhanced_metadata(self, video_id, title, video_folder_path, date_str, video_info):
        """
        Save detailed video metadata to a JSON file, including title, URL, source, ID, publish date, author, thumbnail URL, etc.
        """
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            
            snippet = video_info.get('snippet', {})
            channel_title = snippet.get('channelTitle', 'Unknown')
            description = snippet.get('description', '')
            published_at = snippet.get('publishedAt', date_str + 'T00:00:00Z')
            thumbnails = snippet.get('thumbnails', {})
            thumbnail_url = thumbnails.get('high', {}).get('url', '')
            
            statistics = video_info.get('statistics', {})
            view_count = statistics.get('viewCount', 'N/A')
            like_count = statistics.get('Like_count', 'N/A')
            comment_count = statistics.get('commentCount', 'N/A')
        
            try:
                published_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                formatted_date = published_dt.strftime("%Y-%m-%d")
                formatted_time = published_dt.strftime("%H:%M:%S")
            except:
                formatted_date = date_str
                formatted_time = "00:00:00"
            
            metadata = {
                "topic": title,
                "url": video_url,
                "source": "YouTube",
                "id": video_id,
                "author": channel_title,
                "publication_date": formatted_date,
                "publication_time": formatted_time,
                "description": description,
                "thumbnail_url": thumbnail_url,
                "view_count": view_count,
                "like_count": like_count,
                "comment_count": comment_count
            }
            
            # JSON 文件路径
            json_path = os.path.join(video_folder_path, f"{video_id}_metadata.json")
            
            # 写入 JSON 文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"  ✅ Enhanced metadata saved to JSON: {os.path.basename(json_path)}")
            return True
        except Exception as e:
            self.logger.error(f"  ❌ Error saving enhanced metadata JSON: {e}")
            return False

    def download_videos_parallel(self, videos_to_download):
        """
        Download multiple videos in parallel.
        
        Args:
            videos_to_download: List of (video_id, title) tuples to download.
            
        Returns:
            list: List of (video_path, vtt_path) tuples for each video.
        """
        results = []
        
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            future_to_video = {
                executor.submit(self.download_video, video_id, title): (video_id, title)
                for video_id, title in videos_to_download
            }
            
            for future in concurrent.futures.as_completed(future_to_video):
                video_id, title = future_to_video[future]
                try:
                    video_path, vtt_path = future.result()
                    results.append((video_id, title, (video_path, vtt_path)))
                    self.logger.info(f"Parallel download complete: {video_id} - {title}")
                except Exception as e:
                    self.logger.error(f"Parallel download failed for video {video_id}: {e}")
                    results.append((video_id, title, (None, None)))
        
        return [(r[2][0], r[2][1]) for r in results]

    def crawl_news(self, date_str, max_videos=1000000):
        """
        Crawl videos for the given date. Download and process depending on subtitle availability.
        Returns the number of successfully processed videos.
        """
        self.logger.info(f"Starting to process trending news for {date_str} (duration ≤10 minutes)...")
        self.logger.info(f"Root output directory: {self.output_dir}")
        self.processed_ids_in_run.clear()

        candidate_videos = self.get_trending_news(date_str, max_total_results=max_videos * 5)

        if not candidate_videos:
            self.logger.warning("Error: No candidate videos found.")
            return 0

        self.logger.info(f"Found {len(candidate_videos)} candidate videos, starting download and processing...")

        processed_count = 0
        skipped_count = 0
        video_counter = 0  
        known_id_count = 0 

        videos_to_download = []

        for video_info in candidate_videos:
            if processed_count >= max_videos:
                self.logger.info(f"Reached target processing count ({max_videos}).")
                break

            video_id = video_info.get('id', {}).get('videoId')
            if not video_id:
                continue

            if video_id in self.known_video_ids:
                self.logger.info(f"  Skipping: {video_id}")
                known_id_count += 1
                continue

            title = video_info.get('snippet', {}).get('title', 'N/A')
            self.logger.info(f"\n[{video_counter + skipped_count + known_id_count + 1}/{len(candidate_videos)}] Processing: {title} (ID: {video_id})")

            video_folder_path = os.path.join(self.output_dir, video_id)
            output_txt_path = os.path.join(video_folder_path, f"{video_id}.txt")  # Whisper 输出
            output_paragraphs_txt_path = os.path.join(video_folder_path, f"{video_id}.paragraphs.txt")  # VTT 转换输出

            if os.path.exists(output_txt_path) or os.path.exists(output_paragraphs_txt_path):
                self.logger.info(f"  Skipping (output TXT file exists): {video_id}")
                skipped_count += 1
                self.processed_ids_in_run.add(video_id)
                continue
            elif video_id in self.processed_ids_in_run:
                self.logger.info(f"  Skipping (already processed in this run): {video_id}")
                skipped_count += 1
                continue
            
            videos_to_download.append((video_id, title))
            self.processed_ids_in_run.add(video_id)  

        if videos_to_download:
            self.logger.info(f"Starting parallel download of {len(videos_to_download)} videos...")
            download_results = self.download_videos_parallel(videos_to_download)
            

            for (video_id, title), (video_path, vtt_path) in zip(videos_to_download, download_results):
                
                final_video_path = None
                if video_path:
                    
                    if self._is_vp9(video_path):
                        self.logger.info(f"  Detected VP9 encoding, starting transcoding...")
                        h264_path = self._convert_to_h264(video_path)
                        if h264_path:
                            final_video_path = h264_path
                        else:
                            self.logger.error(f"  VP9 transcoding failed, cannot process video: {video_id}")
                            skipped_count += 1
                            continue 
                    else:
                        final_video_path = video_path  
                else:
                    
                    self.logger.error(f"  Video download failed, cannot process: {video_id}")
                    skipped_count += 1
                    continue

                video_folder_path = os.path.join(self.output_dir, video_id)
                metadata_json_path = os.path.join(video_folder_path, f"{video_id}_metadata.json")
                if not os.path.exists(metadata_json_path):
                    self.save_video_metadata_json(video_id, title, video_folder_path)
                else:
                    self.logger.info(f"  Metadata file exists, skipping save: {metadata_json_path}")
                    
                video_counter += 1
                if video_counter % 5 == 0:
                    self.logger.info(f"  Successfully processed videos: {video_counter}")

        self.logger.info("\n--------------------")
        self.logger.info("Processing complete.")
        self.logger.info(f"Total candidate videos checked: {len(candidate_videos)}")
        self.logger.info(f"Skipped videos (already exists/download/transcode/transcription failed): {skipped_count}")
        self.logger.info(f"Skipped known video IDs: {known_id_count}")
        self.logger.info("--------------------")

        return video_counter  # Return number of successfully processed videos

    def _is_vp9(self, video_path):
        """Check if video is VP9 encoded"""
        try:
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            codec = result.stdout.strip()
            return codec == "vp9"
        except subprocess.CalledProcessError as e:
            return False
        except Exception as e:
            self.logger.warning(f"Failed to detect video codec: {e}")
            return False

    def _convert_to_h264(self, video_path):
        """Transcode VP9 video to H.264 using ffmpeg and save in same directory"""
        try:
            video_dir = os.path.dirname(video_path)
            base_filename, ext = os.path.splitext(os.path.basename(video_path))
            out_filename = f"{base_filename}_h264{ext}"
            out_path = os.path.join(video_dir, out_filename)

            if os.path.abspath(video_path) == os.path.abspath(out_path):
                self.logger.error(f"Error: Transcode input and output paths are the same, skipping: {video_path}")
                return None

            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-strict", "-2",
                out_path
            ]
            self.logger.info(f"Transcoding to H.264: {os.path.basename(video_path)} -> {out_filename}")

            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.logger.info(f"Transcoding complete: {out_filename}")

            try:
                os.remove(video_path)
                self.logger.info(f"  Deleted original VP9 file: {os.path.basename(video_path)}")
            except OSError as del_e:
                self.logger.warning(f"  Failed to delete original VP9 file: {del_e}")
            return out_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Transcoding failed: {e}")
            self.logger.error(f"ffmpeg stderr: {e.stderr}")
            if os.path.exists(out_path):
                try: os.remove(out_path)
                except OSError: pass
            return None
        except Exception as e:
            self.logger.error(f"Unknown error during transcoding: {e}")
            return None

    def download_video(self, video_id, title, timeout=300):  # 添加超时参数，默认 5 分钟
        """
        Download the YouTube video and subtitles for the given video_id to self.output_dir/video_id/.
        Returns (video_path, vtt_path). If download fails, corresponding path is None.
        
        Args:
            video_id: YouTube video ID
            title: Video title (for logging)
            timeout: Download timeout in seconds
        """
        video_path = None
        vtt_path = None
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            output_dir = os.path.join(self.output_dir, video_id)
            os.makedirs(output_dir, exist_ok=True)
            # Use specific names for later checks
            output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
            expected_mp4_path = os.path.join(output_dir, f"{video_id}.mp4")
            expected_h264_path = os.path.join(output_dir, f"{video_id}_h264.mp4") 
            expected_vtt_path = os.path.join(output_dir, f"{video_id}.en.vtt")  
            
            # Check if video already downloaded (including transcoded H264)
            video_exists = False
            video_file_path = None
            
            # Check H264 version first (preferred)
            if os.path.exists(expected_h264_path):
                file_size = os.path.getsize(expected_h264_path)
                if file_size > 1024*1024:  # 大于1MB的文件视为有效文件
                    self.logger.info(f"  H264 video file already exists, skipping download: {os.path.basename(expected_h264_path)}")
                    video_path = expected_h264_path
                    video_exists = True
            
            # If no H264 version, check original file
            if not video_exists and os.path.exists(expected_mp4_path):
                file_size = os.path.getsize(expected_mp4_path)
                if file_size > 1024*1024:  # 大于1MB的文件视为有效文件
                    self.logger.info(f"  Video file already exists, skipping download: {os.path.basename(expected_mp4_path)}")
                    video_path = expected_mp4_path
                    video_exists = True
                else:
                    # 文件存在但可能不完整，尝试删除后重新下载
                    self.logger.warning(f"  Existing video file too small ({file_size} bytes), may be incomplete, will re-download")
                    try:
                        os.remove(expected_mp4_path)
                    except OSError as e:
                        self.logger.warning(f"  Failed to delete incomplete video file: {e}")
            
            # If video file exists, check if subtitle file also exists
            if video_exists and os.path.exists(expected_vtt_path):
                self.logger.info(f"  Subtitle file also exists: {os.path.basename(expected_vtt_path)}")
                vtt_path = expected_vtt_path
                return video_path, vtt_path
                
            # If old VTT exists but no valid video, delete to ensure fresh check
            if not video_exists and os.path.exists(expected_vtt_path):
                try: os.remove(expected_vtt_path)
                except OSError: pass

            ydl_opts = {
                'outtmpl': output_template,
                'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/mp4[height<=720]',
                'merge_output_format': 'mp4',
                'quiet': True,
                'no_warnings': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'subtitlesformat': 'vtt',
                'ignoreerrors': True,  
                'retries': 2,
                'socket_timeout': 30,  
                'extract_flat': 'in_playlist',  
            }
            self.logger.info(f"  Attempting to download video and subtitles: {video_id}")
            
            
            import threading
            import signal
            
            # Create an event flag to signal download completion
            download_complete = threading.Event()
            download_success = [False]  
            
            # Define download function to run in a separate thread
            def download_thread():
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        error_code = ydl.download([video_url])
                        # Only treat as success if error_code is 0
                        download_success[0] = (error_code == 0)
                except Exception as e:
                    self.logger.error(f"    Exception occurred in download thread: {e}")
                    download_success[0] = False
                finally:
                    # Set completion event regardless of success/failure
                    download_complete.set()
            
            # Start download thread
            thread = threading.Thread(target=download_thread)
            thread.daemon = True  
            thread.start()
            
            # Wait for download to complete or timeout
            if download_complete.wait(timeout):
                self.logger.info("    Download process completed")
            else:
                self.logger.warning(f"    ⚠️ Warning: Download timed out ({timeout}s), continuing")
            
            # After download attempt, check if files exist
            if os.path.exists(expected_mp4_path):
                video_path = expected_mp4_path
                self.logger.info(f"    Video file found: {os.path.basename(video_path)}")
            else:
                self.logger.warning(f"    Video file not found or download failed: {expected_mp4_path}")

            if os.path.exists(expected_vtt_path):
                vtt_path = expected_vtt_path
                self.logger.info(f"    VTT subtitle file found: {os.path.basename(vtt_path)}")
            else:
                self.logger.warning(f"    VTT subtitle file not found: {expected_vtt_path}")

            return video_path, vtt_path

        except Exception as e:
            self.logger.error(f"  Exception occurred during video/subtitle download ({video_id}): {e}")
            return None, None  # Return None on exception

    def _is_english_title(self, title):
        """
        Determine if a title is English.
        After removing emoji, &, |, and other special chars, require at least 90% English characters.
        """
        if not title:
            return False  # Empty title is not considered English
            
        self.logger.info(f"Checking if title is English: '{title}'")
        
        import re
        
        # Remove emoji
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251" 
                                   "]+", flags=re.UNICODE)
                                   
        # Remove special characters
        special_chars = r'[&|~!@#$%^*()_+={}\[\]:;"\'<>,.?/\\]'
        
        # Remove emoji then special chars
        clean_title = emoji_pattern.sub(r'', title)
        clean_title = re.sub(special_chars, '', clean_title)
        
        if not clean_title.strip():
            self.logger.info("  Title is empty after cleaning, not English")
            return False
            
        # Calculate ratio of English chars (letters, digits, spaces)
        english_chars = sum(1 for c in clean_title if c.isalpha() or c.isdigit() or c.isspace())
        total_chars = len(clean_title)
        
        english_ratio = english_chars / total_chars if total_chars > 0 else 0
        
        self.logger.info(f"  Cleaned title: '{clean_title}'")
        self.logger.info(f"  English char ratio: {english_ratio:.2f} ({english_chars}/{total_chars})")
        
        is_english = english_ratio >= 0.9
        self.logger.info(f"  Result: {'English' if is_english else 'Not English'} title")
        return is_english

    def save_video_metadata_json(self, video_id, title, video_folder_path):
        """
        Save video metadata to JSON file.
        Title is already confirmed English.
        Save as much metadata as possible.
        """
        try:
            # 获取更详细的视频信息，支持API key自动切换
            video_info = {}
            attempt = 0
            retries = 3
            delay = 5
            while attempt < retries:
                try:
                    video_info_request = self.youtube.videos().list(
                        part='snippet,contentDetails,statistics',
                        id=video_id
                    )
                    video_info_response = video_info_request.execute()
                    video_items = video_info_response.get('items', [])
                    video_info = video_items[0] if video_items else {}
                    break
                except HttpError as e:
                    if e.resp.status == 403 and 'quota' in str(e).lower():
                        self.logger.warning(f"API Key {self.api_keys[self.api_key_index][:8]}... quota exceeded, switching to next key")
                        self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
                        self._init_youtube_client()
                        attempt += 1
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.warning(f"  Unable to get detailed video info: {e}")
                        break
                except Exception as e:
                    self.logger.warning(f"  Unable to get detailed video info: {e}")
                    break

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            snippet = video_info.get('snippet', {})
            content_details = video_info.get('contentDetails', {})
            statistics = video_info.get('statistics', {})
            
            channel_id = snippet.get('channelId', '')
            channel_title = snippet.get('channelTitle', '')
            published_at = snippet.get('publishedAt', '')
            description = snippet.get('description', '')
            tags = snippet.get('tags', [])
            category_id = snippet.get('categoryId', '')
            thumbnails = snippet.get('thumbnails', {})
            
            duration = content_details.get('duration', '')
            dimension = content_details.get('dimension', '')
            definition = content_details.get('definition', '')
            caption = content_details.get('caption', 'false')
            
            view_count = statistics.get('viewCount', '0')
            like_count = statistics.get('likeCount', '0')
            comment_count = statistics.get('commentCount', '0')
            favorite_count = statistics.get('favoriteCount', '0')
            
            formatted_date = ''
            formatted_time = ''
            if published_at:
                try:
                    dt = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
                    formatted_date = dt.strftime('%Y-%m-%d')
                    formatted_time = dt.strftime('%H:%M:%S')
                except Exception as e:
                    self.logger.warning(f"  Date formatting failed: {e}")
            
            duration_seconds = 0
            if duration:
                try:
                    duration_seconds = self._parse_duration(duration)
                except Exception:
                    pass
            
            thumbnail_url = ''
            for quality in ['maxres', 'standard', 'high', 'medium', 'default']:
                if (quality in thumbnails) and (thumbnails[quality].get('url', '')):
                    thumbnail_url = thumbnails[quality].get('url', '')
                    break
            
            metadata = {
                "topic": title,
                "url": video_url,
                "source": "YouTube",
                "id": video_id,
                "channel": {
                    "id": channel_id,
                    "title": channel_title,
                    "url": f"https://www.youtube.com/channel/{channel_id}" if channel_id else ""
                },
                "publication": {
                    "date": formatted_date,
                    "time": formatted_time,
                    "raw": published_at
                },
                "content": {
                    "description": description,
                    "tags": tags,
                    "category_id": category_id,
                    "duration": {
                        "iso": duration,
                        "seconds": duration_seconds
                    },
                    "dimension": dimension,
                    "definition": definition,
                    "has_caption": caption == 'true'
                },
                "statistics": {
                    "views": view_count,
                    "likes": like_count,
                    "comments": comment_count,
                    "favorites": favorite_count
                },
                "thumbnails": {
                    "url": thumbnail_url
                },
                "metadata_created": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            
            json_path = os.path.join(video_folder_path, f"{video_id}_metadata.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"  ✅ Rich metadata saved to JSON: {os.path.basename(json_path)}")
            return True
        except Exception as e:
            self.logger.error(f"  ❌ Error saving metadata JSON: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download YouTube news videos for a date range and process subtitles/transcription automatically.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Root directory for downloading videos and subtitles.")
    parser.add_argument("--start_date", type=str, required=True,
                        help="Start date for crawling (format: YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True,
                        help="End date for crawling (format: YYYY-MM-DD).")
    parser.add_argument("--max_videos_per_day", type=int, default=1000000,
                        help="Maximum number of videos to download per day.")
    parser.add_argument("--whisper_model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model size to use for transcription (default: 'base').")

    args = parser.parse_args()

    main_logger = setup_logger(args.output_dir)
    main_logger.info("Program started")

    api_keys_list = [
        # Input your YouTube API keys here
    ]

    if not api_keys_list or api_keys_list[0] == 'YOUR_API_KEY_HERE':
        main_logger.error("Warning: Please provide a valid YouTube API key (via --api_key/--api_keys or YOUTUBE_API_KEY environment variable).")
        return

    try:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
        if start_dt > end_dt:
            main_logger.error("Error: Start date cannot be later than end date.")
            return
    except ValueError:
        main_logger.error("Error: Invalid date format, please use YYYY-MM-DD.")
        return

    date_range = [start_dt + timedelta(days=x) for x in range((end_dt - start_dt).days + 1)]

    crawler = YouTubeNewsCrawler(api_keys=api_keys_list, output_dir=args.output_dir,
                                 whisper_model=args.whisper_model,
                                 openai_api_key=args.openai_api_key,
                                 openai_model=args.openai_model)

    main_logger.info(f"Starting processing date range: {args.start_date} to {args.end_date}")
    total_downloaded_all_days = 0
    for current_dt in date_range:
        current_date_str = current_dt.strftime("%Y-%m-%d")
        main_logger.info(f"\n{'='*20} Processing date: {current_date_str} {'='*20}")
        downloaded_today = crawler.crawl_news(current_date_str, max_videos=args.max_videos_per_day)
        total_downloaded_all_days += downloaded_today
        main_logger.info(f"Date {current_date_str} done, successfully processed {downloaded_today} videos today.")
        time.sleep(5)  # Small pause after each day's processing

    main_logger.info(f"\nAll specified dates processed. Total successfully processed videos: {total_downloaded_all_days}")
    main_logger.info("Program finished normally")

if __name__ == '__main__':
    main()