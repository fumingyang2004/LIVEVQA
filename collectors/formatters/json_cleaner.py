import json

def remove_duplicates_from_json(file_path):
    """
    Removes duplicate articles from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        bool: True if processed successfully, False otherwise.
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return False
        
        # If not in list format, cannot process
        if not isinstance(data, list):
            print(f"File format error: {file_path} is not a valid topic list")
            return False
        
        # Deduplication logic
        unique_topics = []
        seen_urls = set()
        seen_titles = set()
        
        for item in data:
            if not isinstance(item, dict):
                continue
                
            url = item.get('url', '')
            title = item.get('topic', '')
            
            # Use URL and title as unique identifiers
            if url and url in seen_urls:
                continue
            
            if title and title in seen_titles:
                continue
            
            # Add to results and seen sets
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)
                
            unique_topics.append(item)
        
        # If duplicates found, save the modified file
        if len(unique_topics) < len(data):
            # Create backup
            backup_path = file_path + ".duplicate.bak"
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f"Backup created: {backup_path}")
            except Exception as e:
                print(f"Failed to create backup: {e}")
            
            # Save deduplicated data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(unique_topics, f, ensure_ascii=False, indent=2)
                
            print(f"Removed duplicates from {len(data)} records, kept {len(unique_topics)} unique records")
            return True
        else:
            print("No duplicate records found")
            return True
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return False
