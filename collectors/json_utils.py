"""
JSON data processing utilities, providing safe JSON read/write operations
"""
import os
import json
import shutil
import tempfile
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

def safe_read_json(file_path, default=None):
    """
    Safely reads a JSON file
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if reading fails
        
    Returns:
        JSON data or the default value
    """
    if default is None:
        default = []
        
    if not os.path.exists(file_path):
        return default
    
    # Get file size to check if it's an empty file
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"File is empty: {file_path}")
            return default
    except Exception as e:
        logger.error(f"Failed to check file size: {e}")
    
    # Try to read the original file directly
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Verify if JSON file format is correct (must start with [ and end with ])
            if not (content.startswith('[') and content.endswith(']')):
                logger.warning(f"JSON format error: {file_path}, attempting to repair")
                content = fix_json_array_format(content)
                
            # Attempt to parse JSON
            try:
                data = json.loads(content)
                
                # Ensure the result is a list
                if not isinstance(data, list):
                    logger.warning(f"JSON data is not in list format: {file_path}")
                    return default if isinstance(default, list) else []
                return data
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error {file_path}: {e}")
                # Continue to try other recovery methods
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
    
    # Attempt to restore from backup
    backup_path = file_path + ".bak"
    if os.path.exists(backup_path):
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Validate backup file format
                if not (content.startswith('[') and content.endswith(']')):
                    logger.warning(f"Backup file JSON format error: {backup_path}, attempting to repair")
                    content = fix_json_array_format(content)
                    
                data = json.loads(content)
                logger.info(f"Successfully restored data from backup: {backup_path}")
                return data if isinstance(data, list) else []
        except Exception as backup_error:
            logger.error(f"Failed to attempt restore from backup: {backup_error}")
    
    # If the file exists but cannot be parsed, try to manually repair
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Create backup of corrupted file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrupt_backup = f"{file_path}.corrupt.{timestamp}"
        with open(corrupt_backup, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created backup of corrupted file: {corrupt_backup}")
        
        # Attempt to fix JSON format
        fixed_content = fix_json_array_format(content)
            
        # Attempt to load fixed content
        try:
            data = json.loads(fixed_content)
            # Ensure the result is a list
            if not isinstance(data, list):
                return default if isinstance(default, list) else []
                
            # If successfully repaired, save back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully repaired JSON file: {file_path}")
            return data
        except Exception as repair_error:
            logger.error(f"Failed to attempt JSON repair: {repair_error}")
    except Exception as e:
        logger.error(f"Failed to read file content: {e}")
    
    return default

def fix_json_array_format(content):
    """
    Fixes JSON array format, ensuring it's a valid array format
    
    Args:
        content: JSON text content
        
    Returns:
        Repaired JSON text content
    """
    # Remove leading/trailing whitespace
    content = content.strip()
    
    # If content is empty, return empty array
    if not content:
        return '[]'
    
    # Find the first valid JSON array (from opening [ to corresponding closing ])
    if content.startswith('['):
        # Find matching closing bracket
        bracket_count = 0
        end_pos = -1
        
        for i, char in enumerate(content):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_pos = i
                    break
        
        if end_pos >= 0:
            # Extract valid JSON array portion
            content = content[:end_pos+1]
        else:
            # No matching closing bracket found, manually add
            content = content + ']'
    else:
        # Does not start with [, try to find a valid JSON object and wrap it in []
        if '{' in content:
            # Might be a JSON object, try wrapping as an array
            content = '[' + content + ']'
            # Handle potential extra text
            content = re.sub(r'\]\s*\S+.*$', ']', content)
        else:
            # Unrecognized format, return empty array
            content = '[]'
    
    # Replace common issues
    content = content.replace('null', '""')
    content = content.replace('undefined', '""')
    
    # Final check and ensure it's array format
    if not content.startswith('['):
        content = '[' + content
    if not content.endswith(']'):
        content = content + ']'
    
    # Try to parse the fixed content to validate
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        # If still invalid, return empty array
        logger.error("Failed to fix JSON format, returning empty array")
        return '[]'

def safe_write_json(file_path, data):
    """
    Safely writes a JSON file, using a temporary file and atomic operation
    
    Args:
        file_path: Path to the JSON file
        data: Data to be saved
        
    Returns:
        bool: True if saved successfully
    """
    # Ensure data is a list
    if not isinstance(data, list):
        logger.error(f"Attempting to write non-list data: {type(data)}")
        return False
        
    # If data is empty, log warning but proceed
    if len(data) == 0:
        logger.warning(f"Attempting to write empty list data to {file_path}")
    
    # Check existing file
    original_data = []
    original_count = 0
    if os.path.exists(file_path):
        try:
            original_data = safe_read_json(file_path, [])
            original_count = len(original_data)
            
            # If the data to be written is significantly less than original, it might be data loss
            if len(data) < original_count * 0.9:  # If new data is less than 90% of original
                logger.warning(f"Warning: New data ({len(data)} records) decreased by more than 10% compared to original ({original_count} records)")
        except Exception as e:
            logger.error(f"Failed to check existing data: {e}")
    
    # Create backup before writing
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            backup_path = file_path + ".bak"
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup file: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    
    # Create temporary file and write
    tmp_name = None
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write using a temporary file
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.json') as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp_name = tmp.name
        
        # Verify if temporary file is readable and content is complete
        try:
            with open(tmp_name, 'r', encoding='utf-8') as f:
                temp_content = f.read()
                if not (temp_content.startswith('[') and temp_content.endswith(']')):
                    logger.error(f"Temporary file format is incorrect")
                    return False
                    
                temp_data = json.loads(temp_content)
                if len(temp_data) < len(data):
                    logger.error(f"Temporary file content incomplete: Expected {len(data)} records, got {len(temp_data)} records")
                    return False
        except Exception as read_error:
            logger.error(f"Failed to verify temporary file: {read_error}")
            return False
            
        # Atomically replace original file
        shutil.move(tmp_name, file_path)
        
        # Final verification
        if not verify_json_integrity(file_path, min_expected_items=len(data)):
            logger.error(f"Post-write verification failed: {file_path}")
            # Attempt to restore from backup
            backup_path = file_path + ".bak"
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, file_path)
                    logger.info(f"Restored data from backup")
                except Exception as recovery_error:
                    logger.error(f"Failed to recover from backup: {recovery_error}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Failed to write JSON file {file_path}: {e}")
        # Clean up temporary file
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file: {cleanup_error}")
        
        # Attempt to restore from backup
        backup_path = file_path + ".bak"
        if os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, file_path)
                logger.info(f"Restored data from backup")
            except Exception as recovery_error:
                logger.error(f"Failed to recover from backup: {recovery_error}")
        
        return False

def append_to_json_array(file_path, new_item):
    """
    Safely appends a new item to a JSON array file (without breaking format)
    
    Args:
        file_path: Path to the JSON file
        new_item: New item to append
        
    Returns:
        bool: True if appended successfully
    """
    # Read existing data
    data = safe_read_json(file_path, default=[])
    
    # Ensure data is a list
    if not isinstance(data, list):
        data = []
    
    # Add new item and save
    cleaned_item = clean_json_data(new_item)
    
    # Check if an entry with the same URL already exists
    found = False
    for idx, item in enumerate(data):
        if isinstance(item, dict) and item.get('url') == cleaned_item.get('url'):
            data[idx] = cleaned_item
            found = True
            break
    
    if not found:
        data.append(cleaned_item)
    
    # Save updated data
    return safe_write_json(file_path, data)

def verify_json_integrity(file_path, min_expected_items=None):
    """
    Verifies the integrity of a JSON file
    
    Args:
        file_path: Path to the JSON file
        min_expected_items: Minimum expected number of items, no check if None
        
    Returns:
        bool: True if the file is complete and valid
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Verify if JSON file format is correct (must start with [ and end with ])
            if not (content.startswith('[') and content.endswith(']')):
                logger.error(f"JSON format error: {file_path}")
                return False
                
            try:
                data = json.loads(content)
                
                # Ensure it's a list
                if not isinstance(data, list):
                    logger.error(f"JSON data is not in list format: {file_path}")
                    return False
                    
                # Check minimum number of items
                if min_expected_items is not None and len(data) < min_expected_items:
                    logger.error(f"JSON data item count insufficient: Expected at least {min_expected_items}, got {len(data)}")
                    return False
                    
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format: {e}")
                return False
    except Exception as e:
        logger.error(f"Failed to verify JSON file: {e}")
        return False

def clean_json_data(data):
    """
    Cleans values in JSON data that might cause serialization issues
    
    Args:
        data: Data to clean
        
    Returns:
        Cleaned data
    """
    if data is None:
        return ""
    elif isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items() if k is not None}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data if item is not None]
    elif isinstance(data, (str, int, float, bool)):
        return data
    else:
        # Convert other types to string
        return str(data)

def repair_json_file(file_path):
    """
    Attempts to repair a corrupted JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        bool: True if successfully repaired
    """
    if not os.path.exists(file_path):
        return False
        
    try:
        # Create file backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repair_backup = f"{file_path}.before_repair.{timestamp}"
        shutil.copy2(file_path, repair_backup)
        logger.info(f"Created pre-repair backup: {repair_backup}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Attempt to fix JSON content
        fixed_content = fix_json_array_format(content)
        
        try:
            # Verify if fixed content is valid JSON
            data = json.loads(fixed_content)
            
            # Ensure it's a list type
            if not isinstance(data, list):
                data = []
                fixed_content = '[]'
            
            # Save fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
                
            logger.info(f"Successfully repaired JSON file: {file_path}")
            return True
        except json.JSONDecodeError:
            # Repair failed, try to restore from backup
            backup_path = file_path + ".bak"
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, file_path)
                    logger.info(f"Repair failed, restored from backup: {backup_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to restore from backup: {e}")
            
            # If unable to repair and no valid backup, create a new empty JSON array
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('[]')
            logger.warning(f"Unable to repair JSON, created empty array file")
            return False
            
    except Exception as e:
        logger.error(f"Failed to repair JSON file: {e}")
        return False
