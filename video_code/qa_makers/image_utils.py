
import os
import base64
import logging

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path):
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in encoding image {image_path}: {str(e)}")
        return None
