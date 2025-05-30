"""
Image processing utility module, including image encoding and processing functions
"""

import os
import base64
import logging

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path):
    """
    Encodes an image to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image, or None if an error occurs.
    """
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image file does not exist: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error {image_path}: {str(e)}")
        return None
