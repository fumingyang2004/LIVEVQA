"""Date-related utility functions"""
from datetime import datetime, timedelta
import re

def get_current_timestamp():
    """Gets the current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_mmddhhmi_timestamp():
    """
    Gets a timestamp in a specific format: MMDDHHMI
    For example: 04191710 means April 19th, 17:10
    MM: Month (01-12)
    DD: Day (01-31)
    HH: Hour (00-23)
    MI: Minute (00-59)
    """
    return datetime.now().strftime("%m%d%H%M")
