"""
Creates the required directory structure for the project
"""
import os
from collectors.config import DATA_DIR, IMG_DIR, WORKSPACE

def create_project_directories():
    """
    Creates all necessary directories for the project
    """
    directories = [
        os.path.join(WORKSPACE, "logs"),
        DATA_DIR,
        IMG_DIR
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            
    return True

if __name__ == "__main__":
    create_project_directories()
