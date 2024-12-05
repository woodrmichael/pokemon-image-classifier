import os
from PIL import Image

def remove_invalid_images(directory):
    """
    Removes invalid or truncated PNG and JPEG images from a directory (including subdirectories).
    
    Parameters:
    directory (str): The root directory containing images to validate and clean.
    """
    
    count = 0
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            file_ext = filename.lower().endswith(('.png', '.jpg', '.jpeg'))

            if not file_ext:
                # Skip non-image files
                continue
            
            try:
                # Check if the file is larger than a minimal valid size
                min_size = 33 if filename.lower().endswith('.png') else 2
                if os.path.getsize(file_path) < min_size:
                    print(f"Deleting file (too small to be valid): {file_path}")
                    os.remove(file_path)
                    continue
                
                # Attempt to open and verify the image
                with Image.open(file_path) as img:
                    img.verify()  # Ensures image integrity
            except (OSError, ValueError):
                # Catch truncation or corruption errors
                print(f"Deleting corrupted or truncated image: {file_path}")
                os.remove(file_path)
                count += 1
                
    print("Removed: " + str(count) + " Files")

# Example usage:
remove_invalid_images('/home/woodm/CSC2611/pokemon-image-classifier/data')