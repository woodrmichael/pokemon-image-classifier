#!/usr/bin/env python

import os

def delete_gif_files_recursive(directory):
    """
    Recursively deletes all .gif files in the specified directory and its subdirectories.
    
    Parameters:
    directory (str): The root directory to start searching for and deleting .gif files.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print("The directory doesn't exist")
        return
    
    # Walk through all directories and subdirectories
    count = 0
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            # Check if the file is a .gif file
            if filename.lower().endswith('.gif'):
                file_path = os.path.join(foldername, filename)
                try:
                    # Delete the .gif file
                    os.remove(file_path)
                    print("Deleted: " + file_path)
                    count += 1
                except Exception as e:
                    print("Error deleting " + file_path)
                    
    print("Deleted: " + str(count) + " Gifs")


delete_gif_files_recursive('/home/woodm/CSC2611/pokemon-image-classifier/data')