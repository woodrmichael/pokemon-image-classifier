#!/usr/bin/env python

import os

def check_images(directory):
    corrupted_files = []

    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            count += 1
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "rb") as f:
                    f.read()  # Read the entire file to ensure it's not truncated
                print("Checked: " + file_path + " OK")
            except Exception as e:
                print("Error with file: " + file_path)
                corrupted_files.append(file_path)

    print("Total Files: " + str(count))
    return corrupted_files


# Directory containing your image dataset
data_directory = "/home/woodm/CSC2611/pokemon-image-classifier/data"
corrupted = check_images(data_directory)

print("\nCorrupted Files: " + str(len(corrupted)))
for f in corrupted:
    print(f)