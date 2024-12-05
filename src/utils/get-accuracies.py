#!/usr/bin/env python

# --data - subdirectory of images for accuracies
# --model - path to model 

import os
import argparse
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow import keras

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--model', type=str, default='')
    return parser.parse_args()

def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, target_size=(224,224))
    return image

def create_dictionary(data_path):
    if os.path.exists(data_path) and os.path.isdir(data_path):
        # Get all folder names in the specified path
        folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
        
        # Create a dictionary with incremental keys starting from 1
        dictionary = {index: folder for index, folder in enumerate(sorted(folders))}
        
        # Print the resulting dictionary
        print(dictionary)
        
        return dictionary
        
    else:
        print(f"Directory {data_path} does not exist or is not accessible.")
        
        return None
        
def get_accuracies(data_dir, model, dictionary):
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    # Initialize lists to store accuracy and class names
    accuracies = []
    class_names = []
    
    # Iterate through all subdirectories in data_dir
    for subdir in os.listdir(data_dir):
        current_dir = os.path.join(data_dir, subdir)
    
        # Ensure the path is a directory
        if not os.path.isdir(current_dir):
            continue
    
        # Use the directory name as the label
        label = os.path.basename(current_dir)
        class_names.append(label)
    
        total = 0
        correct = 0
    
        # Get all valid image files in the directory
        all_files = [f for f in os.listdir(current_dir) if f.lower().endswith(valid_extensions)]
    
        for file in all_files:
            total += 1
            file_path = os.path.join(current_dir, file)
    
            # Preprocess the image
            image = load_and_scale_image(file_path)
            image = image_utils.img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
    
            # Make prediction
            prediction = model.predict(image)
            guess = dictionary[np.argmax(prediction)]
            print(f"File: {file}, Guess: {guess} Correct: {subdir}")
    
            if guess == label:
                correct += 1
    
        # Calculate accuracy for the current class
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)
        print(f"Accuracy for {label}: {accuracy:.2f}")
    
    # Print out all accuracies at the end
    print("\nOverall Results:")
    for class_name, accuracy in zip(class_names, accuracies):
        print(f"Class: {class_name}, Accuracy: {accuracy:.2f}")
        
def main():
    # start by parsing the command line arguments
    args = parse_args()
    data = args.data
    model = args.model
    model = keras.models.load_model(model)
    
    dictionary = create_dictionary(data)
    get_accuracies(data, model, dictionary)
    
    
if __name__ == "__main__":
    main()