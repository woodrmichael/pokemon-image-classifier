#!/usr/bin/env python

def create_stratified_split(data_dir, temp_dir, test_size=0.2):
    """
    Split the dataset into stratified training and validation datasets.

    Args:
        data_dir (str): Path to the folder containing the dataset.
        temp_dir (str): Path to the temporary folder for train/test split.
        test_size (float): Fraction of data to use as validation set.
    """
    # Clear the temp_dir if it exists, and recreate it
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    train_dir = os.path.join(temp_dir, 'train')
    val_dir = os.path.join(temp_dir, 'test')
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    # Iterate through each class folder
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Get all file paths for the class
        image_paths = [os.path.join(class_path, fname) for fname in os.listdir(class_path)]

        # Stratified split
        train_paths, val_paths = train_test_split(image_paths, test_size=test_size,
                                                  stratify=[class_name] * len(image_paths))

        # Create class directories in train and validation folders
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy files into the respective directories
        for path in train_paths:
            shutil.copy(path, train_class_dir)
        for path in val_paths:
            shutil.copy(path, val_class_dir)

    return train_dir, val_dir
    
data_dir = "/home/woodm/CSC2611/pokemon-image-classifier/data"
stratified_dir = "/home/woodm/CSC2611/pokemon-image-classifier/stratified-data"
test_size = 0.2

create_stratified_split(data_dir, stratified_dir, test_size)