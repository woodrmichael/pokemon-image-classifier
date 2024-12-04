#!/usr/bin/env python
"""
Pokemon Image Predictor

# Building a model
# --data - subdirectory of images for training
# --batch_size - batch size to use for training
# --epochs - amount of epochs to use for training
# --main_dir - where to save produced models, defaults to working directory
# --augment_data - boolean indication for whether to use data augmentation
# --fine_tune - boolean indication for whether to use fine tuning

Note:
    - directory arguments must not be followed by a '/'
        Good: home/username
        Bad: home/username/

Example:

    python Lab11.py --data /data/cs2300/L9/fruits --batch_size 32 --epochs 10 --main_dir home/<username> --augment_data false --fine_tune true

"""

import os
import shutil
from sklearn.model_selection import train_test_split
import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import itertools
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn import preprocessing
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--batch_size', type=str, default='')
    parser.add_argument('--epochs', type=str, default='')
    parser.add_argument('--main_dir', type=str, default='')
    parser.add_argument('--augment_data', type=str, default='')
    parser.add_argument('--fine_tune', type=str, default='')
    return parser.parse_args()


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


def main():
    # start by parsing the command line arguments
    args = parse_args()
    data = args.data
    my_batch_size = int(args.batch_size)
    my_epochs = int(args.epochs)
    augment_data = args.augment_data
    fine_tune = args.fine_tune
    h5modeloutput = 'model_b' + args.batch_size + '_e' + args.epochs + '_aug' + \
                    args.augment_data + '_ft' + args.fine_tune + '.h5'
    print(args)

    # Create stratified train/test split
    temp_dir = os.path.join(args.main_dir, 'temp_split')
    train_dir, val_dir = create_stratified_split(data, temp_dir)

    # Load weights pre-trained on the ImageNet model
    base_model = keras.applications.VGG16(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False)

    # Next, we will freeze the base model so that all the learning from the ImageNet
    # dataset does not get destroyed in the initial training.
    base_model.trainable = False

    # Create inputs with correct shape
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)

    # Add pooling layer or flatten layer
    x = keras.layers.GlobalAveragePooling2D()(x)

    # TODO - CHANGE TO HOW MANY OUTPUT CLASSES WE HAVE
    # Add final dense layer with 150 classes for the 150 types of pokemon
    outputs = keras.layers.Dense(150, activation='softmax')(x)

    # Combine inputs and outputs to create model
    model = keras.Model(inputs, outputs)

    # uncomment the following code if you want to see the model
    # model.summary()

    # Now it's time to compile the model with loss and metrics options.
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0,  # randomly zoom image
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # These are data augmentation steps
    if (augment_data.lower() in ['true', '1', 't', 'y', 'yes']):
        datagen = ImageDataGenerator(
            samplewise_center=True,  # set each sample mean to 0
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.01,  # randomly zoom image
            width_shift_range=0.01,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.01,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    # load and iterate training dataset
    train_it = datagen.flow_from_directory(train_dir,
                                           target_size=(224, 224),
                                           color_mode='rgb',
                                           batch_size=my_batch_size,
                                           class_mode="categorical")
    # load and iterate validation dataset
    valid_it = datagen.flow_from_directory(val_dir,
                                           target_size=(224, 224),
                                           color_mode='rgb',
                                           batch_size=my_batch_size,
                                           class_mode="categorical")

    # Train the model
    history_object = model.fit(train_it,
                               validation_data=valid_it,
                               steps_per_epoch=train_it.samples / train_it.batch_size,
                               validation_steps=valid_it.samples / valid_it.batch_size,
                               epochs=my_epochs,
                               verbose=2)

    if (fine_tune.lower() in ['true', '1', 't', 'y', 'yes']):
        # This will improve the accuracy of the model by fine tuning the training on the entire unfrozen model.
        # Unfreeze the base model
        base_model.trainable = True
        # Compile the model with a low learning rate
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.00001),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        history_object = model.fit(train_it,
                                   validation_data=valid_it,
                                   steps_per_epoch=train_it.samples / train_it.batch_size,
                                   validation_steps=valid_it.samples / valid_it.batch_size,
                                   epochs=my_epochs)

    save_loss_plot(history_object.history, args)
    model.save(args.main_dir + '/' + h5modeloutput)


def save_loss_plot(history, args):
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Pokemon Classification, Batch Size: ' + args.batch_size + ' Epochs: ' + args.epochs)
    plt.legend()
    plt.savefig(args.main_dir + '/' + 'model_b' + args.batch_size + '_e' + args.epochs + '.png')


if __name__ == "__main__":
    main()
