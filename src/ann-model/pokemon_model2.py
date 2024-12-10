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

# +
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# +
# Hyperparameters from the best previous run
best_hidden_neurons = 16  # Example value from previous best run
best_epochs = 50  # Example value from previous best run
dropout_rate = 0.3  # Typical dropout rate (30%)

# Create a new MLP model with an additional dense layer and dropout
def build_mlp_model(input_shape, hidden_neurons, dropout_rate):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))  # Dropout layer
    model.add(Dense(hidden_neurons, activation='relu'))  # Additional dense layer
    model.add(Dense(4, activation='softmax'))  # Output layer
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])
    return model

# Build and train the model
mlp_model = build_mlp_model(input_shape=(X_high.shape[1],), hidden_neurons=best_hidden_neurons, dropout_rate=dropout_rate)

# Fit the model on the high-variance data
mlp_model.fit(X_high, y_high, epochs=best_epochs, verbose=2)

# Evaluate the model
predicted_classes_train = np.argmax(mlp_model.predict(X_high), axis=1)
train_accuracy = 100. * accuracy_score(y_high, predicted_classes_train)

# Evaluate on the same dataset as test (again, in practice, use a separate test set)
predicted_classes_test = np.argmax(mlp_model.predict(X_high), axis=1)
test_accuracy = 100. * accuracy_score(y_high, predicted_classes_test)

# Get the total number of model parameters
total_params = mlp_model.count_params()

# Print the results
print(f"\nTraining Accuracy with Dropout: {train_accuracy:.2f}%")
print(f"Testing Accuracy with Dropout: {test_accuracy:.2f}%")
print(f"Total # of Model Parameters: {total_params}")
# -

