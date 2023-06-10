from Dataset_eng import create_word_level_dataset
from LRCN import create_LRCN_model
import numpy as np
# Import the required libraries.
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from moviepy.editor import *
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

if __name__ == "__main__":
    seed_constant = 27
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)

    CLASSES_LIST = ['sil', 'at', 'five', 'bin', 'red', 'two', 'a', 'j', 'green', 'p', 'eight', 'now', 'place', 'again',
                    'f', 'b', 'nine', 'n', 'o', 'lay', 'with', 'g', 'q', 's', 'x', 'in', 'd', 'four', 'soon', 'one',
                    'k', 'v', 'please', 'c', 'e', 'y', 'z', 'i', 'blue', 'by', 'zero', 'l', 'u', 'seven', 't', 'set',
                    'h', 'three', 'r', 'm', 'white', 'six']


    # Important variables DEFINED IN PREPROCESSOR FILES
    SEQUENCE_LENGTH = 13
    IMAGE_HEIGHT = 50
    IMAGE_WIDTH = 100

    f1 = np.load('../Preprocessing/s1_features.npy')
    l1 = np.load('../Preprocessing/s1_labels.npy')

    features = f1[0:2006]
    labels = l1[0:2006]

    del f1
    del l1

    print(len(features))
    print(len(labels))
    print(features.shape)
    print(labels.shape)

    # Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
    one_hot_encoded_labels = to_categorical(labels)

    # Split the Data into Train ( 75% ) and Test Set ( 25% ).
    features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                                test_size=0.25, shuffle=True,
                                                                                random_state=seed_constant)

    # Construct the required LRCN model.
    LRCN_model = create_LRCN_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST)

    # Display the success message.
    print("Model Created Successfully!")

    # Create an Instance of Early Stopping Callback.
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

    # Compile the model and specify loss function, optimizer and metrics to the model.
    LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

    # Start training the model.
    LRCN_model_training_history = LRCN_model.fit(x=features_train, y=labels_train, epochs=300, batch_size=4,
                                                 shuffle=True, validation_split=0.2,
                                                 callbacks=[early_stopping_callback])

    # Evaluate the trained model.
    model_evaluation_history = LRCN_model.evaluate(features_test, labels_test)
    # Get the loss and accuracy from model_evaluation_history.
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

    # Define the string date format.
    # Get the current Date and Time in a DateTime Object.
    # Convert the DateTime object to string according to the style mentioned in date_time_format string.
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

    # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
    model_file_name = f'./Weights/LRCN_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

    # Save the Model.
    LRCN_model.save(model_file_name)