#\---> IMPORTS <-------------------------------------------------------------\#
# Modules
import numpy as np
import csv

# utils
from sklearn.model_selection import train_test_split
from utils import transformed_data_generator

# Keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2


#\---> CONSTANTS <-----------------------------------------------------------\#
# Data
DATA_DIR = 'data/'
CSV_FILE = 'driving_log.csv'

# batch size
BATCH_SIZE = 32


#\---> LOAD AND SPLIT DATA <-------------------------------------------------\#
# Load csv log file
csv_file = []
with open(DATA_DIR+CSV_FILE) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        csv_file.append(line)

# Split csv in train and validation sets
csv_train, csv_valid = train_test_split(csv_file, test_size = 0.2, shuffle = True)


#\---> CREATE AND TRAIN MODEL <----------------------------------------------\#
keep_prob = .7
def create_nv_model():
    # Define nvidia model
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))

    model.add(Conv2D(24, (5, 5), padding='same', strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(36, (5, 5), padding='same', strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(48, (5, 5), padding='same', strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(keep_prob))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(keep_prob))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(keep_prob))

    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam())
    return model


nv_model = create_nv_model()
nv_hist = nv_model.fit_generator(transformed_data_generator(csv_train),
                                steps_per_epoch = len(csv_train)/BATCH_SIZE*3,
                                epochs=8,
                                validation_data = transformed_data_generator(csv_valid),
                                validation_steps = len(csv_valid)/BATCH_SIZE*3)


# Save model
nv_model.save('models/nv_model.h5')
