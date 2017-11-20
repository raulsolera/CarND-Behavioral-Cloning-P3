#\---------------------------------------------------------------------------\#
#\---> IMPORTS & CONSTANTS <-------------------------------------------------\#
#\---------------------------------------------------------------------------\#

#\---> IMPORTS
# Modules
import numpy as np
import csv

# utils
from utils import transformed_data_generator, original_data_generator
from utils import save_model, load_data

# Keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import Adam


#\---> CONSTANTS
# Data
DATA_DIR = 'data/'
NEW_DATA_DIR = 'new-data/'
CSV_FILE = 'driving_log.csv'

# batch size
BATCH_SIZE = 32

# Camera parameters
cameras = ['center', 'left', 'right']


#\---------------------------------------------------------------------------\#
#\---> LOAD AND SPLIT DATA <-------------------------------------------------\#
#\---------------------------------------------------------------------------\#

# Load csv log file
csv_file = []
with open(DATA_DIR+CSV_FILE) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        csv_file.append(line)


# We are not setting apart a validation data set as the validation is done
# in the autonomous drive simulator.
# Not very elegant but a way to measure loss in a fix data set
validation_idx = np.random.choice(range(len(csv_file)), int(len(csv_file)/10),
                                  replace=False)
csv_valid = np.array(csv_file)[validation_idx]


#\---------------------------------------------------------------------------\#
#\---> MODEL DEFINITION <----------------------------------------------------\#
#\---------------------------------------------------------------------------\#

def create_c4f1_128_model():
    '''
    Simple model 4 convolutional layers with maxpooling to reduce size
    and single dense fully connected layer
    Relu activation function in all layers
    Adam optimizer and mean square error loss measure
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
    model.add(Conv2D(16, (5, 5), padding = 'same', strides = (1, 1),
                     activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding = 'same', strides = (1, 1),
                     activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), padding = 'same', strides = (1, 1),
                     activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding = 'same', strides = (1, 1),
                     activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))    
    model.compile(optimizer=Adam(), loss='mse')
    return model


#\---------------------------------------------------------------------------\#
#\---> MODEL TRAINING <------------------------------------------------------\#
#\---------------------------------------------------------------------------\#

#\--> STEP 0 - Create model
model = create_c4f1_128_model()


#\--> STEP 1 - Initial training: training using full dataset.
#              Use center, left and rigt cameras

# Generators
train_generator = transformed_data_generator(csv_file, read_data = True,
                                             lateral_cameras = cameras)
valid_generator = original_data_generator(csv_valid, read_data = True)

# Parameters
epochs = 3
initial_epoch = 0
steps_train = len(csv_file)/BATCH_SIZE
steps_valid = len(csv_valid)/BATCH_SIZE

# Model fit
model.fit_generator(train_generator, steps_per_epoch = steps_train,
                    epochs = epochs, initial_epoch = initial_epoch,
                    validation_data = valid_generator,
                    validation_steps = steps_valid)



#\--> STEP 2 - Fine tune diffcult road sections using ad-hoc data set.
#              Use center, left and rigt cameras

# Load new data (nd)
csv_nd_file = []
with open(NEW_DATA_DIR+CSV_FILE) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        csv_nd_file.append(line)

# Generators
train_generator = transformed_data_generator(csv_nd_file, read_data = True,
                                             data_dir = NEW_DATA_DIR,
                                             lateral_cameras = cameras)
valid_generator = original_data_generator(csv_valid, read_data = True)

# Parameters
epochs = 7
initial_epoch = 3
steps_train = len(csv_nd_file)/BATCH_SIZE
steps_valid = len(csv_valid)/BATCH_SIZE

# Model fit
model.fit_generator(train_generator, steps_per_epoch = steps_train,
                    epochs = epochs, initial_epoch = initial_epoch,
                    validation_data = valid_generator,
                    validation_steps = steps_valid)


#\--> STEP 3 - Save model

# Save model
model.save('models/model.h5')