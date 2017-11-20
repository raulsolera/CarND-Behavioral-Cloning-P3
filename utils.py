#\---> IMPORTS <-------------------------------------------------------------\#
import numpy as np
import csv
import cv2
import os, errno
from scipy.ndimage import rotate
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from random import random


#\---> CONSTANTS <-----------------------------------------------------------\#
# Data
DATA_DIR = 'data/'
CSV_FILE = 'driving_log.csv'

# Resize constants
TOP_CUT = 30
BOTTOM_CUT = 30
NEW_WIDTH = 64
NEW_HEIGHT = 64
MAX_ROTATION_ANGLE = 15
MAX_SHEAR_SHIFT = 200
STEERING_CORRECTION = 0.23

# batch size
BATCH_SIZE = 32

# Camera parameters
cameras = ['center', 'left', 'right']
cameras_index = {'center': 0, 'left': 1, 'right': 2}  # 0:center, 1:left, 2:right
cameras_steering_correction = {'center': 0., 'left': STEERING_CORRECTION, 'right': -STEERING_CORRECTION}


#\---> IMAGE LOAD <----------------------------------------------------------\#
# Load images to memory to speed up generator
def get_csv_file():
    csv_file = []
    with open(DATA_DIR+CSV_FILE) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            csv_file.append(line)
    return csv_file


def load_data(csv_file, data_dir = DATA_DIR):
    # Load center camera data
    data = []
    for line in csv_file:
        data_line = []
        for camera in cameras:
            path = data_dir + 'IMG/' + line[cameras_index[camera]].strip().split('/')[-1]
            data_line.append(cv2.imread(path))
        data_line.append(float(line[3]))
        data.append(data_line)
    return data 


#\---> IMAGE TRANSFORMATIONS <-----------------------------------------------\#
### Image transformations
# Crop image
def crop(image, top_cut=TOP_CUT, bottom_cut=BOTTOM_CUT, left_cut=0, right_cut=0):
    """
    Crop top, bottom, left and right sides of image
    """
    height, width = image.shape[0:2]
    cropped_image = image[top_cut:height - bottom_cut, left_cut:width - right_cut, :]
    return cropped_image


# Resize image
def resize(image, new_width=NEW_WIDTH, new_height=NEW_HEIGHT):
    """
    Resize image to new size
    """
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# Crop and resize
def crop_resize(images, top_cut=TOP_CUT, bottom_cut=BOTTOM_CUT, left_cut=0, right_cut=0,
                new_width=NEW_WIDTH, new_height=NEW_HEIGHT):
    """
    Crop and resize list of images
    """
    if not images is np.ndarray:
        images = np.array(images)
    height, width = images[0].shape[0:2]
    cropped_images = images[:, top_cut:height - bottom_cut, left_cut:width - right_cut, :]
    resized_images = [resize(image) for image in cropped_images]
    return np.array(resized_images)


# Random flip
def random_flip(image, angle):
    """
    Random flip image horizontally with probability of 0.5
    """
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        angle *= -1.
    return image, angle


# Random brightnes correction
def random_brightness_correction(image, angle):
    """
    Correct brightness of image in a random factor from (0.3, 1.7) in all channels
    """
    brightness_factor = np.random.uniform(0.4, 1.6)
    image = image * brightness_factor
    image[image > 255] = 255
    image = np.array(image, dtype=np.uint8)
    return image, angle


#\---> GENERATORS <----------------------------------------------------------\#    
def transformed_data_generator(data,
                               read_data = False,
                               data_dir = DATA_DIR,
                               batch_size=BATCH_SIZE,
                               lateral_cameras = cameras,
                               flip = True,
                               brightness = True,
                               new_width=NEW_WIDTH, new_height=NEW_HEIGHT):
    
    # data = load_data(csv_file)
    num_samples = len(data)

    while 1:  # Loop forever so the generator never terminates

        data = shuffle(data)

        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset + batch_size]
            
            images = []
            angles = []
            
            for line in batch_samples:

                ### Randomly choose center, left or right image
                # Get random camera 0:center, 1:left, 2:right
                camera = np.random.choice(lateral_cameras)
                
                if read_data:
                    path = data_dir + 'IMG/' + line[cameras_index[camera]].strip().split('/')[-1]                    
                    image = cv2.imread(path)
                else:
                    image = line[cameras_index[camera]]
                angle = float(line[3]) + cameras_steering_correction[camera]

                ### Random transformations
                if flip:
                    image, angle = random_flip(image, angle)
                if brightness:
                    image, angle = random_brightness_correction(image, angle)

                images.append(image)
                angles.append(angle)

            ### Resize
            images = crop_resize(images, new_width=new_width, new_height=new_height)
                
            yield np.array(images), np.array(angles)            

def original_data_generator(data,
                            read_data = False,
                            data_dir = DATA_DIR,
                            batch_size=BATCH_SIZE,
                            new_width=NEW_WIDTH, new_height=NEW_HEIGHT):

    num_samples = len(data)

    while 1:  # Loop forever so the generator never terminates

        data = shuffle(data)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset + batch_size]
            
            images = []
            angles = []

            for line in batch_samples:

                # Load image and angle
                if read_data:
                    path = data_dir + 'IMG/' + line[cameras_index['center']].strip().split('/')[-1]                    
                    image = cv2.imread(path)
                else:
                    image = line[cameras_index['center']]
                angle = float(line[3])

                images.append(image)
                angles.append(angle)
                
            ### Resize
            images = crop_resize(images, new_width=new_width, new_height=new_height)
                
            yield np.array(images), np.array(angles)


#\---> MODEL SAVING <--------------------------------------------------------\#
# Delete previous files:
def delete_file(file_name):
    try:
        os.remove(file_name)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

def save_model(model, model_name):
    delete_file(model_name)
    model.save(model_name)