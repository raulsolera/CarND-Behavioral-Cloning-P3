#\---> IMPORTS <-------------------------------------------------------------\#
import numpy as np
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
    height, width = images.shape[1:3]
    cropped_images = images[:, top_cut:height - bottom_cut, left_cut:width - right_cut, :]
    resize_images = [resize(image) for image in cropped_images]
    return np.array(resize_images)


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


# Random shear
def random_shear(image, angle, shear_range=MAX_SHEAR_SHIFT):
    """
    Shear (affine transformation) horizontally by a random shift in the range of shear_range
    Implies also steering angle correction
    https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
    """
    rows, cols, channels = image.shape
    shear_shift = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + shear_shift, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    shear_angle = shear_shift / (rows / 2) * 180 / (np.pi * 25.0) / 6

    return image, angle + shear_angle


# Random rotation
def random_rotate(image, angle, rotation_range=MAX_ROTATION_ANGLE):
    """
    Rotate image by a random angle in the range of rotation_range
    Implies also steering angle correction
    """
    rotation_angle = np.random.uniform(-rotation_range, rotation_range+1)
    rotate_fun = iaa.Affine(rotate=(-rotation_angle))
    image = rotate_fun.augment_images([image])[0]
    rad = (np.pi / 180.0) * rotation_angle

    return image, angle - rad


# Convert to grayscale
def conv_to_graysale(image):
    """
    Convert to graysacale image an 1 channel, shape [X:X:1]
    """
    image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    image = np.expand_dims(image, axis=2)
    return image


#\---> GENERATOR <-----------------------------------------------------------\#
def transformed_data_generator(data, data_dir=DATA_DIR,
                               batch_size=BATCH_SIZE,
                               use_lateral_cameras = True,
                               new_width=NEW_WIDTH, new_height=NEW_HEIGHT,
                               image_load=True, shear_prob = 0.5):
    # Camera parameters
    cameras = ['center', 'left', 'right']
    cameras_index = {'center': 0, 'left': 1, 'right': 2}  # 0:center, 1:left, 2:right
    cameras_steering_correction = {'center': 0., 'left': STEERING_CORRECTION, 'right': -STEERING_CORRECTION}

    num_samples = len(data)

    while 1:  # Loop forever so the generator never terminates

        data = shuffle(data)
        images = []
        angles = []

        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset + batch_size]

            for line in batch_samples:

                ### Randomly choose center, left or right image
                # Get random camera 0:center, 1:left, 2:right
                if use_lateral_cameras:
                    camera = np.random.choice(cameras)
                else:
                    camera = 'center'
                file_name = line[cameras_index[camera]].split('/')[-1]
                path = data_dir + 'IMG/' + file_name
                if image_load:
                    image = cv2.imread(path)
                else:
                    image = np.zeros((160, 320, 3), dtype=np.uint8)
                # Adjust angle
                angle = float(line[3]) + cameras_steering_correction[camera]

                ### Random transformations
                image, angle = random_flip(image, angle)
                image, angle = random_brightness_correction(image, angle)
                if shear_prob > np.random.random():
                    image, angle = random_shear(image, angle)
                # image, angle = random_rotate(image, angle)

                ### Resize
                image = crop(image)
                image = resize(image, new_width=new_width, new_height=new_height)

                images.append(image)
                angles.append(angle)

            yield np.array(images), np.array(angles)


def original_data_generator(data, data_dir=DATA_DIR,
                            batch_size=BATCH_SIZE,
                            tr_fn = [crop, resize],
                            image_load=True):

    num_samples = len(data)

    while 1:  # Loop forever so the generator never terminates

        data = shuffle(data)
        images = []
        angles = []

        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset + batch_size]

            for line in batch_samples:

                # Load image and angle
                file_name = line[0].split('/')[-1]
                path = data_dir + 'IMG/' + file_name
                if image_load:
                    image = cv2.imread(path)
                else:
                    image = np.zeros((160, 320, 3), dtype=np.uint8)
                angle = float(line[3])

                ### Resizing
                for fn in tr_fn:
                    image = fn(image)

                images.append(image)
                angles.append(angle)

            yield np.array(images), np.array(angles)


def lenet_generator(data, data_dir=DATA_DIR,
                    batch_size=BATCH_SIZE,
                    use_lateral_cameras = True,
                    image_load=True, shear_prob = 0.5):
    # For lenet architecture 32x32x1 grayscaled images required
    for images, angles in transformed_data_generator(data, data_dir=DATA_DIR,
                                                     batch_size=BATCH_SIZE,
                                                     new_width=32, new_height=32,
                                                     use_lateral_cameras = True,
                                                     image_load=True, shear_prob = 0.5):
        images = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        images = np.expand_dims(images, axis=3)
        yield images, angles


def lenet_orig_generator(csv_file):
    # For lenet architecture 32x32x1 grayscaled images required
    for images, angles in original_data_generator(csv_file, new_width=32, new_height=32):
        images = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        images = np.expand_dims(images, axis=3)
        yield images, angles


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


#\---> OTHER_UTILS <---------------------------------------------------------\#

    

