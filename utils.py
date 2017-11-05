# Imports
import cv2

# Cropping parameters
TOP_STRIDE_TO_CROP = 30
BOTTOM_STRIDE_TO_CROP = 25
NEW_WIDTH = 64
NEW_HEIGHT = 64

# crop resize funtion to in both training and driving
def crop_resize(image):
    height = image.shape[0]
    cropped_image = image[TOP_STRIDE_TO_CROP:height-BOTTOM_STRIDE_TO_CROP, :, :]
    return cv2.resize(cropped_image,(NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_AREA)