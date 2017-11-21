# Behavioral Cloning Project

Udacity Self Driving Car Nanodegree project 3: behavioral cloning.

[//]: # (Image References)

[image1]: ./report-images/zero_bias.png "Zero bias angle distribution"
[image2]: ./report-images/cameras_image.png "Left, center and right image"
[image3]: ./report-images/random_flip.png "Random flip"
[image4]: ./report-images/random_brightness_correction.png "Random brightness correction"
[image5]: ./report-images/resize_pipeline.png "Resize pipeline"
[image6]: ./report-images/c4f1_128_model.png "Model architecture"
[image7]: ./report-images/image_generator.png "Image generator"
[image8]: ./report-images/angle_distribution.png "Angle distribution"
[image9]: ./report-images/driving-log.png "Driving log"
[image10]: ./report-images/model-summary.png "Model summary"
[image11]: ./report-images/video-thumbnail.png "It drives!"


## Overview

The goal of this project is to use deep learning to train a Convolutional Neural Network (CNN) capable of driving a car in a simulator. The car is equipped with three cameras (center, left right) that provide 12 to 13 shots per second wich are associated with the values of the steering angle, speed, throttle and brake. However in this project only the steering angle needs to be predicted and will not address the other values.

## Data

Data consists of three images from the center, left and right cameras in the car associated with the value of the steering angle. In the data collection process a CSV file is created that associates the path of the three images with its corresponding steering angle value, the following table shows the estructure of the log file:
![alt text][image9]

An inspection of the steering angle values shows that it is highly biased towards the zero value as "a car is most times drive in straight way", the following graph show this distribution:
![alt text][image1]

This zero biased issue will be adressed later.

On the other hand the data is skewed towards negative angle which is due to driving the car around the track in one direction only. This issue and can be easily eliminated by flipping each recorded image and its corresponding steering angle.


## Model Architecture and Training Strategy

### Final model

A model with 4 convolutional layers followed each one with a maxpooling layer (to reduce the model complexity and speed up the training) and a fully connected layer of size 128 was finally chosen. The input images were normalize to -1 to 1 interval in a lambda layer.

The following graph and tables provide details of the final model:
![alt text][image10]
![alt text][image6]


#### Overfitting prevention

The overfitting issue was address following the three principles:
- Use a simple model: that is one of the reasons to choose model with a single fully connected layer.
- Stop training early: 2 to 4 epochs were tested and a final number of 3 found optimal.
- Use some techniques of data augmentation: to prevent training over same image several times.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

A simple model was used as first approach with a single fully connected layer and relu activation funaction, this model worked OK on the easy sections of the road but fail to cope with difficult ones, no matter how short or long was the training.

Then a Lenet architecture model was tested but the results didn't get the car in the road for a full lap. Fine tuning with specific driving behaviors data was tried but failed to mantain the car in the road.

Finally, inspired by the NVidia successful model I decided to use 4 convolutional layers to capture the patterns of the road and then a single fully connected layer to predict the angle to keep the model as simple as possible. Fully connected layers of size 64, 128 and 256 were tested and the results were optimal for the 128 size fc layer.

#### 2. Training data

##### Data collection

Training data provided by Udacity was used. It consisted of 8.036 measures each one providing three images (shot from center, left and right cameras) and the measures for steering, throttle, brake and speed. Only the steering angle was used for the project.
![alt text][image2]

##### Data augmentation

As it was stated the data was heavily biased towards zero angle and this produced that the predicted angle was also biased towards zero. To cope with this issue both left and right cameras were used with a steering correction of 0.23 rad for left camera images and -0.23 rad for right camera images. At each measure one of the three cameras was randomly chosen and the steering angle corrected, the resulting angle distribution using this correction compared to the original dist. using the center camera is shown in the figure:
![alt text][image8]

The biased towards the negative angles due to the driving round the circuit in the same direcction was easily solved by randomly flipping the images and taking the negative steering angle.

This two techniques multiply by a factor of 6 the number of input images however to reduce even more the possibility of using the same images more than 1 time, a random bright correction was applied to the images.

Finally the original images had a size of 160 (height) x 320 (width) and 3 (RGB) channels, but the top and bottom stripes do not provide meaningful information so the 30 top and bottom pixels were cropped and then the resulting image was resize to 64x64x3 that was enough to capture the images features and grealy reduced the model complexity and hence speeded up the training.

The transformation pipeline is showed in the following images:
- Random flip:
![alt text][image3]
- Random brightness correction:
![alt text][image4]
- Resize pipeline
![alt text][image5]


##### Generator

These augmentation tranformations made impossible to load data into memory and hence a generator was used which allowed the training practically at no memory cost. The generator encapsulated the transformation and resizing pipeline as follows:
1. Shuffle the data.
2. Choose a random camera (out of center, left, right) with uniform probability.
3. Random flip the image.
4. Crop top and bottom stripes.
5. Resize the image.

The following images showed a 32 batch of images returned by the generator:
![alt text][image7]

#### Training and validation data set

Initially data was splitted among training and validation set, however measuring in mean square error in the validation set didn't show to be a fair approximation hence the training was done using the full data set and results validated in the autonomous drive simulator.

## Final result

After some attemps the vehicle was able to drive autonomously and smoothly in the intial section but failed to pass the section just after the bridge where the lane lines on the right of the road dissapeared.

To cope with this section I record the vehicle in this short section and fine tuned the model for 4 epochs in this specific section.

Finally, at the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.
[![alt text][image11]](https://youtu.be/CMWLqC6mgGk)