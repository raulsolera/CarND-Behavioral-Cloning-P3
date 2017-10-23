import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

# Load csv log file
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)

print(lines[0])
exit()

# Load images and steering data
images = []
measurements = []
for line in lines:
    source_path = 'data/' + line[0]
    image = cv2.imread(source_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append[measurement]
X_train = np.array(images)
y_train = np.array(measurements)

# Define simple model
model = Sequential()
model.add(Flatten(input_shape= (160, 320, 3)))
model.add(Dense(1))

# Train simple model
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7)

# Save model
model.save('model.h5')