## Importing Required Libraries
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Download the Train and Test Data Using These Commands
# Downloading the train data into temporal storage
#!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip -O /data/rps.zip

# Downloading the test data into temporal storage
#!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip -O /data/rps-test-set.zip

## Extracting Train and Test Data Zip Files
# Extracting the train data
local_zip = '/data/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/data/')
zip_ref.close()

# Extracting the test data
local_zip = '/data/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/data')
zip_ref.close()

## Getting Directory Paths for Each Class
# Setting up class directories for training data
rock_dir = '/data/rps/rock'
paper_dir = '/data/rps/paper'
scissors_dir = '/data/rps/scissors'

# Printing number of images in each class
print('Total training rock images: ', len(os.listdir(rock_dir)))
print('Total training paper images: ', len(os.listdir(paper_dir)))
print('Total training scissors images: ', len(os.listdir(scissors_dir)))

# getting list of image names for each class
rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)

## Printing Image for Each Class for Given Index
# Getting a valid index between 0 and 839
image_index = 5

# Creating the image directories for the index for each class
next_rock = os.path.join(rock_dir, rock_files[image_index])
next_paper = os.path.join(paper_dir, paper_files[image_index])
next_scissors = os.path.join(scissors_dir, scissors_files[image_index])

for img_path in [next_rock, next_paper, next_scissors]:
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('off')
  plt.show()

## Setting Up Image Data Generators for Training and Validation Sets
# Setting up train directory and its ImageDataGenerator
TRAINING_DIR = '/data/rps'
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2, 
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode= 'nearest'
)

# Setting up validation directory and its ImageDataGenerator
VALIDATION_DIR = '/data/rps-test-set'
validation_datagen = ImageDataGenerator(rescale = 1./255)

# Setting up training generator using flow_from_directory
training_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (150, 150),
    class_mode = 'categorical',
    batch_size = 126
)

# Setting up validation generator using flow_from_directory
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size = (150, 150),
    class_mode = 'categorical',
    batch_size = 126
)

## Setting Up the Deep Neural Network Structure
# Defining the neural network model structure
model = tf.keras.models.Sequential([
    # First convolutional layer for 150*150 image size with 3 RGB channels                                 
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # second convolutional layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Third convolutional layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Fourth convolutional layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flattening the output to feed into dense layers
    tf.keras.layers.Flatten(),
    # Dropout layer to provide regularization
    tf.keras.layers.Dropout(0.5),
    # First dense layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Output dense layer
    tf.keras.layers.Dense(3, activation='softmax')
])

# Printing model summary
model.summary()

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Training the model on train set and validating it on validation set
history = model.fit(training_generator, epochs=25, steps_per_epoch = 2520//126, validation_data=validation_generator, validation_steps=3, verbose=1)

## Plotting Training and Validation Accuracy and Loss Functions
# Getting training and validation accuracies and losses
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Getting number of epochs
epochs = range(len(acc))

# Plotting the accuracy graphs
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc=0)
plt.show()

# Plotting the loss functions
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc=0)
plt.show()

## Saving the Trained Model
# Saving the trained model 
model.save('rock_paper_scissors.h5')



