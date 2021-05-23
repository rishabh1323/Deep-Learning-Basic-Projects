## Importing Required Libraries
import os
import random
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Download the Train and Test Data Using These Commands

# Downloading train data into temporal storage
#!wget --no-check-certificate https://archive.org/download/mnist_digit_recognizer_dataset/train.zip -O /data/train.zip

# Downloading test data into temporal storage
#!wget --no-check-certificate https://archive.org/download/mnist_digit_recognizer_dataset/test.zip -O /data/test.zip

## Extract From Train and Test Zip Files
# Extracting train data
local_zip = '/data/train.zip'
zip_file = zipfile.ZipFile(local_zip, 'r')
zip_file.extractall('/data')
zip_file.close()

# Extracting test data
local_zip = '/data/test.zip'
zip_file = zipfile.ZipFile(local_zip, 'r')
zip_file.extractall('/data')
zip_file.close()

## Getting Directory Paths for Train and Test Data
# Getting number of images in train data
count = 0
for class_name in sorted(os.listdir('/data/train')):
  count += len(os.listdir('/data/train/' + class_name))

# Printing number of images in train and test images
print('Number of images in train data:', count)
print('Number of images in test data:', len(os.listdir('/data/test/test')))

## Printing Random Images for Each Class
# Creating a plotting figure and axes
fig, axes = plt.subplots(1, 10)

# Iterating over every class and printing a single random image
for class_name in sorted(os.listdir('/data/train')):
  random_index = random.randint(0, len(os.listdir('/data/train/' + class_name)) - 1)
  random_image_name = os.listdir('/data/train/' + class_name)[random_index]
  image_path ='/data/train/' + class_name + '/' + random_image_name
  image = mpimg.imread(image_path)
  axes[int(class_name)].imshow(image)
  axes[int(class_name)].axis('off')
  axes[int(class_name)].set_title(class_name)

## Setting Up Image Data Generators
# Creating batch size
batch_size = 1000

# Setting up train data directory and its ImageDataGenerator
TRAINING_DIR = '/data/train/'
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest',
    validation_split = 0.2
)

# Setting up test data directory and its ImageDataGenerator
TEST_DIR = '/data/test/'
test_datagen = ImageDataGenerator(rescale = 1./255)

# Setting up data generator flow
training_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(28, 28), class_mode='categorical', 
                                                          batch_size=batch_size, subset='training', seed=42)

validation_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(28, 28), class_mode='categorical', 
                                                            batch_size=batch_size, subset='validation', seed=42)

test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=(28, 28), class_mode=None, 
                                                  shuffle=False, batch_size=1)

## Setting Up Deep Neural Network Structure
# Defining the neural network model structure
model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3), padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

# Printing model summary
model.summary()

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Setting some required parameters
epochs = 15
steps_per_epoch = training_generator.n // training_generator.batch_size
validation_steps = validation_generator.n // validation_generator.batch_size
test_steps = test_generator.n // test_generator.batch_size

## Training and Evaluating the Model
# Training the model
history = model.fit(training_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                    validation_data=validation_generator, validation_steps=validation_steps, verbose=1)

# Evaluating the model
print('\nValidation Loss and Accuracy')
model.evaluate(validation_generator, steps=validation_steps)

## Plotting Training and Validation Accuracy and Loss Functions
# Extracting the accuracy and loss functions
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Creating range of number of epochs
epochs = range(len(acc))

# Plotting training and validation accuracy
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting training and validation loss functions
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

## Predicting on Test Data Using the Model
y_pred = model.predict(test_generator, steps=test_steps, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
test_file_names = test_generator.filenames

## Plotting Random Test Images with Their Predicted Outputs
# Selecting 25 random indices between 0 and 27999
random_indices = [random.randint(0, 27999) for i in range(25)]
print(random_indices)

# Plotting test images on these indices with their predicted class
fig, axes = plt.subplots(1, 25, figsize=(24,3))
for i, index in enumerate(random_indices):
  image = mpimg.imread('/data/test/' + test_file_names[index])
  axes[i].imshow(image)
  axes[i].set_title(y_pred[index])
  axes[i].axis('off')

## Saving the trained model 
model.save('mnist_digit_recognizer.h5')