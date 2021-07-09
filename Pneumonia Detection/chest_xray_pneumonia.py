# Chest X-Ray Pneumonia Classification

#### Importing Required Files

import os
import random
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from google.colab import files
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#### Extracting the Train and Test Data Zip Files

# Extracting the train data
local_zip = '/tmp/train.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

# Extracting the validation data
local_zip = '/tmp/val.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# Extracting the test data
local_zip = '/tmp/test.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()


#### Getting Directory Paths for Each Class

# Setting up class directories for training data
normal_dir = '/tmp/train/NORMAL'
pneumonia_dir = '/tmp/train/PNEUMONIA'

# Printing number of images in each class
print('Total training normal images: ', len(os.listdir(normal_dir)))
print('Total training pneumonia images: ', len(os.listdir(pneumonia_dir)))

# Getting list of image names for each class
normal_files = os.listdir(normal_dir)
pneumonia_files = os.listdir(pneumonia_dir)
    

#### Printing Image for Each Class for Given Index

# Getting a valid random index between 0 and 1341
image_index = random.randint(0, 1340)

# Creating the image directories for the index for each class
next_normal = os.path.join(normal_dir, normal_files[image_index])
next_pneumonia = os.path.join(pneumonia_dir, pneumonia_files[image_index])

# Plotting the images for both classes
for img_path in [next_normal, next_pneumonia]:
  img = mpimg.imread(img_path)
  plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.show()


#### Setting up Image Data Generators for Training and Validation Data

# Fixing batch size
BATCH_SIZE = 32

# Setting up train directory and its ImageDataGenerator
TRAINING_DIR = '/tmp/train'
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40, 
    zoom_range = 0.2,
    vertical_flip = True,
    fill_mode= 'nearest',
    validation_split = 0.2
)

# Setting up test directory and its ImageDataGenerator
TEST_DIR = '/tmp/test'
test_datagen = ImageDataGenerator(rescale = 1./255)

# Setting up training generator using flow_from_directory
training_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (299, 299),
    class_mode = 'binary',
    batch_size = BATCH_SIZE,
    subset = 'training'
)

# Setting up validation generator using flow_from_directory
validation_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (299, 299),
    class_mode = 'binary',
    batch_size = BATCH_SIZE,
    subset = 'validation'
)

# Setting up validation test using flow_from_directory
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = (299, 299),
    class_mode = None,
    batch_size = 1
)    


#### Setting up the Deep Neural Network Structure

# Defining the neural network model structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(299, 299, 3), padding='same'),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Printing model summary
model.summary()

# Fixing some required parameters
num_epochs = 10
steps_per_epoch = training_generator.n // training_generator.batch_size
validation_steps = validation_generator.n // validation_generator.batch_size
test_steps = test_generator.n // test_generator.batch_size

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Setting up callback function for reducing learning rate when validation loss stops reducing
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 2, verbose = 2, mode = 'max')

# Training the model on train set and validating it on validation set
history = model.fit(training_generator, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, 
                    validation_steps=validation_steps, callbacks=[lr_reduce], verbose=1)


#### Plotting Training and Validation Accuracies and Loss Functions


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


#### Saving the Trained Model

# Saving the trained model 
model.save('saved_model/my_model')