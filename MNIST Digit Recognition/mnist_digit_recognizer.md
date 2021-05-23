# MNIST Digit Recognizer

#### Importing Required Libraries


```python
import os
import random
import pickle
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from google.colab import files
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

#### Downloading Train and Test Data


```python
# Downloading train data into temporal storage
!wget --no-check-certificate https://archive.org/download/mnist_digit_recognizer_dataset/train.zip -O /tmp/train.zip

# Downloading test data into temporal storage
!wget --no-check-certificate https://archive.org/download/mnist_digit_recognizer_dataset/test.zip -O /tmp/test.zip
```

    --2021-05-11 12:06:27--  https://archive.org/download/mnist_digit_recognizer_dataset/train.zip
    Resolving archive.org (archive.org)... 207.241.224.2
    Connecting to archive.org (archive.org)|207.241.224.2|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://ia801408.us.archive.org/17/items/mnist_digit_recognizer_dataset/train.zip [following]
    --2021-05-11 12:06:27--  https://ia801408.us.archive.org/17/items/mnist_digit_recognizer_dataset/train.zip
    Resolving ia801408.us.archive.org (ia801408.us.archive.org)... 207.241.228.148
    Connecting to ia801408.us.archive.org (ia801408.us.archive.org)|207.241.228.148|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 30205580 (29M) [application/zip]
    Saving to: ‘/tmp/train.zip’
    
    /tmp/train.zip      100%[===================>]  28.81M   348KB/s    in 55s     
    
    2021-05-11 12:07:23 (537 KB/s) - ‘/tmp/train.zip’ saved [30205580/30205580]
    
    --2021-05-11 12:07:23--  https://archive.org/download/mnist_digit_recognizer_dataset/test.zip
    Resolving archive.org (archive.org)... 207.241.224.2
    Connecting to archive.org (archive.org)|207.241.224.2|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://ia801408.us.archive.org/17/items/mnist_digit_recognizer_dataset/test.zip [following]
    --2021-05-11 12:07:24--  https://ia801408.us.archive.org/17/items/mnist_digit_recognizer_dataset/test.zip
    Resolving ia801408.us.archive.org (ia801408.us.archive.org)... 207.241.228.148
    Connecting to ia801408.us.archive.org (ia801408.us.archive.org)|207.241.228.148|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 20236148 (19M) [application/zip]
    Saving to: ‘/tmp/test.zip’
    
    /tmp/test.zip       100%[===================>]  19.30M   794KB/s    in 26s     
    
    2021-05-11 12:07:50 (758 KB/s) - ‘/tmp/test.zip’ saved [20236148/20236148]
    
    

#### Extracting From Train and Test Zip Files


```python
# Extracting train data
local_zip = '/tmp/train.zip'
zip_file = zipfile.ZipFile(local_zip, 'r')
zip_file.extractall('/tmp')
zip_file.close()

# Extracting test data
local_zip = '/tmp/test.zip'
zip_file = zipfile.ZipFile(local_zip, 'r')
zip_file.extractall('/tmp')
zip_file.close()
```

#### Getting Directory Paths for Train and Test Data


```python
# Getting number of images in train data
count = 0
for class_name in sorted(os.listdir('/tmp/train')):
  count += len(os.listdir('/tmp/train/' + class_name))

# Printing number of images in train and test images
print('Number of images in train data:', count)
print('Number of images in test data:', len(os.listdir('/tmp/test/test')))
```

    Number of images in train data: 42000
    Number of images in test data: 28000
    

#### Printing Random Images for Each Class


```python
# Creating a plotting figure and axes
fig, axes = plt.subplots(1, 10)

# Iterating over every class and printing a single random image
for class_name in sorted(os.listdir('/tmp/train')):
  random_index = random.randint(0, len(os.listdir('/tmp/train/' + class_name)) - 1)
  random_image_name = os.listdir('/tmp/train/' + class_name)[random_index]
  image_path ='/tmp/train/' + class_name + '/' + random_image_name
  image = mpimg.imread(image_path)
  axes[int(class_name)].imshow(image)
  axes[int(class_name)].axis('off')
  axes[int(class_name)].set_title(class_name)
```


    
![png](/outputs/output_10_0.png)
    


#### Setting Up Image Data Generators


```python
# Creating batch size
batch_size = 1000
```


```python
# Setting up train data directory and its ImageDataGenerator
TRAINING_DIR = '/tmp/train/'
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
TEST_DIR = '/tmp/test/'
test_datagen = ImageDataGenerator(rescale = 1./255)

# Setting up data generator flow
training_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(28, 28), class_mode='categorical', 
                                                          batch_size=batch_size, subset='training', seed=42)

validation_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(28, 28), class_mode='categorical', 
                                                            batch_size=batch_size, subset='validation', seed=42)

test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=(28, 28), class_mode=None, 
                                                  shuffle=False, batch_size=1) 
```

    Found 33604 images belonging to 10 classes.
    Found 8396 images belonging to 10 classes.
    Found 28000 images belonging to 1 classes.
    

#### Setting Up Deep Neural Network Structure


```python
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
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_3 (Conv2D)            (None, 28, 28, 32)        896       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 14, 14, 64)        18496     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 7, 7, 64)          0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 7, 7, 128)         73856     
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 3, 3, 128)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1152)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               590336    
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 688,714
    Trainable params: 688,714
    Non-trainable params: 0
    _________________________________________________________________
    

#### Training and Evaluating the Model


```python
# Training the model
history = model.fit(training_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                    validation_data=validation_generator, validation_steps=validation_steps, verbose=1)

# Evaluating the model
print('\nValidation Loss and Accuracy')
model.evaluate(validation_generator, steps=validation_steps)
```

    Epoch 1/15
    33/33 [==============================] - 40s 1s/step - loss: 1.9492 - acc: 0.3057 - val_loss: 0.9885 - val_acc: 0.6766
    Epoch 2/15
    33/33 [==============================] - 37s 1s/step - loss: 0.8212 - acc: 0.7388 - val_loss: 0.5605 - val_acc: 0.8278
    Epoch 3/15
    33/33 [==============================] - 38s 1s/step - loss: 0.5072 - acc: 0.8477 - val_loss: 0.4287 - val_acc: 0.8694
    Epoch 4/15
    33/33 [==============================] - 37s 1s/step - loss: 0.3887 - acc: 0.8832 - val_loss: 0.3062 - val_acc: 0.9078
    Epoch 5/15
    33/33 [==============================] - 38s 1s/step - loss: 0.2983 - acc: 0.9092 - val_loss: 0.2692 - val_acc: 0.9161
    Epoch 6/15
    33/33 [==============================] - 37s 1s/step - loss: 0.2598 - acc: 0.9199 - val_loss: 0.2336 - val_acc: 0.9298
    Epoch 7/15
    33/33 [==============================] - 37s 1s/step - loss: 0.2223 - acc: 0.9315 - val_loss: 0.1894 - val_acc: 0.9383
    Epoch 8/15
    33/33 [==============================] - 38s 1s/step - loss: 0.1883 - acc: 0.9424 - val_loss: 0.1776 - val_acc: 0.9439
    Epoch 9/15
    33/33 [==============================] - 38s 1s/step - loss: 0.1788 - acc: 0.9433 - val_loss: 0.1692 - val_acc: 0.9469
    Epoch 10/15
    33/33 [==============================] - 38s 1s/step - loss: 0.1728 - acc: 0.9461 - val_loss: 0.1658 - val_acc: 0.9464
    Epoch 11/15
    33/33 [==============================] - 38s 1s/step - loss: 0.1535 - acc: 0.9518 - val_loss: 0.1378 - val_acc: 0.9570
    Epoch 12/15
    33/33 [==============================] - 37s 1s/step - loss: 0.1350 - acc: 0.9576 - val_loss: 0.1471 - val_acc: 0.9531
    Epoch 13/15
    33/33 [==============================] - 38s 1s/step - loss: 0.1382 - acc: 0.9568 - val_loss: 0.1281 - val_acc: 0.9586
    Epoch 14/15
    33/33 [==============================] - 37s 1s/step - loss: 0.1249 - acc: 0.9620 - val_loss: 0.1149 - val_acc: 0.9626
    Epoch 15/15
    33/33 [==============================] - 38s 1s/step - loss: 0.1158 - acc: 0.9629 - val_loss: 0.1254 - val_acc: 0.9625
    
    Validation Loss and Accuracy
    8/8 [==============================] - 7s 862ms/step - loss: 0.1242 - acc: 0.9604
    




    [0.12417110800743103, 0.9603750109672546]



#### Plotting Training and Validation Accuracy and Loss Functions


```python
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
```


    
![png](/outputs/output_19_0.png)
    



    
![png](/outputs/output_19_1.png)
    


#### Predicting on Test Data using the Model


```python
# Predicting on test data
y_pred = model.predict(test_generator, steps=test_steps, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
test_file_names = test_generator.filenames
```

    28000/28000 [==============================] - 62s 2ms/step
    

#### Plotting Random Test images with their Predicted Value


```python
# Selecting 25 random indices between 0 and 27999
random_indices = [random.randint(0, 27999) for i in range(25)]
print(random_indices)

# Plotting test images on these indices with their predicted class
fig, axes = plt.subplots(1, 25, figsize=(24,3))
for i, index in enumerate(random_indices):
  image = mpimg.imread('/tmp/test/' + test_file_names[index])
  axes[i].imshow(image)
  axes[i].set_title(y_pred[index])
  axes[i].axis('off')

```

    [18212, 24623, 5449, 23008, 20596, 2573, 2055, 26804, 7760, 18883, 23062, 26317, 3084, 21341, 16067, 3867, 15982, 25471, 9693, 22859, 26702, 7666, 4116, 23350, 16854]
    


    
![png](/outputs/output_23_1.png)
    


#### Saving the Trained Model


```python
# Saving the trained model 
model.save('mnist_digit_recognizer.h5')
```

#### Predicting on New Image


```python
# Uploading new image
uploaded = files.upload()

# Predicting for each uploaded file
for file_name in uploaded.keys():
  img = image.load_img(file_name, target_size=(28, 28))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=1)

  # Printing the image and predictions
  print()
  plt.figure(figsize=(1, 1))
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  print('Prediction:', np.argmax(classes)) 
``` 

