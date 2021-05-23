# Rock Paper Scissors

#### Importing Required Files


```python
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from google.colab import files
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
```

#### Downloading the Train and Test Data


```python
# Downloading the train data into temporal storage
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip -O /tmp/rps.zip
# Downloading the test data into temporal storage
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip -O /tmp/rps-test-set.zip
```

    --2021-05-11 12:39:33--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.203.128, 74.125.204.128, 64.233.188.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.203.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 200682221 (191M) [application/zip]
    Saving to: ‘/tmp/rps.zip’
    
    /tmp/rps.zip        100%[===================>] 191.38M   134MB/s    in 1.4s    
    
    2021-05-11 12:39:35 (134 MB/s) - ‘/tmp/rps.zip’ saved [200682221/200682221]
    
    --2021-05-11 12:39:35--  https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.203.128, 74.125.204.128, 64.233.188.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.203.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 29516758 (28M) [application/zip]
    Saving to: ‘/tmp/rps-test-set.zip’
    
    /tmp/rps-test-set.z 100%[===================>]  28.15M   102MB/s    in 0.3s    
    
    2021-05-11 12:39:36 (102 MB/s) - ‘/tmp/rps-test-set.zip’ saved [29516758/29516758]
    
    

#### Extracting the Train and Test Data Zip Files


```python
# Extracting the train data
local_zip = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

# Extracting the test data
local_zip = '/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
```

#### Getting Directory Paths for Each Class


```python
# Setting up class directories for training data
rock_dir = '/tmp/rps/rock'
paper_dir = '/tmp/rps/paper'
scissors_dir = '/tmp/rps/scissors'

# Printing number of images in each class
print('Total training rock images: ', len(os.listdir(rock_dir)))
print('Total training paper images: ', len(os.listdir(paper_dir)))
print('Total training scissors images: ', len(os.listdir(scissors_dir)))

# getting list of image names for each class
rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)
```

    Total training rock images:  840
    Total training paper images:  840
    Total training scissors images:  840
    

#### Printing Image for Each Class for Given Index


```python
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
```


    
![png](/outputs/output_10_0.png)
    



    
![png](/outputs/output_10_1.png)
    



    
![png](/outputs/output_10_2.png)
    


#### Setting up Image Data Generators for Training and Validation Data


```python
# Setting up train directory and its ImageDataGenerator
TRAINING_DIR = '/tmp/rps'
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
VALIDATION_DIR = '/tmp/rps-test-set'
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
```

    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.
    

#### Setting up the Deep Neural Network Structure


```python
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
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 148, 148, 64)      1792      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 74, 74, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 72, 72, 64)        36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 34, 34, 128)       73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 17, 17, 128)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 15, 15, 128)       147584    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 6272)              0         
    _________________________________________________________________
    dropout (Dropout)            (None, 6272)              0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               3211776   
    _________________________________________________________________
    dense_1 (Dense)              (None, 3)                 1539      
    =================================================================
    Total params: 3,473,475
    Trainable params: 3,473,475
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/25
    20/20 [==============================] - 32s 1s/step - loss: 1.5888 - accuracy: 0.3290 - val_loss: 1.0914 - val_accuracy: 0.3871
    Epoch 2/25
    20/20 [==============================] - 24s 1s/step - loss: 1.0894 - accuracy: 0.3947 - val_loss: 1.0780 - val_accuracy: 0.3495
    Epoch 3/25
    20/20 [==============================] - 24s 1s/step - loss: 1.0473 - accuracy: 0.4606 - val_loss: 1.0507 - val_accuracy: 0.6102
    Epoch 4/25
    20/20 [==============================] - 24s 1s/step - loss: 0.9898 - accuracy: 0.5350 - val_loss: 0.7843 - val_accuracy: 0.6559
    Epoch 5/25
    20/20 [==============================] - 24s 1s/step - loss: 0.8166 - accuracy: 0.6294 - val_loss: 0.3744 - val_accuracy: 0.9892
    Epoch 6/25
    20/20 [==============================] - 24s 1s/step - loss: 0.7459 - accuracy: 0.6754 - val_loss: 0.2393 - val_accuracy: 0.9597
    Epoch 7/25
    20/20 [==============================] - 24s 1s/step - loss: 0.5665 - accuracy: 0.7629 - val_loss: 0.1534 - val_accuracy: 1.0000
    Epoch 8/25
    20/20 [==============================] - 24s 1s/step - loss: 0.6186 - accuracy: 0.7544 - val_loss: 0.1665 - val_accuracy: 0.9919
    Epoch 9/25
    20/20 [==============================] - 24s 1s/step - loss: 0.4767 - accuracy: 0.8271 - val_loss: 0.0957 - val_accuracy: 1.0000
    Epoch 10/25
    20/20 [==============================] - 24s 1s/step - loss: 0.2897 - accuracy: 0.8945 - val_loss: 0.2646 - val_accuracy: 0.8952
    Epoch 11/25
    20/20 [==============================] - 24s 1s/step - loss: 0.3628 - accuracy: 0.8373 - val_loss: 0.6575 - val_accuracy: 0.6640
    Epoch 12/25
    20/20 [==============================] - 23s 1s/step - loss: 0.4321 - accuracy: 0.8255 - val_loss: 0.0440 - val_accuracy: 0.9866
    Epoch 13/25
    20/20 [==============================] - 24s 1s/step - loss: 0.2081 - accuracy: 0.9273 - val_loss: 0.3069 - val_accuracy: 0.8226
    Epoch 14/25
    20/20 [==============================] - 24s 1s/step - loss: 0.2318 - accuracy: 0.9175 - val_loss: 0.0582 - val_accuracy: 0.9758
    Epoch 15/25
    20/20 [==============================] - 24s 1s/step - loss: 0.2707 - accuracy: 0.8975 - val_loss: 0.0664 - val_accuracy: 0.9946
    Epoch 16/25
    20/20 [==============================] - 24s 1s/step - loss: 0.1590 - accuracy: 0.9400 - val_loss: 0.0349 - val_accuracy: 1.0000
    Epoch 17/25
    20/20 [==============================] - 24s 1s/step - loss: 0.1245 - accuracy: 0.9586 - val_loss: 0.0257 - val_accuracy: 0.9919
    Epoch 18/25
    20/20 [==============================] - 24s 1s/step - loss: 0.1264 - accuracy: 0.9504 - val_loss: 0.0160 - val_accuracy: 1.0000
    Epoch 19/25
    20/20 [==============================] - 24s 1s/step - loss: 0.0851 - accuracy: 0.9693 - val_loss: 0.0138 - val_accuracy: 1.0000
    Epoch 20/25
    20/20 [==============================] - 23s 1s/step - loss: 0.0863 - accuracy: 0.9729 - val_loss: 0.0097 - val_accuracy: 1.0000
    Epoch 21/25
    20/20 [==============================] - 24s 1s/step - loss: 0.1087 - accuracy: 0.9667 - val_loss: 0.1087 - val_accuracy: 0.9597
    Epoch 22/25
    20/20 [==============================] - 24s 1s/step - loss: 0.2432 - accuracy: 0.9208 - val_loss: 0.0620 - val_accuracy: 0.9731
    Epoch 23/25
    20/20 [==============================] - 24s 1s/step - loss: 0.0508 - accuracy: 0.9893 - val_loss: 0.0165 - val_accuracy: 0.9946
    Epoch 24/25
    20/20 [==============================] - 24s 1s/step - loss: 0.2230 - accuracy: 0.9221 - val_loss: 0.0302 - val_accuracy: 0.9892
    Epoch 25/25
    20/20 [==============================] - 24s 1s/step - loss: 0.0532 - accuracy: 0.9809 - val_loss: 0.0428 - val_accuracy: 0.9892
    

#### Plotting Training and Validation Accuracies and Loss Functions


```python
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
```


    
![png](/outputs/output_16_0.png)
    



    
![png](/outputs/output_16_1.png)
    


#### Saving the Trained Model


```python
# Saving the trained model 
model.save('rock_paper_scissors.h5')
```

#### Predicting on New Image


```python
# Uploading new image
uploaded = files.upload()

print()
print('Class Order [Paper  Rock  Scissors]\n')

# Predicting for each uploaded file
for file_name in uploaded.keys():
  img = image.load_img(file_name, target_size=(150, 150))  # Loading image from path
  x = image.img_to_array(img)                              # Converting image pixel intensities to array
  x = np.expand_dims(x, axis=0)                            # Transforming into 1-D array from 2-D array

  # Stacking multiple images vertically to feed into model to predict
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)

  # Printing the predictions
  print(file_name)
  print(classes) 
  print()

```



<input type="file" id="files-579578b7-f657-4393-9165-0fca5c46bfff" name="files[]" multiple disabled
   style="border:none" />
<output id="result-579578b7-f657-4393-9165-0fca5c46bfff">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    
    Class Order [Paper  Rock  Scissors]
    
    
