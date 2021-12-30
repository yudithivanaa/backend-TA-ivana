# -*- coding: utf-8 -*-
"""CNN1.ipynb"""
from zipfile import ZipFile

""""**Ekstrak Dataset**"""

import os
import zipfile

local_zip = 'CK+48.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

"""**Set Directory **"""
base_dir = 'dataset/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training anger pictures
train_anger_dir = os.path.join(train_dir, 'anger')

# Directory with our training disgust pictures
train_disgust_dir = os.path.join(train_dir, 'disgust')

# Directory with our training fear pictures
train_fear_dir = os.path.join(train_dir, 'fear')

# Directory with our training sadness pictures
train_sadness_dir = os.path.join(train_dir, 'sadness')

# Directory with our training happy pictures
train_happy_dir = os.path.join(train_dir, 'happy')

# Directory with our training surprise pictures
train_surprise_dir = os.path.join(train_dir, 'surprise')

# Directory with our validation anger pictures
validation_anger_dir = os.path.join(validation_dir, 'anger')

# Directory with our validation disgust pictures
validation_disgust_dir = os.path.join(validation_dir, 'disgust')

# Directory with our validation fear pictures
validation_fear_dir = os.path.join(validation_dir, 'fear')

# Directory with our validation sadness pictures
validation_sadness_dir = os.path.join(validation_dir, 'sadness')

# Directory with our validation happy pictures
validation_happy_dir = os.path.join(validation_dir, 'happy')

# Directory with our validation surprise pictures
validation_surprise_dir = os.path.join(validation_dir, 'surprise')

"""**Melihat Isi Dataset**"""
train_anger_fname = os.listdir(train_anger_dir)
train_anger_fname.sort()
print(train_anger_fname[:10])

train_disgust_fname = os.listdir(train_disgust_dir)
train_disgust_fname.sort()
print(train_disgust_fname[:10])

train_fear_fname = os.listdir(train_fear_dir)
train_fear_fname.sort()
print(train_fear_fname[:10])

train_sadness_fname = os.listdir(train_sadness_dir)
train_sadness_fname.sort()
print(train_sadness_fname[:10])

train_happy_fname = os.listdir(train_happy_dir)
train_happy_fname.sort()
print(train_happy_fname[:10])

train_surprise_fname = os.listdir(train_surprise_dir)
train_surprise_fname.sort()
print(train_surprise_fname[:10])

print('total training anger images: ', len(os.listdir(train_anger_dir)))
print('total training disgust images: ', len(os.listdir(train_disgust_dir)))
print('total training fear images: ', len(os.listdir(train_fear_dir)))
print('total training sadness images: ', len(os.listdir(train_sadness_dir)))
print('total training happy images: ', len(os.listdir(train_happy_dir)))
print('total training surprise images: ', len(os.listdir(train_surprise_dir)))

print('total validation anger images: ', len(os.listdir(validation_anger_dir)))
print('total validation disgust images: ', len(os.listdir(validation_disgust_dir)))
print('total validation fear images: ', len(os.listdir(validation_fear_dir)))
print('total validation sadness images: ', len(os.listdir(validation_sadness_dir)))
print('total validation happy images: ', len(os.listdir(validation_happy_dir)))
print('total validation surprise images: ', len(os.listdir(validation_surprise_dir)))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph. we'll output images in 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_anger_pix = [os.path.join(train_anger_dir, fname)
                  for fname in train_anger_fname[pic_index - 8:pic_index]]
next_disgust_pix = [os.path.join(train_disgust_dir, fname)
                    for fname in train_disgust_fname[pic_index - 8:pic_index]]
next_fear_pix = [os.path.join(train_fear_dir, fname)
                 for fname in train_fear_fname[pic_index - 8:pic_index]]
next_sadness_pix = [os.path.join(train_sadness_dir, fname)
                    for fname in train_sadness_fname[pic_index - 8:pic_index]]
next_happy_pix = [os.path.join(train_happy_dir, fname)
                  for fname in train_happy_fname[pic_index - 8:pic_index]]
next_surprise_pix = [os.path.join(train_surprise_dir, fname)
                     for fname in train_surprise_fname[pic_index - 8:pic_index]]

for i, img_path in enumerate(
        next_anger_pix + next_disgust_pix + next_fear_pix + next_sadness_pix + next_happy_pix + next_surprise_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i=1)
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

"""**Import Library Tensorflow**"""

from tensorflow.keras import layers
from tensorflow.keras import Model

"""**Setting Arsitektur CNN (Konvolusi)**"""

# Our input feature map is 150x150x3:150x150 for the image pixels, and 3 fot
#  the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that  are 3x3
# Convolutional is followed by max-pooling layer with a  2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that  are 3x3
# Convolutional is followed by max-pooling layer with a  2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# FirThirdst convolution extracts 64 filters that  are 3x3
# Convolutional is followed by max-pooling layer with a  2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

"""**Setting Arsitektur CNN (Fully Connected)**"""

# Flatten featuremap to a 1-dim tensor so we can add Fully Connected Layers
x = layers.Flatten()(x)

# Create A fully connected layer with ReLU astivation and 512 hidden units
output = layers.Dense(512, activation='relu')(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Create output layer with a single node and sigmoid activation
# input = input feature map
#  output = input feture map + stacked convolution / maxpooling layers + fully
# connected layer + sigmoid output layer
model = Model(img_input, output)

"""**Compile CNN**"""

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

"""**Setting Preparasi Dataset**"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary'
)

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

"""**Proses Training**"""

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size*steps
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size*steps
    verbose=2
)
