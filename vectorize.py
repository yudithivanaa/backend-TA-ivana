from keras.preprocessing.image import array_to_img, img_to_array, load_img

import os

for fname in os.listdir('/Dataset/train/anger'):
  print(fname)
