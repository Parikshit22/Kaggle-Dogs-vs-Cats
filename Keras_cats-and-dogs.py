# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:02:57 2019

@author: MUJ
"""

import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
import itertools

train_path = "cats-and-dogs/train"
test_path = "cats-and-dogs/test"
valid_path = "cats-and-dogs/valid"

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224), classes = ['dog','cat'], batch_size = 10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224), classes = ['dog','cat'], batch_size = 10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size = (224,224), classes = ['dog','cat'], batch_size = 4)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

imgs, labels = next(train_batches)
plots(imgs,titles = labels)

model = Sequential([Conv2D(32,(3,3),input_shape = (224,224,3),activation = 'relu'),Flatten(),Dense(2,activation= "softmax")])
model.compile(Adam(lr = 0.001), loss = "categorical_crossentropy", metrics = ['accuracy'])
model.fit_generator(train_batches, steps_per_epoch = 4, validation_data = valid_batches, validation_steps = 4,epochs = 5, verbose = 2)

test_images, test_labels = next(test_batches)
test_labels = test_labels[:,0]
test_labels
pridiction = model.predict_generator(test_batches, steps = 1 , verbose=0)

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()
model= Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
model.layers.pop()

for layers in model.layers:
    layers.trainable = False
model.add(Dense(2,activation = "softmax"))
model.summary()
model.compile(Adam(lr =0.01),loss = ("categorical_crossentropy"), metrics = ["accuracy"])
model.fit_generator(train_batches,  steps_per_epoch = 4, validation_data = valid_batches, validation_steps = 4,epochs = 5, verbose = 2)



