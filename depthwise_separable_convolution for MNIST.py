# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:59:15 2019

@author: andy3
"""

#################################
#
#
#depthwise separable convolution sturcture
#
#
#
#
#################################
from keras.datasets import mnist

from keras.utils import to_categorical

#we input the data from MNIST datasets.

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#Next,we need to deal with the datas as suitable array (match the CNN as inputing data)
#For fitting the CNN, we need the reshape the origin data(ex:train_images.shape=(60000,28,28)->(60000,28,28,1))
#(60000->the number of graph,28->length,28->width,1->RGB number(because our data is black and ))

train_images=train_images.reshape((60000,
                                   28,
                                   28,
                                   1))

train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,
                                 28,
                                 28,
                                 1))

test_images=test_images.astype('float32')/255

train_labels=to_categorical(train_labels)

test_labels=to_categorical(test_labels)
##Create the depthwise separable convolution##

from keras.models import Sequential,Model
from keras import layers,models

#Use Sequential method to create the model

model=models.Sequential()

model.add(layers.SeparableConv2D(32,
                                 (3,3),
                                 activation='relu',
                                 input_shape=(28,28,1)))

model.add(layers.SeparableConv2D(64,3,activation='relu'))

model.add(layers.MaxPool2D(2,2))

model.add(layers.SeparableConv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPool2D(2,2))

model.add(layers.SeparableConv2D(64,(3,3),activation='relu'))

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

from keras.utils import plot_model

plot_model(model,show_shapes=True,to_file='model.png')

history=model.fit(train_images,
                  train_labels,
                  epochs=20,
                  batch_size=64)

#After training our CNN, we use the test_images to test CNN.

test_loss,test_acc=model.evaluate(test_images,
                                  test_labels)

#Print the accuracy and loss quantity of training

import matplotlib.pyplot as plt

acc=history.history['acc']

loss=history.history['loss']

epochs=range(1,
             len(acc)+1)

plt.plot(epochs,acc,
         'bo',
         label='Traning accuracy')

plt.title('The accuracy of training ')

plt.legend()

plt.figure()

plt.plot(epochs,loss,
         'bo',
         label='Training loss quantity')

plt.title('Training loss')

plt.legend()

plt.show()

print('The accuracy of test_images on CNN:')

print(test_acc)

print('The loss of test_images on CNN:')

print(test_loss)