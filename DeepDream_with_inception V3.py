# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:23:58 2019

@author: andy3
"""

#############################################
#
#DeepDream in Keras
#
#
#Use Inception V3 to achieve the graph generator
#
#
############################################

###Download the inception V3 model###

from keras.applications import inception_v3
from keras import layers,models
from keras import backend as K

K.set_learning_phase(0)#We don't train this model

#Use the Inception V3 as our model
model=inception_v3.InceptionV3(weights='imagenet',
                               include_top=False)#include_top=False didn't include the top layers

###Choose the Inception V3 model of weights(4 layers in parallel)
layer_contributions={
        'mixed2':3.,
        'mixed3':3.,
        'mixed4':2.,
        'mixed5':1.5,
}
#Show the structure of Inception V3 after setting
model.summary()

###Define the loss maximize###

#Create the layers of names as dict
layer_dict=dict([(layer.name,layer) for layer in model.layers])

#use the backend K to create the scalar varable which initial value is zero,in training, the loss quantity will add there.
loss=K.variable(0.)

for layer_name in layer_contributions:
    
    coeff=layer_contributions[layer_name]
    
    activation=layer_dict[layer_name].output
    
    scaling=K.prod(K.cast(K.shape(activation),'float32'))
    
    loss=loss+coeff*K.sum(K.square(activation[:,2:-2,2:-2,:]))/scaling

###Upper granient method###
dream=model.input
print(dream.shape)

grads=K.gradients(loss,dream)[0]
grads /=K.maximum(K.mean(K.abs(grads)),1e-7)

#Define the keras function to get the loss and gradient quantity
outputs=[loss,grads]
fetch_loss_and_grads=K.function([dream],outputs)

def eval_loss_and_grads(x):
    outs=fetch_loss_and_grads([x])
    loss_value=outs[0]
    grad_values=outs[1]
    return loss_value,grad_values

def gradient_ascent(x,iterations,step,max_loss=None):
    for i in range(iterations):
        loss_value,grad_values=eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('....Loss value at:',i,':',loss_value)
        print('....grad value at:',i,':',grad_values)
        x +=step*grad_values
    return x

import scipy
import numpy as np
from keras.preprocessing import image

#Prepare the graph into function
def preprocess_image(image_path):
    img=image.load_img(image_path)
    img=image.img_to_array(img)
    print(img.shape)
    img=np.expand_dims(img,axis=0)
    print(img.shape)
    img=inception_v3.preprocess_input(img)
    return img

#Let Inception V3 to the anti 
def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#進行圖片縮放

import imageio
def resize_img(img,size):
    img=np.copy(img)
    factors=(1,
             float(size[0])/img.shape[1],
             float(size[1])/img.shape[2],
             1)
    return scipy.ndimage.zoom(img,factors,order=1)
#save the graph
def save_img(img,fname):
    pil_img=deprocess_image(np.copy(img))
    imageio.imwrite(fname,pil_img)

#To implement the gradient
step=0.003
num_octave=8
octave_scale=1.4
iterations=20
max_loss=10.

base_image_path='original_photo_deep_dream.jpg'
img=preprocess_image(base_image_path)

original_shape=img.shape[1:3]
successive_shapes=[original_shape]
for i in range(1,num_octave):
    shape=tuple([int(dim/(octave_scale**i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes=successive_shapes[::-1]

original_img=np.copy(img)

shrunk_original_img=resize_img(img,successive_shapes[0])

for shape in successive_shapes:
    print('image_shape:',shape)
    img=resize_img(img,shape)
    img=gradient_ascent(img,
                        iterations=iterations,
                        step=step,
                        max_loss=max_loss)
    upscaled_shrunk_original_img=resize_img(shrunk_original_img,
                                            shape)
    same_size_original=resize_img(original_img,shape)
    lost_detail=same_size_original-upscaled_shrunk_original_img
    img+=lost_detail
    shrunk_original_img=resize_img(original_img,shape)
    save_img(img,
             fname='draem_at_scale_'+str(shape)+'.png')
save_img(img,fname='final_dream.png')

from matplotlib import pyplot as plt

plt.imshow(deprocess_image(np.copy(img)))

plt.show()