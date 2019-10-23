# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:59:53 2019

@author: andy3
"""

#######################################
#
#Generator sequence with RNN(LSTM)
#
#
#
#
#
#######################################
###Download the analysze the training article.###
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

path = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))
#######################################
#
#
#
#
#
#
#
#
#######################################

###Vectorize the word's sequence.###

max_word_union=60

step=3

sentences=[]

next_chars=[]

for i in range(0,len(text)-max_word_union,step):
    
    sentences.append(text[i:i+max_word_union])
    
    next_chars.append(text[i+max_word_union])
    
print('The number of senquence:',len(sentences))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('Unique_word_union:',len(chars))

###Put the unique word_union into {} dict type means:{a,b,c...}

print('Encodeing to vectorization words one-hot encoding')

x = np.zeros((len(sentences), max_word_union, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print("The shape of x:",x.shape)

print('The shape of y:',y.shape)

###Create the Neural Network###
import keras

from keras import layers



model=keras.models.Sequential()

model.add(layers.LSTM(128,input_shape=(max_word_union,len(chars))))

model.add(layers.Dense(len(chars),activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy',optimizer=optimizer)

def sample(predicts,temperature=1.0):
    
    predicts=np.asarray(predicts).astype('float64')
    
    predicts=np.log(predicts) / temperature
    
    exp_predicts=np.exp(predicts)
    
    predicts=exp_predicts/np.sum(exp_predicts)
    
    probas=np.random.multinomial(1,predicts,1)
    
    return np.argmax(probas)

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('-----epoch:', epoch)

    start_index = random.randint(0, len(text) - max_word_union - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- temperature:', diversity)

        generated = ''
        sentence = text[start_index: start_index + max_word_union]
        generated += sentence
        print('---The random of initial word_sentence: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, max_word_union, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])
