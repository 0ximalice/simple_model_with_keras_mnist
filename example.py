"""

This code written by IMALICE
https://github.com/imalic3

"""

import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from visualization import load_image
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

img_rows, img_cols = 28, 28
num_classes = 10

if len(sys.argv) >= 2:
    images = []
    try:
        path = sys.argv[1]
        files = os.listdir(path) 
        for img in files:
            try:
                X = load_image('%s/%s' % (path, img))
                X = X.reshape(1, img_rows * img_cols)
                X = X.astype('float32') / 255
                images.append((img, X))
            except Exception:
                pass

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(img_rows * img_cols,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # load pre-trained weight
        model.load_weights('model/with_dropout.hdf5')

        for img, XX in images:
            print('Filename : ', img)
            print('Number : ', numpy.argmax(model.predict([XX])[0]))
            print('----------------------------')

    except Exception as e:
        print(e)
        print('It\'s not a folder')

else:
    print('Please assign an image path.')