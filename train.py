"""

This code written by IMALICE
https://github.com/imalic3

"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint
from visualization import training_visualization as vs
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

batch_size = 100
num_classes = 10
epochs = 50
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten input 28 * 28 into hot vector
x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)

# normalize the RGB colors from range 0-255 into 0-1, 
# which is reducing the floating number in weight matrix.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# convert class vectors to binary class matrices
# ex. 5 => [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# display utils 
tensorBoard = TensorBoard(log_dir='./Graph', histogram_freq=0, 
    write_graph=True, write_images=True)
filepath = "model/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
modelCallback = ModelCheckpoint(filepath, monitor='val_acc', 
    verbose=1, save_best_only=True, mode='auto')

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(img_rows * img_cols,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(x_test, y_test), 
    callbacks=[tensorBoard, modelCallback])

# final testing...
score = model.evaluate(x_test, y_test)
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])