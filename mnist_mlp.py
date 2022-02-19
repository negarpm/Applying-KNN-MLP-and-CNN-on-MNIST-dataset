# Importing required libraries
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import time

# loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_shape = x_train.shape
x_test_shape = x_test.shape

# reshaping the dataset
x_train = np.reshape(x_train, (x_train_shape[0], x_train_shape[1] * x_train_shape[2]))
x_test = np.reshape(x_test, (x_test_shape[0], x_test_shape[1] * x_test_shape[2]))

# converting the pixel values from 0 to 1
x_train = x_train / 255
x_test = x_test / 255

num_classes = 10
# converting the labels to one-hot encoding format
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# dividing the training data to training and validation subsets
x_val = x_train[:20000, :]
y_val = y_train[:20000, :]
x_train = x_train[20000:, :]
y_train = y_train[20000:, :]

# creating the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train[0].shape)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

start = time.time()

history = model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_val, y_val), verbose=2)

end = time.time()

accuracy = model.evaluate(x_test, y_test, verbose=2)

print('classification accuracy is {}%'.format(round(accuracy[1] * 100, 2)))

print('The processing time is {} seconds'.format(round(end - start, 2)))
