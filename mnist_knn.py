# Importing required libraries
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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

start = time.time()

# running the KNN algorithm for different k values (number of neighbors)
K = list()
accuracies = list()

for k in range(1, 40, 2):
  classifier = KNeighborsClassifier(k)
  classifier.fit(x_train, y_train)
  predictions = classifier.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)
  K.append(k)
  accuracies.append(accuracy)

end = time.time()

# plotting accuracy for different k values
plt.scatter(K, accuracies, s=50)

# getting the highest accuracy and its corresponding k
highest_accuracy = max(accuracies)
optimum_k = K[accuracies.index(highest_accuracy)]

print('k = {} produces the highest accuracy of {}'.format(optimum_k, round(highest_accuracy, 3)))
print('The processing time is {} seconds'.format(round(end - start, 2)))

