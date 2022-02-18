# Importing required libraries

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# loading the MNIST dataset
mnist = datasets.load_digits()
data, labels = mnist.data, mnist.target

# splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

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

# plotting accuracy for different k values
plt.scatter(K, accuracies, s=50)

# getting the highest accuracy and its corresponding k
highest_accuracy = max(accuracies)
optimum_k = K[accuracies.index(highest_accuracy)]

print('k = {} produces the highest accuracy of {}'.format(optimum_k, round(highest_accuracy, 3)))