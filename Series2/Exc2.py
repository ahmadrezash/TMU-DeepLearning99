import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from utils import initializer, sigmoid


class MLP:
	def __init__(self, sizes, epochs=100, l_rate=0.1):
		self.sizes = sizes
		self.epochs = epochs
		self.l_rate = l_rate
		self.accuracy_history = []

		self.params = self.initialization()

	def initialization(self):
		input_layer = self.sizes[0]
		hidden_1 = self.sizes[1]
		hidden_2 = self.sizes[2]
		hidden_3 = self.sizes[3]
		output_layer = self.sizes[4]

		params = {
			'W1': initializer(hidden_1, input_layer),
			'W2': initializer(hidden_2, hidden_1),
			'W3': initializer(hidden_3, hidden_2),
			'W4': initializer(output_layer, hidden_3)
		}

		return params

	def forward(self, x_train):
		params = self.params

		params['A0'] = x_train

		params['Z1'] = np.dot(params["W1"], params['A0'])
		params['A1'] = sigmoid(params['Z1'])

		params['Z2'] = np.dot(params["W2"], params['A1'])
		params['A2'] = sigmoid(params['Z2'])

		params['Z3'] = np.dot(params["W3"], params['A2'])
		params['A3'] = sigmoid(params['Z3'])

		params['Z4'] = np.dot(params["W4"], params['A3'])
		params['A4'] = sigmoid(params['Z4'])

		return params['A4']

	def backward(self, y_train, output):
		params = self.params
		change_w = {}

		error = 2 * (output - y_train) * sigmoid(params['Z4'], derivative=True)
		change_w['W4'] = np.outer(error, params['A3'])

		error = np.dot(params['W4'].T, error) * sigmoid(params['Z3'], derivative=True)
		change_w['W3'] = np.outer(error, params['A2'])

		error = np.dot(params['W3'].T, error) * sigmoid(params['Z2'], derivative=True)
		change_w['W2'] = np.outer(error, params['A1'])

		error = np.dot(params['W2'].T, error) * sigmoid(params['Z1'], derivative=True)
		change_w['W1'] = np.outer(error, params['A0'])

		return change_w

	def update_parameters(self, changes_to_w):

		for key, value in changes_to_w.items():
			self.params[key] -= self.l_rate * value

	def get_accuracy(self, x_test, y_test):

		predictions = []

		for x, y in zip(x_test, y_test):
			output = self.forward(x)
			pred = np.argmax(output)
			predictions.append(pred == np.argmax(y))

		return np.mean(predictions)

	def train(self, x_train, y_train, x_test, y_test):
		for iteration in range(self.epochs):
			start_time = time.time()
			y_true = []
			y_pred = []
			for x, y in zip(x_train, y_train):
				output = self.forward(x)
				y_true.append(y)
				y_pred.append(output)
				changes_to_w = self.backward(y, output)
				self.update_parameters(changes_to_w)

			accuracy = self.get_accuracy(x_test, y_test)
			print(f'Epoch: {iteration + 1}, Time: {time.time() - start_time:.2f}s, Accuracy: {accuracy * 100:.2f}%')
			self.accuracy_history.append(accuracy)


data = load_iris()
x = data['data']
y = data['target']
y = LabelBinarizer().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
dnn = MLP(sizes=[4, 4, 4, 4, 3])
dnn.train(x_train, y_train, x_test, y_test)
plt.plot(dnn.accuracy_history)
plt.savefig('./acc.jpg')
