from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
import time

# x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# x = (x / 255).astype('float32')

# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

from utils import sigmoid, initializer


class MLP:
	def __init__(self, sizes, epochs=3, l_rate=0.001):
		self.sizes = sizes
		self.epochs = epochs
		self.l_rate = l_rate

		# we save all parameters in the neural network in this dictionary
		self.params = self.initialization()

	def initialization(self):
		# number of nodes in each layer
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

	def forward_pass(self, x_train):
		params = self.params

		# input layer activations becomes sample
		params['A0'] = x_train

		# input layer to hidden layer 1
		params['Z1'] = np.dot(params["W1"], params['A0'])
		params['A1'] = sigmoid(params['Z1'])

		# hidden layer 1 to hidden layer 2
		params['Z2'] = np.dot(params["W2"], params['A1'])
		params['A2'] = sigmoid(params['Z2'])

		# hidden layer 2 to hidden layer 3
		params['Z3'] = np.dot(params["W3"], params['A2'])
		params['A3'] = sigmoid(params['Z3'])

		# hidden layer 2 to output layer
		params['Z4'] = np.dot(params["W4"], params['A3'])
		params['A4'] = sigmoid(params['Z4'])

		return params['A4']

	def backward_pass(self, y_train, output):
		params = self.params
		change_w = {}

		# Calculate W4 update
		error = 2 * (output - y_train) * sigmoid(params['Z4'], derivative=True)
		change_w['W4'] = np.outer(error, params['A3'])

		# Calculate W3 update
		error = np.dot(params['W4'].T, error) * sigmoid(params['Z3'], derivative=True)
		change_w['W3'] = np.outer(error, params['A2'])

		# Calculate W2 update
		error = np.dot(params['W3'].T, error) * sigmoid(params['Z2'], derivative=True)
		change_w['W2'] = np.outer(error, params['A1'])

		# Calculate W1 update
		error = np.dot(params['W2'].T, error) * sigmoid(params['Z1'], derivative=True)
		change_w['W1'] = np.outer(error, params['A0'])

		return change_w

	def update_network_parameters(self, changes_to_w):

		for key, value in changes_to_w.items():
			self.params[key] -= self.l_rate * value

	def compute_accuracy(self, x_val, y_val):

		predictions = []

		for x, y in zip(x_val, y_val):
			output = self.forward_pass(x)
			pred = np.argmax(output)
			predictions.append(pred == np.argmax(y))

		return np.mean(predictions)

	def train(self, x_train, y_train, x_val, y_val):
		start_time = time.time()
		for iteration in range(self.epochs):
			y_true = []
			y_pred = []
			for x, y in zip(x_train, y_train):
				output = self.forward_pass(x)
				y_true.append(y)
				y_pred.append(output)
				changes_to_w = self.backward_pass(y, output)
				self.update_network_parameters(changes_to_w)

			accuracy = self.compute_accuracy(x_val, y_val)
			print(f'Epoch: {iteration + 1}, Time Spent: {time.time() - start_time:.2f}s, Accuracy: {accuracy * 100:.2f}%')


from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer

data = load_iris()
x = data['data']
y = data['target']
y = LabelBinarizer().fit_transform(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
dnn = MLP(sizes=[4, 4, 4, 4, 3])
dnn.train(x_train, y_train, x_val, y_val)
