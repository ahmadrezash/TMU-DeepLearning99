import numpy as np


def initializer(input_dim, output_dim):
	np.random.seed(0)
	w = np.random.normal(0, 1, (input_dim, output_dim))
	b = np.random.normal(0, 1, 1)[0]

	return w


def sigmoid(x, derivative=False):
	if derivative:
		return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
	return 1 / (1 + np.exp(-x))
