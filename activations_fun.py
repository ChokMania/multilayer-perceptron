import numpy as np


def sigmoid(x):
	return(1 / (1 + np.exp(-x)))


def relu(x):
	return(np.max(x, 0))


def softmax(x):
	expo = np.exp(x)
	expo_sum = np.sum(np.exp(x))
	return expo / expo_sum
