import numpy as np
from activations_fun import softmax

# class neuralNetwork:

# 	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, activation_function):
# 		self.inodes = inputnodes
# 		self.hnodes = hiddennodes
# 		self.onodes = outputnodes
# 		self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
# 		self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
# 		self.lr = learningrate
# 		self.activation_function = activation_function

# 	def train(self, inputs_list, targets_list):
# 		inputs = np.array(inputs_list, ndmin=2).T
# 		targets = np.array(targets_list, ndmin=2).T
# 		hidden_inputs = np.dot(self.wih, inputs)
# 		hidden_outputs = self.activation_function(hidden_inputs)
# 		final_inputs = np.dot(self.who, hidden_outputs)
# 		final_outputs = self.activation_function(final_inputs)
# 		output_errors = targets - final_outputs
# 		hidden_errors = np.dot(self.who.T, output_errors)
# 		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
# 		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

# 	def query(self, inputs_lists):
# 		inputs = np.array(inputs_lists).T
# 		hidden_inputs = np.dot(self.wih, inputs)
# 		hidden_outputs = self.activation_function(hidden_inputs)
# 		final_inputs = np.dot(self.who, hidden_outputs)
# 		final_outputs = self.activation_function(final_inputs)
# 		return final_outputs


class neuralNetwork:

	def __init__(self, n_input, n_output, hidden_layers, learningrate, activation_function):
		self.input = n_input
		self.output = n_output
		self.hidden = hidden_layers
		if type(self.hidden) is tuple:
			self.w = [np.random.normal(0.0, pow(self.input, -0.5), (self.hidden[0], self.input))]
			for i in range(len(self.hidden) - 1):
				self.w.append(np.random.normal(0.0, pow(self.hidden[i], -0.5), (self.hidden[i + 1], self.hidden[i])))
			self.w.append(np.random.normal(0.0, pow(self.hidden[-1], -0.5), (self.output, self.hidden[-1])))
		else:
			self.w = [np.random.normal(0.0, pow(self.input, -0.5), (self.hidden, self.input))]
			self.w.append(np.random.normal(0.0, pow(self.hidden, -0.5), (self.output, self.hidden)))
		self.lr = learningrate
		self.activation_function = activation_function

	def feedforward(self, inputs):
		hidden_inputs = []
		hidden_outputs = []
		for i in range(len(self.w)):
			if i == 0:
				hidden_inputs.append(np.dot(self.w[i], inputs))
			else:
				hidden_inputs.append(np.dot(self.w[i], hidden_outputs[i - 1]))
			hidden_outputs.append(self.activation_function(hidden_inputs[i]))
		hidden_outputs.insert(0, inputs)
		return hidden_outputs


	def backward_propagation(self, output_errors, hidden_outputs, inputs):
		for i in range(1, len(self.w) + 1):
			if i == 1:
				error = output_errors
			else:
				error = np.dot(self.w[-(i - 1)].T, error)
			self.w[-i] += self.lr * np.dot((error * hidden_outputs[-i] * (1.0 - hidden_outputs[-i])), hidden_outputs[-(i + 1)].T)


	def train(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		hidden_outputs = self.feedforward(inputs)
		output_errors = targets - hidden_outputs[-1]
		self.backward_propagation(output_errors, hidden_outputs, inputs)

	def query(self, inputs_lists):
		value = np.array(inputs_lists, ndmin=2).T
		for layer in range(len(self.w)):
			if layer == len(self.w) - 1:
				value = softmax(np.dot(self.w[layer], value))
			else:
				value = self.activation_function(np.dot(self.w[layer], value))
		return value
