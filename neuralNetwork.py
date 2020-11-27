import numpy as np


class neuralNetwork:

	# def __init__(self, inputnodes, hiddennodes, outputnodes, nb_hidden_layers, learningrate, activation_function):
	# 	self.inodes = inputnodes
	# 	# self.hnodes = hiddennodes
	# 	self.onodes = outputnodes
	# 	self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
	# 	self.whh = []
	# 	for _ in range(nb_hidden_layers - 1):
	# 		self.whh.append(np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.hnodes)))
	# 	self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
	# 	self.lr = learningrate
	# 	self.activation_function = activation_function

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

	def train(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		hidden_inputs = []
		hidden_outputs = []
		for i in range(len(self.w)):
			if i == 0:
				hidden_inputs.append(np.dot(self.w[i], inputs))
			else:
				hidden_inputs.append(np.dot(self.w[i], hidden_outputs[i - 1]))
			hidden_outputs.append(self.activation_function(hidden_inputs[i]))
		
		output_errors = targets - hidden_outputs[-1]
		for i in range(1, len(self.w)):
			if i == 1:
				error = output_errors
			else:
				error = np.dot(self.w[-(i - 1)].T, error)
			self.w[-i] += self.lr * np.dot((error * hidden_outputs[-i] * (1.0 - hidden_outputs[-i])), hidden_outputs[-(i + 1)].T)
		error = np.dot(self.w[0].T, error)
		self.w[0] += self.lr * np.dot((error * hidden_outputs[0] * (1.0 - hidden_outputs[0])), inputs.T)

	def query(self, inputs_lists):
		value = np.array(inputs_lists, ndmin=2).T
		for weight in self.w:
			value = self.activation_function(np.dot(weight, value))
		return value

