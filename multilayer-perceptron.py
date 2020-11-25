import argparse
import numpy as np
import pandas as pd
import sys


def sigmoid(x):
	return(1 / (1 + np.exp(-x)))


class neuralNetwork:

	# initialise the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# set number of nodes in each input, hidden, output layer
		self.inodes = inputnodes
		# list
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		# link weight matrices, wih and who
		# weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
		# w11 w21
		# w12 w22 etc
		self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
		# list
		# self.whh
		self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
		# learning rate
		self.lr = learningrate
		# activation function is the sigmoid function
		self.activation_function = lambda x: sigmoid(x)
		pass

	# train the neural network

	def train(self, inputs_list, targets_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		# output layer error is the (target - actual)
		output_errors = targets - final_outputs
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = np.dot(self.who.T, output_errors)

		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

		# update the weights for the links between the input and hidden layers
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

		pass

	# query the neural network

	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs


def open_datafile(datafile):
	try:
		# cols = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_ste", "texture_ste", "perimeter_ste", "area_ste", "smoothness_ste", "compactness_ste", "concavity_ste", "concave points_ste", "symmetry_ste", "fractal_dimension_ste", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
		data = pd.read_csv(datafile, header=None)  # , names=cols)
	except pd.errors.EmptyDataError:
		sys.exit("Empty data file.")
	except pd.errors.ParserError:
		sys.exit("Error parsing file, needs to be a well formated csv.")
	except:
		sys.exit(f"File {datafile} corrupted or does not exist.")
	return data


def normalize(df):
	result = df.copy()
	for feature_name in df.columns[1:]:
		if feature_name != 0:
			max_value = df[feature_name].max()
			# min_value = df[feature_name].min()²
			result[feature_name] = (df[feature_name] / max_value * 0.99) + 0.01
			# result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("data", type=open_datafile)
	args = parser.parse_args()

	input_nodes = 30
	# nb_hidden_layers = 4
	hidden_nodes = 30
	output_nodes = 2
	learning_rate = 0.1

	n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
	epochs = 50

	df = args.data.drop(args.data.columns[0], axis=1)
	train = df.iloc[:454, :]  # 80 %
	test = df.iloc[455:, :]  # 20 %

	data = normalize(train)
	data = np.array(data)
	for e in range(epochs):
		for values in np.array(data):
			targets = np.zeros(output_nodes) + 0.01
			if values[0] == "M":
				targets[0] = 0.99
			elif values[0] == "B":
				targets[1] = 0.99
			n.train(np.array(values[1:], dtype=np.float64), targets)
		pass

	scorecard = []

	test = normalize(test)
	test = np.array(test)

	# go through all the records in the test data set
	for values in np.array(test):
		correct_label = values[0]
		outputs = n.query(np.array(values[1:], dtype=np.float64))
		label = np.argmax(outputs)
		if label == 0:
			label = "M"
		else:
			label = "B"
		if (label == correct_label):
			scorecard.append(1)
		else:
			scorecard.append(0)
	scorecard_array = np.asarray(scorecard)
	print("performance = ", scorecard_array.sum() / scorecard_array.size)
