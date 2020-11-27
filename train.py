import argparse
from neuralNetwork import neuralNetwork
import numpy as np
from utils import open_datafile, normalize, save_model, binary_cross_entropy
from activations_fun import sigmoid, relu, softmax

dict_act_fun = {"sigmoid": sigmoid, "relu": relu, "softmax": softmax}


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("dataset_train", type=open_datafile)
	parser.add_argument("-af", "--activation_function", help="Choose the activation function", choices={"sigmoid", "relu", "softmax"})
	args = parser.parse_args()

	input_n = 30
	hidden_layers = (10, 20)
	# hidden_n = 30
	output_n = 2
	learning_rate = 0.1

	# n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, nb_hidden_layers, learning_rate, dict_act_fun.get(args.activation_function))
	n = neuralNetwork(input_n, output_n, hidden_layers, learning_rate, dict_act_fun.get(args.activation_function))
	epochs = 50

	df = args.dataset_train.drop(args.dataset_train.columns[0], axis=1)
	size_train = int(len(df) * 0.8)
	train = df.iloc[:size_train, :]  # 80 %
	test = df.iloc[size_train + 1:, :]  # 20 %
	data = normalize(train)
	validation_data = normalize(test)

	data = np.array(data)
	validation_data = np.array(validation_data)
	best_var_loss = 1
	before = np.copy(n.w[0])
	for e in range(epochs):
		for values in data:
			targets = np.zeros(output_n) + 0.01
			if values[0] == "M":
				targets[0] = 0.99
			elif values[0] == "B":
				targets[1] = 0.99
			# save
			n.train(np.array(values[1:], dtype=np.float64), targets)
		loss = binary_cross_entropy(data, n)
		var_loss = binary_cross_entropy(validation_data, n)
		print(f"epoch {e + 1:>3}/{epochs:<3} - loss: {loss:7.7f} - val_loss: {var_loss:7.7f}", end="\r")
		# if loss < best_var_loss:
		# 	best_var_loss = loss
		# else:
		# 	print(f"\nEarly stopping, step back to epoch {e}")
		# 	break
	print("\nEnd")
	save_model(n)
