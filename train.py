import argparse
from neuralNetwork import neuralNetwork
import numpy as np
from utils import open_datafile, normalize, sigmoid, save_model, binary_cross_entropy


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("dataset_train", type=open_datafile)
	args = parser.parse_args()

	input_nodes = 30
	# nb_hidden_layers = 4
	hidden_nodes = 30
	output_nodes = 2
	learning_rate = 0.1

	n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, sigmoid)
	epochs = 180

	df = args.dataset_train.drop(args.dataset_train.columns[0], axis=1)
	size_train = int(len(df) * 0.8)
	train = df.iloc[:size_train, :]  # 80 %
	test = df.iloc[size_train + 1:, :]  # 20 %
	data = normalize(train)
	validation_data = normalize(test)

	data = np.array(data)
	validation_data = np.array(validation_data)
	best_var_loss = 1
	for e in range(epochs):
		for values in data:
			targets = np.zeros(output_nodes) + 0.01
			if values[0] == "M":
				targets[0] = 0.99
			elif values[0] == "B":
				targets[1] = 0.99
			# save
			n.train(np.array(values[1:], dtype=np.float64), targets)
		loss = binary_cross_entropy(data, n)
		var_loss = binary_cross_entropy(validation_data, n)
		print(f"epoch {e + 1:>3}/{epochs:<3} - loss: {loss:7.7f} - val_loss: {var_loss:7.7f}", end="\r")
		# if var_loss < best_var_loss:
		# 	best_var_loss = var_loss
		# else:
		# 	print(f"\nEarly stopping, step back to epoch {e}")
		# 	break
	print("\nEnd")
	save_model(n)
