import argparse
from copy import deepcopy
from neuralNetwork import neuralNetwork
import numpy as np
from utils import open_datafile, normalize, save_model, binary_cross_entropy, display
from activations_fun import sigmoid, relu, softmax

dict_act_fun = {"sigmoid": sigmoid, "relu": relu, "softmax": softmax}

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("dataset_train", type=open_datafile)
	parser.add_argument("-af", "--activation_function", help="Choose the activation function", choices={"sigmoid", "relu", "softmax"}, default="sigmoid")
	parser.add_argument("-e", "--epochs", help="Choose number of epochs", type=int, default=50)
	parser.add_argument("-p", "--patience", help="Choose patience for early stopping", type=int, default=-1)
	args = parser.parse_args()
	input_n = 30
	hidden_layers = (30, 30)
	output_n = 2
	learning_rate = 0.1

	n = neuralNetwork(input_n, output_n, hidden_layers, learning_rate, dict_act_fun.get(args.activation_function))
	epochs = args.epochs

	df = args.dataset_train.drop(args.dataset_train.columns[0], axis=1)
	size_train = int(len(df) * 0.8)
	train = df.iloc[:size_train, :]  # 80 %
	test = df.iloc[size_train + 1:, :]  # 20 %
	data = normalize(train)
	validation_data = normalize(test)

	data = np.array(data)
	validation_data = np.array(validation_data)
	best_val_loss = 1
	patience = 0
	loss_history = [binary_cross_entropy(data, n)]
	val_loss_history = [binary_cross_entropy(validation_data, n)]
	for e in range(epochs):
		for values in data:
			targets = np.zeros(output_n) + 0.01
			if values[0] == "M":
				targets[0] = 0.99
			elif values[0] == "B":
				targets[1] = 0.99
			n.train(np.array(values[1:], dtype=np.float64), targets)
		loss = binary_cross_entropy(data, n)
		val_loss = binary_cross_entropy(validation_data, n)
		loss_history.append(loss)
		val_loss_history.append(val_loss)
		print(f"epoch {e + 1:>3}/{epochs:<3} - loss: {loss:7.7f} - val_loss: {val_loss:7.7f}", end="\r")
		if loss < best_val_loss:
			best_val_loss = val_loss
			saved = deepcopy(n.w)
			patience = 0
		else:
			if patience == args.patience:
				print(f"\nEarly stopping, step back to epoch {e - args.patience}")
				n.w = saved[:]
				loss = binary_cross_entropy(data, n)
				val_loss = binary_cross_entropy(validation_data, n)
				print(f"loss: {loss:10.10f} - val_loss: {val_loss:10.10f}", end="\r")
				loss_history = loss_history[:-patience]
				val_loss_history = val_loss_history[:-patience]
				break
			patience += 1
	print()
	display(loss_history, val_loss_history)
	save_model(n)
