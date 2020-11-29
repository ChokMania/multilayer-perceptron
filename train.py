import argparse
from copy import deepcopy
from neuralNetwork import neuralNetwork
import numpy as np
from utils import open_datafile, normalize, save_model, binary_cross_entropy, display, check_hidden_layer
from activations_fun import sigmoid
import matplotlib.pyplot as plt


def fit(args, n):
	epochs = args.epochs
	df = args.dataset_train.drop(args.dataset_train.columns[0], axis=1)
	size_train = int(len(df) * 0.8)
	train = df.iloc[:size_train, :]  # 80 %
	test = df.iloc[size_train + 1:, :]  # 20 %
	data = normalize(train)
	validation_data = normalize(test)

	data = np.array(data)
	validation_data = np.array(validation_data)
	best_val_loss = 10
	best_loss = 10
	patience = 0
	loss, acc = binary_cross_entropy(data, n)
	n.loss.append(loss)
	n.acc.append(acc)
	val_loss, val_acc = binary_cross_entropy(validation_data, n)
	n.val_loss.append(val_loss)
	n.val_acc.append(val_acc)
	print(f"epoch {e:>3}/{epochs:<3} - loss: {loss:10.10f} - acc {acc:5.5f} - val_loss: {val_loss:10.10f} - val_acc {val_acc:5.5f}", end="\r")
	for e in range(epochs):
		for values in data:
			targets = np.zeros(output_n) + 0.01
			if values[0] == "M":
				targets[0] = 0.99
			elif values[0] == "B":
				targets[1] = 0.99
			n.train(np.array(values[1:], dtype=np.float64), targets)
		loss, acc = binary_cross_entropy(data, n)
		n.loss.append(loss)
		n.acc.append(acc)
		val_loss, val_acc = binary_cross_entropy(validation_data, n)
		n.val_loss.append(val_loss)
		n.val_acc.append(val_acc)
		print(f"epoch {e + 1:>3}/{epochs:<3} - loss: {loss:10.10f} - acc {acc:5.5f} - val_loss: {val_loss:10.10f} - val_acc {val_acc:5.5f}", end="\r")
		if val_loss < best_val_loss and loss < best_loss:
			best_val_loss = val_loss
			best_loss = loss
			saved = deepcopy(n.w)
			patience = 0
		else:
			if patience == args.patience:
				epoch = e - args.patience - 1
				print(f"\nEarly stopping, step back to epoch {epoch}")
				n.w = saved[:]
				n.loss = n.loss[:epoch + 2]
				n.val_loss = n.val_loss[:epoch + 2]
				n.acc = n.acc[:epoch + 2]
				n.val_acc = n.val_acc[:epoch + 2]
				print(f"loss: {n.loss[-1]:10.10f} - acc {n.acc[-1]:5.5f} - val_loss: {n.val_loss[-1]:10.10f} - val_acc {n.val_acc[-1]:5.5f}")
				break
			patience += 1


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("dataset_train", type=open_datafile)
	parser.add_argument("-e", "--epochs", metavar="n", help="Choose number of epochs", type=int, default=50)
	parser.add_argument("-p", "--patience", metavar="n", help="Choose patience for early stopping", type=int, default=-1)
	parser.add_argument("-hl", "--hidden_layer", metavar="(n1, n2, ...)", help="Make your own hidden layers", type=check_hidden_layer, default=(10, 10))
	parser.add_argument("-vi", "--visu", help="Display graphs", action="store_true")
	args = parser.parse_args()
	input_n = 30
	output_n = 2
	hidden_layers = args.hidden_layer
	learning_rate = 0.1
	n = neuralNetwork(input_n, output_n, hidden_layers, learning_rate, sigmoid)

	fit(args, n)
	print()
	if args.visu is True:
		fig1, ax1 = display(n.loss, n.val_loss, "Loss Trend", "loss", "val_loss")
		fig2, ax2 = display(n.acc, n.val_acc, "Accuracy Trend", "acc", "val_acc")
		plt.show()
	save_model(n)
