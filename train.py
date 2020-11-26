import argparse
from neuralNetwork import neuralNetwork
import numpy as np
from utils import open_datafile, normalize
import pickle



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("data", type=open_datafile)
	args = parser.parse_args()

	input_nodes = 30
	# nb_hidden_layers = 4
	hidden_nodes = 8
	output_nodes = 2
	learning_rate = 0.1

	n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
	epochs = 50

	df = args.data.drop(args.data.columns[0], axis=1)
	train = df.iloc[:454, :]  # 80 %

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
			# print(f"epoch {e:>3}/{epochs:<3} - loss: {} - val_loss: {}")
		pass
	with open("model.p", "wb") as fp:
		pickle.dump(n, fp)
