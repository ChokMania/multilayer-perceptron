import argparse
from neuralNetwork import neuralNetwork
import numpy as np
from utils import open_datafile, normalize
import pickle

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("data", type=open_datafile)
	args = parser.parse_args()
	try:
		n = pickle.load(open("model.p", "rb" ))
	except:
		sys.exit("Error can't load model.p")
	df = args.data.drop(args.data.columns[0], axis=1)
	test = df.iloc[455:, :]  # 20 %
	scorecard = []

	test = normalize(test)
	test = np.array(test)

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
