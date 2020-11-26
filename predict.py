import argparse
import numpy as np
from utils import open_datafile, normalize, load_model, binary_cross_entropy

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("dataset_test", type=open_datafile, help="dataset to use")
	parser.add_argument("model", help="model to use")
	args = parser.parse_args()
	n = load_model(args.model)
	test = args.dataset_test.drop(args.dataset_test.columns[0], axis=1)
	predicted = []
	test = normalize(test)
	test = np.array(test)
	error = binary_cross_entropy(test, n)
	print(f"Cross Binary Entropy Error = {error:.5f}")
