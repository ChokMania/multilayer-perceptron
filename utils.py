from activations_fun import sigmoid, relu, softmax
import argparse
from math import log
import matplotlib.pyplot as plt
import numpy as np
from os import path, mkdir
import pandas as pd
import pickle
import sys

def on_press(event, loss, val_loss, fig, ax):
	if event.button != 1:
		return
	x, y = event.xdata, event.ydata
	if x is not None and y is not None:
		ax[1].set_xlim(int(x), len(val_loss))
		ax[1].set_ylim(0, max(max(loss[int(x):]), max(val_loss[int(x):])))
		fig.canvas.draw()

def display(loss, val_loss):
	fig, ax = plt.subplots(2, 1, constrained_layout=True)
	ax[0].set(title='Error Progression', ylabel="Error", xlabel="Epochs")
	ax[1].set(title='Zoomed window')
	ax[0].plot(loss, label="loss")
	ax[1].plot(range(0 ,len(loss)), loss[0:], label="loss")
	ax[0].plot(val_loss, label="val_loss")
	ax[1].plot(range(0 ,len(val_loss)), val_loss[0:], label="val_loss")
	ax[0].legend(title='Parameter where:')
	fig.canvas.mpl_connect('button_press_event', lambda event: on_press(event, loss, val_loss, fig, ax))
	plt.show()


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


def binary_cross_entropy(real, n):
	predicted = []
	actual = np.copy(real)
	actual[actual[:, 0] == "B"] = 0
	actual[actual[:, 0] == "M"] = 1
	for index in range(len(actual)):
		predicted_values = n.query(np.array(actual[index][1:], dtype=np.float64))[::-1]
		predicted.append(predicted_values)
	actual, sum_ = actual[:, 0], 0
	for index in range(len(actual)):
		sum_ += log(predicted[index][actual[index]])
	error = (-1 / len(actual)) * sum_
	return error


def load_model(file):
	try:
		with open(file, "rb") as fp:
			return pickle.load(fp)
	except:
		sys.exit(f"Error can't load file : {file}")


def save_model(n):
	i = 0
	if not path.exists("models"):
		try:
			mkdir("models")
		except OSError:
			sys.exit("Creation of the directory %s failed" % path)
	while path.exists("models/model_" + str(i) + ".p"):
		i += 1
	with open("models/model_" + str(i) + ".p", "wb") as fp:
		pickle.dump(n, fp)
		print(f"Model saved in file models/model_{str(i)}.p")
