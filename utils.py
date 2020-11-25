import numpy as np
import pandas as pd
import sys


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


def sigmoid(x):
	return(1 / (1 + np.exp(-x)))


def normalize(df):
	result = df.copy()
	for feature_name in df.columns[1:]:
		if feature_name != 0:
			max_value = df[feature_name].max()
			# min_value = df[feature_name].min()²
			result[feature_name] = (df[feature_name] / max_value * 0.99) + 0.01
			# result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
	return result
