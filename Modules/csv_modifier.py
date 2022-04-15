"""Module to populate individual annotation .csv files.

Test methods that are unused are commented out (they also may not be in a working state)

Methods
-------
__populate_csv
	populate a specified .csv file with the filenames from a specified filepath.


"""
# ignore the FutureWarning printed from pandas
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import pandas as pd
import numpy as np
import os, os.path
from PIL import Image

def __populate_csv(target_filename, target_filepath):
	"""Populate the specified .csv file with all filenames from the specified filepath.

	The population will only happen if the .csv file is initially empty to avoid removal
	of important annotation data.

	Parameters
	----------
	target_filename : str
		the target .csv file
	target_filepath : str
		the target filepath of images

	Returns
	-------
	None

	"""
	# get the target_filename .csv file
	csv = pd.read_csv(target_filename)

	# when the .csv has no rows under the column headings
	if csv.shape[0] == 0:
		path = target_filepath
		filenames = [[],[],[],[],[],[],[]]

		# get the filename of each image
		for img in os.listdir(path):
			filenames[0].append(str(img))

    	# turn it into a dataframe
		fndf = pd.DataFrame(filenames)
		fndf = fndf.transpose()

		# set the col headings to match the .csv
		col_names = csv.columns
		fndf.columns = col_names

		# append the filenames to the .csv file
		csv = csv.append(fndf)

		# export back to the original .csv file
		csv.to_csv(target_filename, index = False)

#def get_feature_count(csv_df):
#	"""
#
#	"""
#	total_borders = csv_df.Borders[csv_df.Borders == 1].count()
#	total_hair = csv_df.Borders[csv_df.Hair == 1].count()
#	total_measurement_d = csv_df.Measurement_Device[csv_df.Measurement_Device == 1].count()
#	total_air_pockets = csv_df.Air_Pockets[csv_df.Air_Pockets == 1].count()
#	total_clin_marking = csv_df.Clinical_Markings[csv_df.Clinical_Markings == 1].count()
#	total_oth = csv_df.Other[csv_df.Other == 1].count()
#features = {"Borders": [total_borders], "Hair": [total_hair], "Measurement_Device": [total_measurement_d],
#				"Air_Pockets": [total_air_pockets], "Clinical_Markings": [total_clin_marking], 
#				"Other": [total_oth]}
#	df = pd.DataFrame(data = features)
#	return df

#def get_csv_data(csv: str = "train_mel.csv", d_type: str = "DCA"):
#	"""Get the filenames that contain the specified artifact from a specified .csv file.
#
#	Default is to get all filenames for DCA images from train_mel.csv
#
#	!!! Improve this later !!!
#
#	Parameters
#	----------
#	csv : str
#		filename to get data from
#   type : str
#		column heading to get required data
#
#	Returns
#	-------
#	ndarray
#		ndarray of all filenames that match the criteria
#
#	"""
#	filepath = r"../Data/Annotations/" + csv
#
#	try:
#		# read in the csv
#		csv = pd.read_csv(filepath)
#
#		# drop all rows that arent required
#		if d_type == "DCA":
#			# get all the dca rows
#			ind = csv[(csv['Border_Type'] != 3) & (csv['Border_Type'] != 5)].index
#		else:
#			try:
#				# get all rows dependant on specified column
#				ind = csv[csv[d_type] != 1].index
#			except:
#				# end this method, the col head isnt there
#				print(d_type + " is not a column in the .csv file")
#				return
#		
#		# remove all the indices 
#		csv.drop(ind, inplace = True)
#
#		# drop all columns except the image name
#		csv.drop(csv.columns.difference(["Image_Name"]), 1, inplace = True)
#
#		# transpose, convert and reduce dimensionality
#		csv = csv.transpose()
#   	csv = csv.to_numpy()
#		csv = csv[0]
#
#		return csv
#	except:
#		# display error message
#		print(csv + " not in ../../Data/Annotations")

def main():
	#__populate_csv("train_mel.csv", r"..\train_balanced_224x224\train\mel")
	#__populate_csv("train_oth.csv", r"..\train_balanced_224x224\train\oth")
	#__populate_csv("val_mel.csv", r"..\train_balanced_224x224\val\mel")
	#__populate_csv("val_oth.csv", r"..\train_balanced_224x224\val\oth")
	#get_csv_data()
	pass

if __name__ == "__main__":
	main()
