"""ISIC Data Loader

This script allows the user to load the balanced ISIC dataset for use in Machine Learning applications.

This script requires Pandas, NumPy and Scilearn-kit to be installed on the python environment in use.

This file should be imported as a module and only the 'get_data()' method should be used. 

	* get_data - returns a tuple of numpy arrays, these contain the associated data

There is the option to load images in:
	* Original [R, G, B] format
	* Grayscale
"""
import pandas as pd
import numpy as np
from PIL import Image
import os, os.path
from sklearn.utils import Bunch
import image_modifications as im # custom image modification module

def _image_alter_selection(image_data):
	"""Determines how to modify images that are of the incorrect shape. 

	The returned image will be either of shape (224,224,3) or (224,224).

	Parameters
	----------
	image_data : numpy.ndarray
		The original color values for each individual pixel of the image

	Returns
	-------
	numpy.ndarray
		an array of pixel color values in the shape correct shape

	"""
	if image_data.shape == (223,224,3) or image_data.shape == (223,224):
		return im.duplicate_bottom_row(image_data)
	elif image_data.shape == (224,223,3) or image_data.shape == (224,223):
		return im.duplicate_right_column(image_data)
	elif image_data.shape == (223,223,3) or image_data.shape == (223,223):
		return im.duplicate_col_and_row(image_data)
	else:
		return image_data

def _shuffle_mel_data(bunch: Bunch):
	"""Shuffles the data in the Bunch together. 

	The data will still be in the correct shape and have the correct associated values.

	Parameters
	----------
	bunch : numpy.Bunch
		The Bunch object to shuffle

	Returns
	-------
	none

	"""
    # create an np array the length of the total images
	image_order = np.arange(bunch.images.shape[0])
    # shuffle the integers randomly
	np.random.seed(10)
	np.random.shuffle(image_order)
    # reassign the bunch values with the same values in the shuffled order
	bunch.data = bunch.data[image_order]
	bunch.images = bunch.images[image_order]
	bunch.target = bunch.target[image_order]
	bunch.filenames = bunch.filenames[image_order]

def _load_isic_images(filepaths: list, images, targets, filenames, type: str = None):
	"""Loads dataset information from file into associated input variables.

	This information is what will be loaded into the Bunch object

	type values: 'gray', 'none'

	Parameters
	----------
	filepaths : list
		The 4 filepaths of the ISIC dataset to load images from
	images : list
		The resulting image color value arrays for each image
	targets : list
		The associated target value for each image
	filenames : list
		The original image filename
	type : list
		The type of augmentation to place on the images (default is None)

	Returns
	-------
	none
	"""
    #loop through each filepath
	for counter, path in enumerate(filepaths):
		# loop through each image in the directory
		for img in os.listdir(path):
		#open the image into temp variable, get the image metadata in np array and append to images list
			image = Image.open(os.path.join(path, img))

			# this is where the image modifications happen
			if type == 'gray':
				imgGray = image.convert('L')
				imagedata = np.asarray(imgGray)
			else:
				imagedata = np.asarray(image)

			#if any image is the wrong size, modify/duplicate rows/columns of the image
			imagedata = _image_alter_selection(imagedata)
            
			#check which target to add value
			if counter < 2:
				images[0].append(imagedata)
				if str(path[-3:]) == 'mel':
					targets[0].append(0)
				else:
					targets[0].append(1)
				filenames[0].append(str(img))
			else:
				images[1].append(imagedata)
				if str(path[-3:]) == 'mel':
					targets[1].append(0)
				else:
					targets[1].append(1)
				filenames[1].append(str(img))

def get_data(img_paths: list = [r"..\\..\\Data\\train_balanced_224x224\\train\\mel",
             r"..\\..\\Data\\train_balanced_224x224\\train\\oth",
             r"..\\..\\Data\\train_balanced_224x224\\val\\mel",
             r"..\\..\\Data\\train_balanced_224x224\\val\\oth"], type: str = None):
	'''Returns a Tuple object containing 2 numpy arrays: X[0] = training dataset, X[1] = validation dataset

	This is the only method that should be used outside of this module.

	Parameters
	----------
	img_paths : list
		A list of image path strings for loading the dataset (default is on my machine). This should be in the order of:
		img_paths[0] - train/mel
		img_paths[1] - train/oth
		img_paths[2] - test/mel
		img_paths[3] - test/oth
	type : str, optional
		The type of augmentation to place on the images (default is none)
	'''
	lesions_train = Bunch(data = [], target = [], feature_names = [], target_names = ['mel', 'oth'], frame = None, 
                      	images = [], filenames = [], DESCR = [])
	lesions_test = Bunch(data = [], target = [], feature_names = [], target_names = ['mel', 'oth'], frame = None, 
						images = [], filenames = [], DESCR = [])

	# load data into relevent datasets
	#               T, V    :    T = Train, V = Validate
	images_setup = [[],[]]
	target_setup = [[],[]]
	fname_setup  = [[],[]]
	_load_isic_images(img_paths, images_setup, target_setup, fname_setup, type = type)
        
	lesions_train['images'] = np.array(images_setup[0])
	lesions_train['target'] = np.asarray(target_setup[0])
	lesions_train['filenames'] = np.array(fname_setup[0])

	lesions_test['images'] = np.array(images_setup[1])
	lesions_test['target'] = np.asarray(target_setup[1])
	lesions_test['filenames'] = np.array(fname_setup[1])

	# flatten the images appropriately and load into data
	if type == 'gray':
		images_flat_t = lesions_train.images.reshape(7848, 50176)
		images_flat_v = lesions_test.images.reshape(1962, 50176)
	else: # for RGB tuples
		images_flat_t = lesions_train.images.reshape(7848, 50176, 3)
		images_flat_v = lesions_test.images.reshape(1962, 50176, 3)
	lesions_train['data'] = images_flat_t
	lesions_test['data'] = images_flat_v

	# create column names for all 50176 possible outcomes
	col_names = []

	for row in range(224):
		for col in range(224):
			col_names.append(str('pixel_' + str(row)+ '_'+ str(col)))
        
	lesions_train['feature_names'] = col_names
	lesions_test['feature_names'] = col_names

	# shuffle the data to randomise the datasets
	_shuffle_mel_data(lesions_train)
	_shuffle_mel_data(lesions_test)

	#return a tuple containing relevent datasets
	return lesions_train, lesions_test

def main():
	"""Main method used only to test the script.
	"""
	X = get_data(type='gray')

	lesions_train = X[0]
	lesions_test = X[1]
	print(__doc__)

if __name__ == '__main__':
	main()