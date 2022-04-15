"""Image modifications module used to modify arrays of RGB values.

This module requires the NumPy package to be installed

Methods
-------
duplicate_bottom_row
    adds another row to the array the same as the last
duplicate_right_column
    adds another column to the array the same as the last
duplicate_col_and_row
    adds a column to the array followed by a row
rgb_to_gray
    returns a grayscale value based on the R, G, B inputs
has_dca
    returns True or False if image flags as having a Dark Corner Artifact
main
    main executable used only to test module
"""
import numpy as np

def duplicate_bottom_row(imagedata):
	"""Return a new array with the bottom row duplicated and appended onto bottom of array.

	Parameters
	----------
	imagedata : numpy.ndarray
		The array of R,G,B values to modify

	Returns
	-------
	numpy.ndarray
		The modified array
	"""
	new_imagedata = np.concatenate([imagedata, imagedata[-1:,:]], axis=0)
	return new_imagedata

def duplicate_right_column(imagedata):
	"""Return a new array with the right column duplicated and appended onto right side of array.

	Parameters
	----------
	imagedata : numpy.ndarray
		The array of R,G,B values to modify

	Returns
	-------
	numpy.ndarray
		The modified array
	"""


	new_imagedata = np.concatenate([imagedata, imagedata[:,-1:]], axis=1)
	return new_imagedata

def duplicate_col_and_row(imagedata):
	"""Return a new array with duplicate column appended to right of array and duplicate row appended to bottom.

	Parameters
	----------
	imagedata : numpy.ndarray
		The array of R,G,B values to modify

	Returns
	-------
	numpy.ndarray
		The modified array
	"""
	new_imagedata = duplicate_right_column(imagedata)
	new_imagedata = duplicate_bottom_row(new_imagedata)
	return new_imagedata

def rgb_to_gray(pixelRGB):
	"""Returns a grayscale representation of the input image data - NOT IN USE

	Parameters
	----------
	pixelRGB : tuple
		The pixel R, G, B values

	Returns
	-------
	int
		the grayscale representation
	"""
	R = pixelRGB[0]
	G = pixelRGB[1]
	B = pixelRGB[2]
	gray = round((0.299) * R + (0.587 * G) + (0.114 * B))
	return gray

def modify_gamma(img, gamma: float):
	"""Returns a gamma modified image

	This method is inspired by the article at https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	using OpenCV to conduct Gamma Correction

	Parameters
	----------
	img : numpy.ndarray
		The image data to modify
	gamma : float
		The gamma value to modify the image with

	Returns
	-------
	numpy.array
		the gamma corrected image
	"""
	gamma_corrected = ((img/255) ** (1/gamma)) * 255
	return gamma_corrected

def has_dca(img, divisor):
	"""Returns True or False if over specified% of image has dark pixels

	If an image has over specified% of pixels that are dark, it most likely has Dark Corner Artifact (DCA) on the image

	Parameters
	----------
	img : numpy.ndarray
		The image data to check
	percent_divisor
		The value in which to threshold the black amount

	Returns
	-------
	boolean
		True or false dependant on if image has DCA or not

	"""
	total_px = img.shape[0] * img.shape[1]
	mask = np.ones(img.shape)
	mask[img <= 1] = 0
	mask[img > 1] = 255

	black_pixels = np.sum(mask == 0)
	if black_pixels >= total_px/divisor:
		return True
	else:
		return False

def main():
	"""Main method used only for testing the module
	"""
	x = np.array(([1,2,3], [4,5,6], [7,8,9]))
	print(x)
	print()
	x = duplicate_bottom_row(x)
	print(x)
	x = duplicate_right_column(x)
	print()
	print(x)
	x = duplicate_col_and_row(x)
	print()
	print(x)
	print()
	print(rgb_to_gray([240,3,90]))
	print(__doc__)

if __name__ == '__main__':
	main()