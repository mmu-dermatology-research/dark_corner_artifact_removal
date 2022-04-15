"""This module applies the full masking process to a given image. This module is
to be imported and used as a library function. Alternatively a mask for an individual
image can be extracted by altering the main method.

Methods
-------
main
    used for module testing only
__get_new_filename
    retrieve a new filename for the generated mask to save
__is_gray
    check if the image is in greyscale
__get_img_thresh
    extract a binary threshold of the image
__get_all_contours
    extract all contours in an image
__find_largest_contour
    retrieve the largest contour from a list of contours
__draw_new_outer_ring
    use minimum enclosing circle on a contour
get_mask
    retrieve the mask of a DCA from an image
save_mask
    save the mask of a DCA extracted from an image

"""
import pandas as pd
import numpy as np
from PIL import Image
import os, os.path
from sklearn.utils import Bunch
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import image_modifications as im # custom image modification module

def main():
    """Main executable method - used only for module testing in this project

    """
    # load in an image from dataset - any image with a DCA will do
    filepath = r"..\\Data\\train_balanced_224x224\\train\\mel"
    img = 'ISIC2017_0000140_mel.jpg'
    image = Image.open(os.path.join(filepath, img))
    imagedata = np.asarray(image)
    #save_mask(img, imagedata, r"..\\Data\\DCA_Masks\\train\\mel\\")

def __get_new_filename(filename: str):
    """Create a new filename for the masked image.

    This filename is used to save the masked image appropriately.

    New filenames will remove the .jpg suffix from the end of the original
    image, append '_MASK' to the filename and format the image as a .png file
    for more image detail when saving.

    Parameters
    ----------
    filename : str
        original image filename

    Returns
    -------
    str
        modified filename

    """
    new_filename = filename[:-4] + "_MASK.png"
    return new_filename

def __is_gray(image):
    """Check if an image is currently in grayscale.

    If the image is in grayscale then nothing needs to happen
    otherwise if the image is in RGB format, convert it to grayscale

    Parameters
    ----------
    image : ndarray
        image to check

    Returns
    -------
    ndarray
        grayscale image

    """
    if len(image.shape) == 3:
        # if it is, convert it to grayscale
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        return gray_image
    elif len(image.shape) != 2:
        print("Incompatible image shape")
        return
    else:
        return image

def __get_img_thresh(gray_image, thresh: int = 100):
    """Retrieve binary threshold of an image
    
    Parameters
    ----------
    gray_image
        image to threshold
    thresh : int
        threshold value
        
    Returns
    -------
    thresh_img
        thresholded representation of image

    """
    # set a threshold value and retrieve the binary threshold for the grayscale image
    image_thresh = thresh 
    ret, thresh_img = cv.threshold(gray_image, image_thresh, 255, cv.THRESH_BINARY)

    return thresh_img

def __get_all_contours(thresh_img):
    """Extract all contours from an image
    
    Parameters
    ----------
    thresh_img
        binary threshold image
    
    Returns
    -------
    contours
        list of all contours extracted from image

    """
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours
    
def __find_largest_contour(contours):
    """Calculate the area of each contour and extract the largest
    
    Parameters
    ----------
    contours : list
        all contours
        
    Returns
    -------
    big_contour
        the contour with the largest area

    """
    contoursB = []
    big_contour = []
    this_max = 0
    for i in contours:
        area = cv.contourArea(i) #--- find the contour having biggest area ---
        if(area > this_max):
            this_max = area
            big_contour = i 
            contoursB.append(i)
    return contoursB, big_contour

def __draw_new_outer_ring(big_contour, gray_image):
    """Perform minimum enclosing circle on a given contour
    
    Parameters
    ----------
    big_contour : np.ndarray
        the contour to contain
    gray_image
        the original image (in greyscale)
        
    Returns
    -------
    this_contour
        the enclosed contour

    """
    (x,y), radius = cv.minEnclosingCircle(np.asarray(big_contour))
    center = (int(x), int(y))
    radius = int(radius) - 2
    this_contour = cv.circle(np.ones(gray_image.shape), center, radius, (0,255,0), -1)
    
    return this_contour

def get_mask(image):
    """Extract a mask of a DCA from an image.
    
    Parameters
    ----------
    image : PIL.Image
        the image to extract the mask from
        
    Returns
    -------
    mask
        the extracted mask

    """
    gray_image = __is_gray(image)
    thresh_image = __get_img_thresh(gray_image)
    contours = __get_all_contours(thresh_image)
    contoursB, largest_contour = __find_largest_contour(contours)
    mask = __draw_new_outer_ring(largest_contour, gray_image)
    mask[mask == 1] = 255 # may need inverting later
    mask = Image.fromarray(mask)
    mask = mask.convert('L')
    return mask

def save_mask(filename, image, savepath):
    """Save the extracted mask to the specified savepath
    
    Parameters
    ----------
    filename : str
        the name of the file to save
    image : PIL.Image
        the image to save
    savepath : str
        the filepath of the destination save location
    
    Returns
    -------
    None.

    """
    mask = get_mask(image)
    savepath += __get_new_filename(filename)
    mask.save(savepath)



if __name__ == '__main__':
    main()