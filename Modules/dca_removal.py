"""This module removes DCA from images using both Navier Stokes method and Telea method.

Methods are duplicated for dermofit removal due to filepathing (they are called from a script
                                                                instead of a notebook)

Methods
-------
__calculate_reduction_rate
    calculate the amount of image to remove from edges
reduce_intensity
    remove edges from images if possible
run_super_resolution
    execute super resolution
inpaint_dca
    run specified inpaining method on image
remove_DCA
    main callable function to trigger DCA removal from image
remove_DCA_dermofit
    modified main callable function to remove DCA from dermofit image
run_super_resolution_dermofit
    execute super resolution of dermofit image
reduce_intensity_dermofit
    remove edges from dermofit image


"""
import pandas as pd 
import numpy as np
import cv2
from PIL import Image
import os, os.path
import matplotlib.pyplot as plt

def __calculate_reduction_rate(center, radius):
    """Calculate the amount of pixels to remove from both the horizontal and
    vertical borders of an image.

    Parameters
    ----------
    center : tuple
        x and y axis coordinates of the circle center
    radius : int
        radius of the circle

    Returns
    -------
    int
        number of pixels to remove
    int
        top edge limit
    int
        bottom edge limit
    int
        left edge limit
    int
        right edge limit
    
    """
    # Calculate the margin widths
    l_edge = int(center[0] - radius) if center[0] - radius >= 0 else 0
    r_edge = int(224 - (center[0] + radius)) if center[0] + radius < 224 else 0
    t_edge = int(center[1] - radius) if center[1] - radius >= 0 else 0
    b_edge = int(224 - (center[1] + radius)) if center[1] + radius < 224 else 0

    # calculate the vert and horizontal totals
    vertical = t_edge + b_edge
    horizontal = l_edge + r_edge
    
    # take the smallest value of the 2, this is the maximum we can remove to keep the image square
    r = min([vertical, horizontal])

    return r, t_edge, b_edge, l_edge, r_edge

def reduce_intensity(image, mask):
    """Reduce the intensity of the DCA by removing as much of the surrounding border 
    as possible. This method calculates the total horizontal and vertical distances
    and uses the minima to retain a square image.

    Parameters
    ----------
    image : np.ndarray
        image to crop
    mask : np.ndarray
        corresponding mask to crop

    Returns
    -------
    np.ndarray
        cropped image
    np.ndarray
        cropped mask

    """
    # Convert the image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold the image, same as done for masking process
    image_thresh = 100
    ret, thresh = cv2.threshold(gray, image_thresh, 255, cv2.THRESH_BINARY)

    # Retrieve all of the contours
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # Find the largest contour
    contours_b = []
    big_contour = []
    max = 0

    for i in contours:
        area = cv2.contourArea(i)
        if max < area:
            max = area
            big_contour = i
            contours_b.append(i)

    # Find the minimum enclosing circle center coordinates and radius
    (x, y), radius = cv2.minEnclosingCircle(big_contour)
    center = (int(x), int(y))
    radius = int(radius) - 2

    r, t_edge, b_edge, l_edge, r_edge = __calculate_reduction_rate(center, radius)

    if r != 0:
        # Only go through cropping process if there is border to be removed

        # Calculate how much to actually remove from the image
        # How many pixels left to remove?
        vertical_r = r
        horizontal_r = r 
    
        new_top = t_edge if t_edge <= vertical_r else vertical_r
        vertical_r -= new_top
        new_bottom = 224 - vertical_r
        
        new_left = l_edge if l_edge <= horizontal_r else horizontal_r
        horizontal_r -= new_left
        new_right = 224 - horizontal_r
        
        cropped_mask = np.copy(mask[new_top:new_bottom, new_left:new_right])
        cropped_image = np.copy(image[new_top:new_bottom, new_left:new_right])
    else:
        cropped_mask = np.copy(mask)
        cropped_image = np.copy(image)

    return cropped_image, cropped_mask

def run_super_resolution(image, mask):
    """Enhance the resolution of the image to combat the reduction in quality.

    Parameters
    ----------
    image : np.ndarray
        the image to modify
    mask : np.ndarray
        the corresponding mask for the image

    Returns
    -------
    np.ndarray
        the modified image
    np.ndarray
        the modified mask

    """
    super_res = cv2.dnn_superres.DnnSuperResImpl_create()
    #try:
    #    path = r'../Models/EDSR_x4.pb'
    #    super_res.readModel(path)
    #except:
    path = r'../../Models/EDSR_x4.pb'
    super_res.readModel(path)
    #super_res.readModel(path)
    super_res.setModel("edsr",4)
    upsampled = super_res.upsample(image)
    upsampled = cv2.resize(upsampled,dsize=(224,224))

    upsampled_mask = super_res.upsample(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
    upsampled_mask = cv2.resize(upsampled_mask,dsize=(224,224))

    return upsampled, cv2.cvtColor(upsampled_mask, cv2.COLOR_RGB2GRAY)

def inpaint_dca(image, mask, i_type = 'ns'):
    """Inpaint the DCA region of the image

    Parameters
    ----------
    image : np.ndarray
        the image to inpaint
    mask : np.ndarray
        the corresponding mask for the image

    Returns
    -------
    np.ndarray
        the image with the DCA region inpainted

    """
    # Set inpaint type
    if i_type == 'ns':
        flags = cv2.INPAINT_NS
    else:
        flags = cv2.INPAINT_TELEA
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius = 10, flags = flags)

    #Image.fromarray(inpainted_image).save(r'test.png')

    return inpainted_image

def remove_DCA(image, mask, removal_method = 'inpaint_ns'):
    """Remove DCA from a specified image. 
    
    Removal methods as follows:
        inpaint_ns = Navier Stokes method
        inpaint_telea = Telea method
    
    Duplicated as different return requirements metric generation

    Parameters
    ----------
    image
        image to remove DCA from
    mask
        mask of DCA to remove
    removal_method
        inpainting method to use The default is 'inpaint_ns'.

    Returns
    -------
    inpainted_image
        final inpainted image

    """
    cropped_image, cropped_mask = reduce_intensity(image, mask)

    if cropped_image.shape != (224,224,3):
        # Only run this method if some of the image has been cropped
        upsampled_image, upsampled_mask = run_super_resolution(cropped_image, cropped_mask)
    else:
        # Otherwise pass the images through
        upsampled_image = cropped_image
        upsampled_mask = cropped_mask

    if removal_method == 'inpaint_ns':
        inpainted_image = inpaint_dca(upsampled_image, upsampled_mask, 'ns')
    elif removal_method == 'inpaint_telea':
        inpainted_image = inpaint_dca(upsampled_image, upsampled_mask, 'telea')
    else:
        pass

    return inpainted_image

def remove_DCA_dermofit(image, mask, original_image, removal_method = 'inpaint_ns'):
    """Remove DCA from a specified image. 
    
    Removal methods as follows:
        inpaint_ns = Navier Stokes method
        inpaint_telea = Telea method
    
    Duplicated as different return requirements metric generation

    Parameters
    ----------
    image
        image to remove DCA from
    mask
        mask of DCA to remove
    original_image
        extra arg for cropping prior to metric generation steps
    removal_method
        inpainting method to use The default is 'inpaint_ns'.

    Returns
    -------
    inpainted_image
        final inpainted image
    upsampled_mask
        final mask used to remove DCA
    upsampled_original
        final result from upsampling

    """
    cropped_image, cropped_mask, cropped_original = reduce_intensity_dermofit(image, mask, original_image)
    if cropped_image.shape != (224,224,3):
        # Only run this method if some of the image has been cropped
        upsampled_image, upsampled_mask, upsampled_original = run_super_resolution_dermofit(cropped_image, cropped_mask, cropped_original)
    else:
        # Otherwise pass the images through
        upsampled_image = cropped_image
        upsampled_mask = cropped_mask
        upsampled_original = cropped_original

    if removal_method == 'inpaint_ns':
        inpainted_image = inpaint_dca(upsampled_image, upsampled_mask, 'ns')
    elif removal_method == 'inpaint_telea':
        inpainted_image = inpaint_dca(upsampled_image, upsampled_mask, 'telea')
    else:
        pass

    return inpainted_image, upsampled_mask, upsampled_original

def run_super_resolution_dermofit(image, mask, original_image):
    """Enhance the resolution of the image to combat the reduction in quality.

    Duplicated as different return requirements metric generation

    Parameters
    ----------
    image : np.ndarray
        the image to modify
    mask : np.ndarray
        the corresponding mask for the image

    Returns
    -------
    np.ndarray
        the modified image
    np.ndarray
        the modified mask

    """
    super_res = cv2.dnn_superres.DnnSuperResImpl_create()
    #try:
    #    path = r'../Models/EDSR_x4.pb'
    #    super_res.readModel(path)
    #except:
    path = r'../Models/EDSR_x4.pb'
    super_res.readModel(path)
    #super_res.readModel(path)
    super_res.setModel("edsr",4)
    upsampled = super_res.upsample(image)
    upsampled = cv2.resize(upsampled,dsize=(224,224))

    upsampled_mask = super_res.upsample(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
    upsampled_mask = cv2.resize(upsampled_mask,dsize=(224,224)) 
    upsampled_original = super_res.upsample(original_image)
    upsampled_original = cv2.resize(upsampled_original,dsize=(224,224))

    return upsampled, cv2.cvtColor(upsampled_mask, cv2.COLOR_RGB2GRAY), upsampled_original

def reduce_intensity_dermofit(image, mask, original_image):
    """Reduce the intensity of the DCA by removing as much of the surrounding border 
    as possible. This method calculates the total horizontal and vertical distances
    and uses the minima to retain a square image.
    
    Duplicated as different return requirements metric generation

    Parameters
    ----------
    image : np.ndarray
        image to crop
    mask : np.ndarray
        corresponding mask to crop

    Returns
    -------
    np.ndarray
        cropped image
    np.ndarray
        cropped mask

    """
    # Convert the image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold the image, same as done for masking process
    image_thresh = 100
    ret, thresh = cv2.threshold(gray, image_thresh, 255, cv2.THRESH_BINARY)

    # Retrieve all of the contours
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # Find the largest contour
    contours_b = []
    big_contour = []
    max = 0

    for i in contours:
        area = cv2.contourArea(i)
        if max < area:
            max = area
            big_contour = i
            contours_b.append(i)

    # Find the minimum enclosing circle center coordinates and radius
    (x, y), radius = cv2.minEnclosingCircle(big_contour)
    center = (int(x), int(y))
    radius = int(radius) - 2

    r, t_edge, b_edge, l_edge, r_edge = __calculate_reduction_rate(center, radius)

    if r != 0:
        # Only go through cropping process if there is border to be removed

        # Calculate how much to actually remove from the image
        # How many pixels left to remove?
        vertical_r = r
        horizontal_r = r 
    
        new_top = t_edge if t_edge <= vertical_r else vertical_r
        vertical_r -= new_top
        new_bottom = 224 - vertical_r
        
        new_left = l_edge if l_edge <= horizontal_r else horizontal_r
        horizontal_r -= new_left
        new_right = 224 - horizontal_r
        cropped_mask = np.copy(mask[new_top:new_bottom, new_left:new_right])
        cropped_image = np.copy(image[new_top:new_bottom, new_left:new_right])
        cropped_original_image = np.copy(original_image[new_top:new_bottom, new_left:new_right])
    else:
        cropped_mask = np.copy(mask)
        cropped_image = np.copy(image)
        cropped_original_image = np.copy(original_image)

    return cropped_image, cropped_mask, cropped_original_image

def main():
    """Main method used for module testing
    
    """
    # Load the training melanoma DCA masks
    filepath = r"../Data/DCA_Masks/train/mel/"
    train_mel_masks = []
    for img in os.listdir(filepath):
        image = Image.open(os.path.join(filepath, img))
        train_mel_masks.append(np.asarray(image))
        
    # Load in the training melanoma masks
    t_mel_csv = pd.read_csv(r"../Data/Annotations/train_mel.csv")
    
    # Load in the training melanoma intensity annotations
    dca_t_mel_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_train_mel.csv")
    
    # Segregate each mask type and retain the old index
    small_dca_masks = dca_t_mel_csv.loc[dca_t_mel_csv['Small_DCA'] == 1].reset_index(drop = False)
    medium_dca_masks = dca_t_mel_csv.loc[dca_t_mel_csv['Medium_DCA'] == 1].reset_index(drop = False)
    large_dca_masks = dca_t_mel_csv.loc[dca_t_mel_csv['Large_DCA'] == 1].reset_index(drop = False)
    oth_dca_masks = dca_t_mel_csv.loc[dca_t_mel_csv['Oth'] == 1].reset_index(drop = False)
    
    # Append the original image name to the dataframe
    small_dca_masks['Original_Image_Name'] = [small_dca_masks['Image_Name'][i][:-9] + '.jpg' for i in range(len(small_dca_masks.index))]
    medium_dca_masks['Original_Image_Name'] = [medium_dca_masks['Image_Name'][i][:-9] + '.jpg' for i in range(len(medium_dca_masks.index))]
    large_dca_masks['Original_Image_Name'] = [large_dca_masks['Image_Name'][i][:-9] + '.jpg' for i in range(len(large_dca_masks.index))]
    oth_dca_masks['Original_Image_Name'] = [oth_dca_masks['Image_Name'][i][:-9] + '.jpg' for i in range(len(oth_dca_masks.index))]

    IMAGE_NUMBER = 0
    DCA_SET = large_dca_masks
    
    test_image_name = DCA_SET['Original_Image_Name'][IMAGE_NUMBER]
    test_image = np.asarray(Image.open(os.path.join(r'../Data/train_balanced_224x224/train/mel/', test_image_name)))
    test_image_mask_name = DCA_SET['Image_Name'][IMAGE_NUMBER]
    test_image_mask = np.asarray(Image.open(os.path.join(r'../Data/DCA_Masks/train/mel/', test_image_mask_name)))


    #inpainted_image = remove_DCA(test_image, test_image_mask)

    #Image.fromarray(inpainted_image).save(r'test_1.png')



if __name__ == '__main__':
    main()


