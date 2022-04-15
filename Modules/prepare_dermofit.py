"""This module is used to load in images from the Dermofit dataset, reshape
them to the required size, save and mask the image ready for generating 
metrics on removal processes.

The dermofit dataset contains images from multiple classes which is perfect 
for including both melanoma and non melanoma images.

Methods
-------
main
    main executable method
load_dermofit
    load the dermofit dataset
save_image_set
    save a list of images to a specified save path
__reshape_image
    reshape an image into the required size
__load_annotations
    load the image annotation files
__load_masks
    load the masks to use for masking the images
__distribute_dataset
    distribute Dermofit into 4 subsets
__mask_dataset
    apply loaded masks to Dermofit images
__remove_dermofit_test_DCAs
    apply DCA removal methods
    
"""
import os
import numpy as np
from PIL import Image
import pandas as pd
import random
from dca_removal import remove_DCA_dermofit

def main():
    """Main method to load and reshape all Dermofit images, mask them and save all required images for the
    metric generation.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    
    """
    ## Load and save all of the Dermofit dataset images into a gt set - this also reshapes
    # the images into the required size (224x224)
    X = load_dermofit(r"..\\Data\\Dermofit\\")
    save_image_set(X, r"..\\Data\\Metrics_Dermofit\\input\\gt\\")
    
    ## Shuffle X for good class distribution
    random.shuffle(X)
    
    ## Load and save the required amount of masks from the DCA masks folders.
    # This resaves the masks so the ones that have been used can be easily located..
    masks = load_masks(save = True)
    
    ## Distribute the dataset into small, med, large, other subsets (for masking)
    __distribute_dataset(X, len(masks[0]), save = True)
    
    ## Apply the masks to the dataset
    __mask_dataset()
    
    ## Execute DCA removal methods on the masked Dermofit images
    remove_dermofit_test_DCAs()

def load_dermofit(path:str, required_shape:tuple = (224,224)):
    """Load all of the images from the dermofit dataset reshaped to the 
    required size.
    
    Parameters
    ----------
    path : str
        the root filepath of the dermofit dataset
    required_shape : tuple
        the required shape of images ((224,224) by default)
    
    Returns
    -------
    list:
        all dermofit images
    
    """
    main_dir = path
    num_images = 0
    images = []
    
    # Navigate Classes
    for class_dir in os.listdir(main_dir):
        # We don't want to do anything with this folder if it's there.
        if class_dir == '__MACOSX':
            continue
        else:
            # Create the new path
            class_dir_path = os.path.join(main_dir, class_dir)
            
            # Navigate Image Folders
            for im_dir in os.listdir(class_dir_path):
                
                # Increment the number of images
                num_images += 1
                
                # Create the new path
                img_path = os.path.join(class_dir_path, im_dir)
                
                # Navigate Images
                for image in os.listdir(img_path):
                    # Ignore the masks in the directories, these are not the 
                    # masks intended for use with this experiment
                    if 'mask' not in image:
                        this_image = Image.open(os.path.join(img_path, image))
                        
                        # Reshape the image and append to all images
                        this_image = __reshape_image(this_image, required_shape)
                        images.append(this_image)
    return images

def save_image_set(images: list, savepath: str, clear_dir: bool = True):
    """Save all of the images from an image set into a specified filepath.
    
    Parameters
    ----------
    images : list
        list of images to save
    savepath : str
        path to save images to
    clear_dir : bool
        flag to determine if directory gets cleared before saving
    
    Returns
    -------
    None
    """    
    # Clear all of the files in the filepath ready for new file creation
    if clear_dir:
        for file in os.listdir(savepath):
            os.remove(os.path.join(savepath, file))
        
    for i, image in enumerate(images):
        if i < 9:
            path = savepath + "000" + str(i+1) + ".png"
        elif i < 99:
            path = savepath + "00" + str(i+1) + ".png"
        elif i < 999:
            path = savepath + "0" + str(i+1) + ".png"
        else:
            path = savepath + str(i+1) + ".png"
        image.save(path)
    
def __reshape_image(image, shape:tuple):
    """Reshape a Pillow Image to a specified shape
    
    Parameters
    ----------
    image : PIL.Image.Image
        the image to reshape
    shape : tuple
        required size
    
    Returns
    -------
    PIL.Image.Image
        reshaped image
    
    """
    reshaped_image = image.resize(shape)
    return reshaped_image

def __load_annotations():
    """
    
    """
    dca_t_mel_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_train_mel.csv")
    dca_t_mel_csv['Subset'] = ["Train_Mel" for i in range(len(dca_t_mel_csv.index))]
    dca_t_oth_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_train_oth.csv")
    dca_t_oth_csv['Subset'] = ["Train_Oth" for i in range(len(dca_t_oth_csv.index))]
    dca_v_mel_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_val_mel.csv")
    dca_v_mel_csv['Subset'] = ["Val_Mel" for i in range(len(dca_v_mel_csv.index))]
    dca_v_oth_csv = pd.read_csv(r"../Data/Annotations/dca_intensities_val_oth.csv")
    dca_v_oth_csv['Subset'] = ["Val_Oth" for i in range(len(dca_v_oth_csv.index))]
    
    tmel_small_dca_masks = dca_t_mel_csv.loc[dca_t_mel_csv['Small_DCA'] == 1].reset_index(drop = False)
    tmel_medium_dca_masks = dca_t_mel_csv.loc[dca_t_mel_csv['Medium_DCA'] == 1].reset_index(drop = False)
    tmel_large_dca_masks = dca_t_mel_csv.loc[dca_t_mel_csv['Large_DCA'] == 1].reset_index(drop = False)
    tmel_oth_dca_masks = dca_t_mel_csv.loc[dca_t_mel_csv['Oth'] == 1].reset_index(drop = False)
    
    toth_small_dca_masks = dca_t_oth_csv.loc[dca_t_oth_csv['Small_DCA'] == 1].reset_index(drop = False)
    toth_medium_dca_masks = dca_t_oth_csv.loc[dca_t_oth_csv['Medium_DCA'] == 1].reset_index(drop = False)
    toth_large_dca_masks = dca_t_oth_csv.loc[dca_t_oth_csv['Large_DCA'] == 1].reset_index(drop = False)
    toth_oth_dca_masks = dca_t_oth_csv.loc[dca_t_oth_csv['Oth'] == 1].reset_index(drop = False)


    vmel_small_dca_masks = dca_v_mel_csv.loc[dca_v_mel_csv['Small_DCA'] == 1].reset_index(drop = False)
    vmel_medium_dca_masks = dca_v_mel_csv.loc[dca_v_mel_csv['Medium_DCA'] == 1].reset_index(drop = False)
    vmel_large_dca_masks = dca_v_mel_csv.loc[dca_v_mel_csv['Large_DCA'] == 1].reset_index(drop = False)
    vmel_oth_dca_masks = dca_v_mel_csv.loc[dca_v_mel_csv['Oth'] == 1].reset_index(drop = False)


    voth_small_dca_masks = dca_v_oth_csv.loc[dca_v_oth_csv['Small_DCA'] == 1].reset_index(drop = False)
    voth_medium_dca_masks = dca_v_oth_csv.loc[dca_v_oth_csv['Medium_DCA'] == 1].reset_index(drop = False)
    voth_large_dca_masks = dca_v_oth_csv.loc[dca_v_oth_csv['Large_DCA'] == 1].reset_index(drop = False)
    voth_oth_dca_masks = dca_v_oth_csv.loc[dca_v_oth_csv['Oth'] == 1].reset_index(drop = False)
    
    # Stack the dataframes
    small_dca_masks = tmel_small_dca_masks.append(toth_small_dca_masks).append(vmel_small_dca_masks).append(voth_small_dca_masks).reset_index(drop=True)
    medium_dca_masks = tmel_medium_dca_masks.append(toth_medium_dca_masks).append(vmel_medium_dca_masks).append(voth_medium_dca_masks).reset_index(drop=True)
    large_dca_masks = tmel_large_dca_masks.append(toth_large_dca_masks).append(vmel_large_dca_masks).append(voth_large_dca_masks).reset_index(drop=True)
    oth_dca_masks = tmel_oth_dca_masks.append(toth_oth_dca_masks).append(vmel_oth_dca_masks).append(voth_oth_dca_masks).reset_index(drop=True)
    
    return small_dca_masks, medium_dca_masks, large_dca_masks, oth_dca_masks

def load_masks(save: bool = True):
    """Load masks from ../Data/DCA_Masks/
    
    Parameters
    ----------
    save : bool
        flag to save images or not
    
    """
    # Lists to store masks
    small_masks = []
    medium_masks = []
    large_masks = []
    oth_masks = []
    
    # Load the training melanoma DCA masks
    tm_filepath = r"../Data/DCA_Masks/train/mel/"
    train_mel_masks = []
    for img in os.listdir(tm_filepath):
        image = Image.open(os.path.join(tm_filepath, img))
        train_mel_masks.append(image)
    
    # Load the training melanoma DCA masks
    to_filepath = r"../Data/DCA_Masks/train/oth/"
    train_oth_masks = []
    for img in os.listdir(to_filepath):
        image = Image.open(os.path.join(to_filepath, img))
        train_oth_masks.append(image)
    
    # Load the training melanoma DCA masks
    vm_filepath = r"../Data/DCA_Masks/val/mel/"
    val_mel_masks = []
    for img in os.listdir(vm_filepath):
        image = Image.open(os.path.join(vm_filepath, img))
        val_mel_masks.append(image)
    
    # Load the training melanoma DCA masks
    vo_filepath = r"../Data/DCA_Masks/val/oth/"
    val_oth_masks = []
    for img in os.listdir(vo_filepath):
        image = Image.open(os.path.join(vo_filepath, img))
        val_oth_masks.append(image)
    
    # Load in the intensity annotations
    small_annotations, medium_annotations, large_annotations, oth_annotations = __load_annotations()
    
    # Smallest number of dcas in the category determines how many of the images 
    # are masked
    num_images_per_category = min([len(small_annotations.index), len(medium_annotations.index), len(large_annotations.index), len(oth_annotations.index)])
    
    for i in range(num_images_per_category): 
        small_mask_subset = small_annotations['Subset'][i]
        #small_mask_name = small_annotations['Image_Name'][i]
        small_mask_index = small_annotations['index'][i]
        
        if small_mask_subset == 'Train_Mel':
            temp_subset = train_mel_masks
        elif small_mask_subset == 'Train_Oth':
            temp_subset = train_oth_masks
        elif small_mask_subset == 'Val_Mel':
            temp_subset = val_mel_masks
        elif small_mask_subset == 'Val_Oth':
            temp_subset = val_oth_masks
            
        small_mask = temp_subset[small_mask_index]
        small_masks.append(small_mask)
        
        medium_mask_subset = medium_annotations['Subset'][i]
        #medium_mask_name = medium_annotations['Image_Name'][i]
        medium_mask_index = medium_annotations['index'][i]
        
        if medium_mask_subset == 'Train_Mel':
            temp_subset = train_mel_masks
        elif medium_mask_subset == 'Train_Oth':
            temp_subset = train_oth_masks
        elif medium_mask_subset == 'Val_Mel':
            temp_subset = val_mel_masks
        elif medium_mask_subset == 'Val_Oth':
            temp_subset = val_oth_masks
            
        medium_mask = temp_subset[medium_mask_index]
        medium_masks.append(medium_mask)
        
        large_mask_subset = large_annotations['Subset'][i]
        #medium_mask_name = medium_annotations['Image_Name'][i]
        large_mask_index = large_annotations['index'][i]
        
        if large_mask_subset == 'Train_Mel':
            temp_subset = train_mel_masks
        elif large_mask_subset == 'Train_Oth':
            temp_subset = train_oth_masks
        elif large_mask_subset == 'Val_Mel':
            temp_subset = val_mel_masks
        elif large_mask_subset == 'Val_Oth':
            temp_subset = val_oth_masks
            
        large_mask = temp_subset[large_mask_index]
        large_masks.append(large_mask)
        
        oth_mask_subset = oth_annotations['Subset'][i]
        oth_mask_index = oth_annotations['index'][i]
        
        if oth_mask_subset == 'Train_Mel':
            temp_subset = train_mel_masks
        elif oth_mask_subset == 'Train_Oth':
            temp_subset = train_oth_masks
        elif oth_mask_subset == 'Val_Mel':
            temp_subset = val_mel_masks
        elif oth_mask_subset == 'Val_Oth':
            temp_subset = val_oth_masks
        
        oth_mask = temp_subset[oth_mask_index]
        oth_masks.append(oth_mask)
    
    masks = [small_masks, medium_masks, large_masks, oth_masks]
    
    if save:
        savepaths = [r"..\Data\Metrics_Dermofit\input\small\masks\\",
                     r"..\Data\Metrics_Dermofit\input\medium\masks\\",
                     r"..\Data\Metrics_Dermofit\input\large\masks\\",
                     r"..\Data\Metrics_Dermofit\input\oth\masks\\"]
        for i, mask_set in enumerate(masks):
            save_image_set(mask_set,savepaths[i])
    
    return masks

def __distribute_dataset(images : list, num_images : int, save : bool = True):
    """Distribute dataset into small/med/large/oth image sets ready for applying
    dca masks.
    
    Parameters
    ----------
    images : list
        images to distribute
    num_images : int
        number of images to load into each set
    save : bool
        flag to save images or not
    
    Returns
    -------
    None
    
    """
    # Counter to keep track of how far into the dataset we are for each subset
    counter = 0
    
    # sets = small, med, large, oth
    sets = [[],[],[],[]]
    
    for i in range(4):
        for j in range(num_images):
            sets[i].append(images[counter])
            counter += 1
            
    savepaths = [r'..\Data\Metrics_Dermofit\input\small\images\\',
                 r'..\Data\Metrics_Dermofit\input\medium\images\\',
                 r'..\Data\Metrics_Dermofit\input\large\images\\',
                 r'..\Data\Metrics_Dermofit\input\oth\images\\']
    
    if save:        
        for i, im_set in enumerate(sets):
            save_image_set(im_set, savepaths[i])

def __mask_dataset():
    """Apply masks from Metrics_Dermofit/x/masks/ to
    Metrics_Dermofit/x/images and save in Metrics_Dermofit/x/modified.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    
    """
    img_paths = [r'..\Data\Metrics_Dermofit\input\small\images\\',
                 r'..\Data\Metrics_Dermofit\input\medium\images\\',
                 r'..\Data\Metrics_Dermofit\input\large\images\\',
                 r'..\Data\Metrics_Dermofit\input\oth\images\\']
    
    mask_paths = [r'..\Data\Metrics_Dermofit\input\small\masks\\',
                  r'..\Data\Metrics_Dermofit\input\medium\masks\\',
                  r'..\Data\Metrics_Dermofit\input\large\masks\\',
                  r'..\Data\Metrics_Dermofit\input\oth\masks\\']
    
    save_paths = [r'..\Data\Metrics_Dermofit\input\small\modified\\',
                  r'..\Data\Metrics_Dermofit\input\medium\modified\\',
                  r'..\Data\Metrics_Dermofit\input\large\modified\\',
                  r'..\Data\Metrics_Dermofit\input\oth\modified\\']
    
    for path in save_paths:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        
    
    for i in range(4):
        # load in the images for relevant set
        images = []
        for j, img in enumerate(os.listdir(img_paths[i])):
            #image_name = str(j+1) + '.png'
            images.append(np.asarray(Image.open(os.path.join(img_paths[i], img))))
            
        # load in the masks for relevant set
        masks = []
        for j, mask in enumerate(os.listdir(mask_paths[i])):
            #mask_name = str(j+1) + '.png'
            masks.append(np.asarray(Image.open(os.path.join(mask_paths[i], mask))))
         
        # save output masked images
        for j, img in enumerate(images):
            #print(j)
            output = img.copy()
            output[masks[j].astype(bool)] = 0
            
            output = Image.fromarray(output)
            if j < 9:
                savepath = save_paths[i] + "000" + str(j+1) + '.png'
            elif j < 99:
                savepath = save_paths[i] + "00" + str(j+1) + ".png"
            elif j < 999:
                savepath = save_paths[i] + "0" + str(j+1) + ".png"
            else:
                savepath = save_paths[i] + str(j+1) + ".png"
            
    
            output.save(savepath)
            
def remove_dermofit_test_DCAs():
    """

    """
    dca_paths = [r'..\Data\Metrics_Dermofit\input\small\modified\\',
                  r'..\Data\Metrics_Dermofit\input\medium\modified\\',
                  r'..\Data\Metrics_Dermofit\input\large\modified\\',
                  r'..\Data\Metrics_Dermofit\input\oth\modified\\']
    
    img_paths = [r'..\Data\Metrics_Dermofit\input\small\images\\',
                 r'..\Data\Metrics_Dermofit\input\medium\images\\',
                 r'..\Data\Metrics_Dermofit\input\large\images\\',
                 r'..\Data\Metrics_Dermofit\input\oth\images\\']
    
    mask_paths = [r'..\Data\Metrics_Dermofit\input\small\masks\\',
                  r'..\Data\Metrics_Dermofit\input\medium\masks\\',
                  r'..\Data\Metrics_Dermofit\input\large\masks\\',
                  r'..\Data\Metrics_Dermofit\input\oth\masks\\']
    
    save_paths_ns = [r'..\Data\Metrics_Dermofit\output\small\ns\\',
                  r'..\Data\Metrics_Dermofit\output\medium\ns\\',
                  r'..\Data\Metrics_Dermofit\output\large\ns\\',
                  r'..\Data\Metrics_Dermofit\output\oth\ns\\']
    
    save_paths_telea = [r'..\Data\Metrics_Dermofit\output\small\telea\\',
                  r'..\Data\Metrics_Dermofit\output\medium\telea\\',
                  r'..\Data\Metrics_Dermofit\output\large\telea\\',
                  r'..\Data\Metrics_Dermofit\output\oth\telea\\']
    
    save_paths_reduced_imgs = [r'..\Data\Metrics_Dermofit\output\small\reduced_originals\\',
                               r'..\Data\Metrics_Dermofit\output\medium\reduced_originals\\',
                               r'..\Data\Metrics_Dermofit\output\large\reduced_originals\\',
                               r'..\Data\Metrics_Dermofit\output\oth\reduced_originals\\']
    
    save_paths_reduced_masks = [r'..\Data\Metrics_Dermofit\output\small\reduced_masks\\',
                                r'..\Data\Metrics_Dermofit\output\medium\reduced_masks\\',
                                r'..\Data\Metrics_Dermofit\output\large\reduced_masks\\',
                                r'..\Data\Metrics_Dermofit\output\oth\reduced_masks\\']
    

    # Loop through all of the images in all 4 dca_paths with DCA's transposed on
    for i, path in enumerate(dca_paths):
        for img in os.listdir(path):
            
            # get the ns inpaint
            output_ns, reduced_mask_ns, reduced_original_ns = remove_DCA_dermofit(np.asarray(Image.open(os.path.join(path, img))), 
                                                                         np.asarray(Image.open(os.path.join(mask_paths[i], img))), 
                                                                         np.asarray(Image.open(os.path.join(img_paths[i], img))),
                                                                         removal_method = 'inpaint_ns')
            # get the telea inpaint
            output_telea, reduced_mask_telea, reduced_original_telea = remove_DCA_dermofit(np.asarray(Image.open(os.path.join(path, img))), 
                                                                         np.asarray(Image.open(os.path.join(mask_paths[i], img))), 
                                                                         np.asarray(Image.open(os.path.join(img_paths[i], img))),
                                                                         removal_method = 'inpaint_telea')
            
            # save the ns inpainting to the relevant ns savepath
            ns_savepath = save_paths_ns[i] + img
            Image.fromarray(output_ns).save(ns_savepath)
            
            # save the telea inpainting to the relevant telea savepath
            telea_savepath = save_paths_telea[i] + img
            Image.fromarray(output_telea).save(telea_savepath)
            
            # save the reduced original image to the relevant savepath
            ro_savepath = save_paths_reduced_imgs[i] + img
            Image.fromarray(reduced_original_ns).save(ro_savepath)
            
            # save the reduced mask to the relevant savepath
            rm_savepath = save_paths_reduced_masks[i] + img
            Image.fromarray(reduced_mask_ns).save(rm_savepath)
    
if __name__ == '__main__':
    main()
