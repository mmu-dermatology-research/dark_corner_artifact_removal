"""Module to generate image comparison metrics.

This script generates metrics for baseline, ns and telea sets and stores all
retrieved data in the specified .csv filepaths.

The standard location for saved metrics is:
    "..//Data//Metrics_Dermofit//generated_metrics//.."

Methods
-------
get_metrics 
    retrieve all required metrics
generate_metrics
    run metrics on all images and sets
main
    main method

"""

from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from skimage.metrics import structural_similarity as ssim,  peak_signal_noise_ratio as psnr
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd

def main():
    """Run the program.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    
    """
    generate_metrics()

def get_metrics(base_img, new_img):
    """Retrieve all metrics generated from comparing two images.
    
    The metrics generated are:
        * MSE
        * MAE
        * SSIM
        * PSNR
    
    Parameters
    ----------
    base_img : np.ndarray
        original image to compare
    new_img : np.ndarray
        new image to compare
    
    Returns
    -------
    list
        all retrieved statistics in order listed above

    """
    MSE = mse(base_img, new_img)
    MAE = mae(base_img, new_img)
    SSIM = ssim(base_img, new_img)
    PSNR = psnr(base_img, new_img)
    
    return [MSE, MAE, SSIM, PSNR]

def generate_metrics():
    """Run metric generation on all images within the base, ns and telea subdirectories
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    
    """
    
    ############ Baseline ###########
    # Define the filepath to load from
    image_filepath = [r"..\Data\Metrics_Dermofit\input\small\images\\",
                      r"..\Data\Metrics_Dermofit\input\medium\images\\",
                      r"..\Data\Metrics_Dermofit\input\large\images\\",
                      r"..\Data\Metrics_Dermofit\input\oth\images\\"]
    
    modified_filepath = [r"..\Data\Metrics_Dermofit\input\small\modified\\",
                         r"..\Data\Metrics_Dermofit\input\medium\modified\\",
                         r"..\Data\Metrics_Dermofit\input\large\modified\\",
                         r"..\Data\Metrics_Dermofit\input\oth\modified\\",]
    
    output_filepath = [r"..\Data\Metrics_Dermofit\generated_metrics\base\base_dermofit_metrics_small.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\base\base_dermofit_metrics_medium.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\base\base_dermofit_metrics_large.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\base\base_dermofit_metrics_oth.csv"]
    
    
    print("---------------------------------------")
    print("Generating Metrics for Baseline results")
    print("---------------------------------------")
    
    for i in range(len(image_filepath)):
        print("\nLoading", image_filepath[i][31:-9], "DCA images")
        
        # Load in the relevant .csv file
        csv = pd.read_csv(output_filepath[i])

        # If there are any values in the csv, clear them
        if csv.shape[0] != 0:
            print("values present in .csv.. clearing old .csv values")
            csv = csv[0:0]
        
        # Blank indices for row population on csv
        index = [[] for i in range(len(csv.columns))]
        
        for img in os.listdir(image_filepath[i]):
            # Load the images
            original_img = np.asarray(Image.open(os.path.join(image_filepath[i], img)))
            modified_img = np.asarray(Image.open(os.path.join(modified_filepath[i], img)))
            
            # Convert to greyscale
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            modified_img = cv2.cvtColor(modified_img, cv2.COLOR_RGB2GRAY)
            
            # Generate metrics
            metrics = get_metrics(original_img, modified_img)
            
            # append filename to index
            index[0].append(img)
            
            # append metrics to index
            for col in range(len(metrics)):
                index[col+1].append(metrics[col])            
            
        df = pd.DataFrame(index)
        df = df.transpose()

        # set the col headings to match the .csv
        col_names = csv.columns
        df.columns = col_names
        
        # append the filenames to the .csv file
        csv = csv.append(df)
        
        # export back to the original .csv file
        csv.to_csv(output_filepath[i], index = False)
        
    print("\nBaseline metrics generated\n")   
            
    ###################################
    
    
    ########### NS ############
    # Define the filepath to load from
    image_filepath = [r"..\Data\Metrics_Dermofit\output\small\reduced_originals\\",
                      r"..\Data\Metrics_Dermofit\output\medium\reduced_originals\\",
                      r"..\Data\Metrics_Dermofit\output\large\reduced_originals\\",
                      r"..\Data\Metrics_Dermofit\output\oth\reduced_originals\\"]
    
    modified_filepath = [r"..\Data\Metrics_Dermofit\output\small\ns\\",
                         r"..\Data\Metrics_Dermofit\output\medium\ns\\",
                         r"..\Data\Metrics_Dermofit\output\large\ns\\",
                         r"..\Data\Metrics_Dermofit\output\oth\ns\\",]
    
    output_filepath = [r"..\Data\Metrics_Dermofit\generated_metrics\ns\ns_dermofit_metrics_small.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\ns\ns_dermofit_metrics_medium.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\ns\ns_dermofit_metrics_large.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\ns\ns_dermofit_metrics_oth.csv"]
    
    
    print("-------------------------------------------------------")
    print("Generating Metrics for Navier-Stokes Inpainting results")
    print("-------------------------------------------------------")
    
    for i in range(len(image_filepath)):
        print("\nLoading", image_filepath[i][32:-20], "DCA images")
        
        # Load in the relevant .csv file
        csv = pd.read_csv(output_filepath[i])

        # If there are any values in the csv, clear them
        if csv.shape[0] != 0:
            print("values present in .csv.. clearing old .csv values")
            csv = csv[0:0]
        
        # Blank indices for row population on csv
        index = [[] for i in range(len(csv.columns))]
        
        for img in os.listdir(image_filepath[i]):
            # Load the images
            original_img = np.asarray(Image.open(os.path.join(image_filepath[i], img)))
            modified_img = np.asarray(Image.open(os.path.join(modified_filepath[i], img)))
            
            # Convert to greyscale
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            modified_img = cv2.cvtColor(modified_img, cv2.COLOR_RGB2GRAY)
            
            # Generate metrics
            metrics = get_metrics(original_img, modified_img)
            
            # append filename to index
            index[0].append(img)
            
            # append metrics to index
            for col in range(len(metrics)):
                index[col+1].append(metrics[col])            
            
        df = pd.DataFrame(index)
        df = df.transpose()

        # set the col headings to match the .csv
        col_names = csv.columns
        df.columns = col_names
        
        # append the filenames to the .csv file
        csv = csv.append(df)
        
        # export back to the original .csv file
        csv.to_csv(output_filepath[i], index = False)
        
    print("\nNavier-Stokes Inpainting metrics generated\n")   
    
    
    
    ###########################
    
    
    
    ########### Telea ############
    # Define the filepath to load from
    image_filepath = [r"..\Data\Metrics_Dermofit\output\small\reduced_originals\\",
                      r"..\Data\Metrics_Dermofit\output\medium\reduced_originals\\",
                      r"..\Data\Metrics_Dermofit\output\large\reduced_originals\\",
                      r"..\Data\Metrics_Dermofit\output\oth\reduced_originals\\"]
    
    modified_filepath = [r"..\Data\Metrics_Dermofit\output\small\telea\\",
                         r"..\Data\Metrics_Dermofit\output\medium\telea\\",
                         r"..\Data\Metrics_Dermofit\output\large\telea\\",
                         r"..\Data\Metrics_Dermofit\output\oth\telea\\",]
    
    output_filepath = [r"..\Data\Metrics_Dermofit\generated_metrics\telea\telea_dermofit_metrics_small.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\telea\telea_dermofit_metrics_medium.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\telea\telea_dermofit_metrics_large.csv",
                       r"..\Data\Metrics_Dermofit\generated_metrics\telea\telea_dermofit_metrics_oth.csv"]
    
    
    print("-----------------------------------------------")
    print("Generating Metrics for Telea Inpainting results")
    print("-----------------------------------------------")
    
    for i in range(len(image_filepath)):
        print("\nLoading", image_filepath[i][32:-20], "DCA images")
        
        # Load in the relevant .csv file
        csv = pd.read_csv(output_filepath[i])

        # If there are any values in the csv, clear them
        if csv.shape[0] != 0:
            print("values present in .csv.. clearing old .csv values")
            csv = csv[0:0]
        
        # Blank indices for row population on csv
        index = [[] for i in range(len(csv.columns))]
        
        for img in os.listdir(image_filepath[i]):
            # Load the images
            original_img = np.asarray(Image.open(os.path.join(image_filepath[i], img)))
            modified_img = np.asarray(Image.open(os.path.join(modified_filepath[i], img)))
            
            # Convert to greyscale
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            modified_img = cv2.cvtColor(modified_img, cv2.COLOR_RGB2GRAY)
            
            # Generate metrics
            metrics = get_metrics(original_img, modified_img)
            
            # append filename to index
            index[0].append(img)
            
            # append metrics to index
            for col in range(len(metrics)):
                index[col+1].append(metrics[col])            
            
        df = pd.DataFrame(index)
        df = df.transpose()

        # set the col headings to match the .csv
        col_names = csv.columns
        df.columns = col_names
        
        # append the filenames to the .csv file
        csv = csv.append(df)
        
        # export back to the original .csv file
        csv.to_csv(output_filepath[i], index = False)
        
    print("\nTelea Inpainting metrics generated\n")   

    ##############################

if __name__ == '__main__':
    main()