# Preliminary Experiments

This folder contains all Jupyter Notebook experiments that were ran prior to 
beginning the official structured project work.

## Folder Contents

These experiments consist of the following:

 * Finding Images with Black Corners
 
 This notebook describes an initial algorithm attempt to find all images in the 
 Cleaned Balanced Dataset that contain DCA's.
 
 * ISIC to Bunch
  
 This notebook breaks down the process that the isic_data.py module file uses to 
 load the dataset into a Bunch object.
 
 * Otsu's threshold masking - Corners affect masking
 
 This notebook demonstrates the issues that DCA's have on standard masking processes.
 
 * Playing around with edge detection
 
 This notebook is used as an exercise to learn more about image edge detection.
 
 * Removing outer ring from mask - largest contour
 
 This notebook details an automated masking process for images with a DCA.