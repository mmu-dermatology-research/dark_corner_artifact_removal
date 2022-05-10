# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:01:16 2022

@author: Sam

This module generates a variety of metrics for a pre-trained model to determine 
the overall performance of the model.

Methods
-------
get_metrics
    calculate all requiredmetrics
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Main method
    
    Modify the model path and source validation path to run experiment.
    
    """
    metrics = get_metrics(
        model_path = "D:/OneDrive - MMU/Complete_Final_Project/Lesion_Classification/Models/Inpaint_Telea/EfficientNetB4/SGD/64/EfficientNetB4_batchSize_0_opt_SGD_model.10.h5",
        src_path_val = r"D:/OneDrive - MMU/Complete_Final_Project/Lesion_Classification/Data/train_balanced_224x224_inpainted_telea/val"
    )

def get_metrics(model_path, src_path_val):
    """Calculate a variety of metrics for a trained model.
    
    Metrics include:
        * Sensitivity
        * Specificity
        * F1 Score
        * Precision
        * Accuracy
        * ROC 
        * AUC
    
    Parameters
    ----------
    model_path : str
        path to saved model
    src_path_val : str
        path to source folder of validation set
    
    Returns
    -------
    list
        list of all metrics in order specified above
    
    """
    # Load model and data from filepaths specified
    model = load_model(model_path)

    val_datagen = ImageDataGenerator(rescale=1 / 255.0,fill_mode="nearest") 

    valid_generator = val_datagen.flow_from_directory(directory=src_path_val,
                                                      color_mode="rgb",
                                                      batch_size=64,
                                                      target_size = (224,224),
                                                      class_mode="categorical",
                                                      subset='training',
                                                      shuffle=False,
                                                      seed=42) 

    # Generate predictions on dataset and retrieve original evaluation of model
    preds = model.predict(valid_generator)
    evaluation = model.evaluate(valid_generator)


    print("\nEvals:\n", "Val_Loss, Val_Accuracy, Val_AUC\n", evaluation, "\n")

    # Create confusion matrix

    y_pred = np.argmax(preds, axis=1)
    matrix = confusion_matrix(valid_generator.classes, y_pred)
    print("Confusion Matrix:\n", matrix, "\n")

    # Compute metrics
    TP, FP, FN, TN = matrix[0][0], matrix[1][0], matrix[0][1], matrix[1][1]
    
    ###### If in doubt with the confusion matrix values, use this plot to help understand the correct TP,FP positioning
    #disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Mel", "Oth"])
    #disp.plot()
    #plt.show()
    
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    f1 = 2*TP/(2*TP+FP+FN)
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes =2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(valid_generator.classes, preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", precision)
    print("Accuracy:", accuracy)
    #print("ROC:", roc_auc[0])
    #print("AUC:", roc_auc[1])
    
    return [sensitivity, specificity, f1, precision, accuracy, roc_auc[0], roc_auc[1]]

if __name__ == '__main__':
    main()
