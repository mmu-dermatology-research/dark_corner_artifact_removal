# -*- coding: utf-8 -*-
"""This module is a code dump for the final showcase notebook. It has been created
to limit the amount of code visible on the notebook to reduce clutter.

The methods contained within this module are direct copies of code that has been 
used in other notebooks/modules to conduct the entire experiment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from isic_data import get_data
import os, os.path
import cv2
from PIL import Image
import gradcam
import tensorflow as tf
import tensorflow.keras as keras


def gradcam_method_comparison(base_model, ns_model, telea_model):
    img_name = "ISIC2019_0063404_oth"
    base_image = cv2.imread(r"..\\Data\\train_balanced_224x224\\val\\" + img_name[-3:] + "\\" + img_name + ".jpg")
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    base_image = base_image.astype('float32') / 255
    base_image = np.expand_dims(base_image, axis=0)
    ns_image = cv2.imread(r"..\\Data\\train_balanced_224x224_inpainted_ns\\val\\" + img_name[-3:] + "\\" + img_name + ".png")
    ns_image = cv2.cvtColor(ns_image, cv2.COLOR_BGR2RGB)
    ns_image = ns_image.astype('float32') / 255
    ns_image = np.expand_dims(ns_image, axis=0)
    telea_image = cv2.imread(r"..\\Data\\train_balanced_224x224_inpainted_telea\\val\\" + img_name[-3:] + "\\" + img_name + ".png")
    telea_image = cv2.cvtColor(telea_image, cv2.COLOR_BGR2RGB)
    telea_image = telea_image.astype('float32') / 255
    telea_image = np.expand_dims(telea_image, axis=0)
    base_pred = base_model.predict(base_image)
    ns_pred = ns_model.predict(ns_image)
    telea_pred = telea_model.predict(telea_image)
    base_i = np.argmax(base_pred[0])
    ns_i = np.argmax(ns_pred[0])
    telea_i = np.argmax(telea_pred[0])
    base_cam = gradcam.GradCAM(base_model, base_i, 'conv_7b_ac')
    ns_cam = gradcam.GradCAM(ns_model, ns_i, 'conv_7b_ac')
    telea_cam = gradcam.GradCAM(telea_model, telea_i, 'conv_7b_ac')
    base_heatmap = base_cam.compute_heatmap(base_image)
    ns_heatmap = ns_cam.compute_heatmap(ns_image)
    telea_heatmap = telea_cam.compute_heatmap(telea_image)
    base_image = cv2.imread(r"..\\Data\\train_balanced_224x224\\val\\" + img_name[-3:] + "\\" + img_name + ".jpg")
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(base_image, (0,0), (90,20), (0,0,0), -1);
    cv2.putText(base_image, "GT: " + img_name[-3:], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    ns_image = cv2.imread(r"..\\Data\\train_balanced_224x224_inpainted_ns\\val\\" + img_name[-3:] + "\\" + img_name + ".png")
    ns_image = cv2.cvtColor(ns_image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(ns_image, (0,0), (90,20), (0,0,0), -1);
    cv2.putText(ns_image, "GT: " + img_name[-3:], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    telea_image = cv2.imread(r"..\\Data\\train_balanced_224x224_inpainted_telea\\val\\" + img_name[-3:] + "\\" + img_name + ".png")
    telea_image = cv2.cvtColor(telea_image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(telea_image, (0,0), (90,20), (0,0,0), -1);
    cv2.putText(telea_image, "GT: " + img_name[-3:], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    (base_heatmap, base_output) = base_cam.overlay_heatmap(base_heatmap, base_image, alpha=0.5)
    (ns_heatmap, ns_output) = ns_cam.overlay_heatmap(ns_heatmap, ns_image, alpha=0.5)
    (telea_heatmap, telea_output) = telea_cam.overlay_heatmap(telea_heatmap, telea_image, alpha=0.5)
    cv2.rectangle(base_output, (0,0), (100,20), (0,0,0), -1);
    cv2.putText(base_output, "Pred: mel" if base_i == 0 else "Pred: oth", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(ns_output, (0,0), (100,20), (0,0,0), -1);
    cv2.putText(ns_output, "Pred: mel" if ns_i == 0 else "Pred: oth", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(telea_output, (0,0), (100,20), (0,0,0), -1);
    cv2.putText(telea_output, "Pred: mel" if telea_i == 0 else "Pred: oth", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    fig, axes = plt.subplots(3, 3, figsize = (15,15));
    axes[0][0].imshow(base_image);
    axes[0][0].set_title("Baseline", fontsize = 20)
    axes[0][0].set_ylabel('Original', fontsize = 20)
    axes[1][0].imshow(base_heatmap);
    axes[1][0].set_ylabel('Heatmap', fontsize = 20)
    axes[2][0].imshow(base_output);
    axes[2][0].set_ylabel('Combined', fontsize = 20)
    axes[0][1].imshow(ns_image);
    axes[0][1].set_title("Navier Stokes", fontsize = 20)
    axes[1][1].imshow(ns_heatmap);
    axes[2][1].imshow(ns_output);
    axes[0][2].imshow(telea_image);
    axes[0][2].set_title("Telea", fontsize = 20)
    axes[1][2].imshow(telea_heatmap);
    axes[2][2].imshow(telea_output);
    for row in axes:
        for ax in row:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0.04, hspace = 0.04)

def gradcam_inscribed_dca(base_model):
    img_name = "ISIC2019_0027486_oth"
    clean_image = cv2.imread(r"..\\Data\\train_balanced_224x224\\val\\" + img_name[-3:] + "\\" + img_name + ".jpg")
    clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    s_mask = np.asarray(Image.open(r"..\Data\DCA_Masks\val\mel\ISIC2019_0071951_mel_MASK.png"))
    m_mask = np.asarray(Image.open(r"..\Data\DCA_Masks\val\mel\ISIC2019_0053555_mel_MASK.png"))
    l_mask = np.asarray(Image.open(r"..\Data\DCA_Masks\val\mel\ISIC2019_0057747_mel_MASK.png"))
    # small mask image
    s_masked_image = clean_image.copy()
    s_masked_image[s_mask.astype(bool)] = 0
    s_masked_image = s_masked_image.astype('float32') / 255
    s_masked_image = np.expand_dims(s_masked_image, axis=0)
    # med mask image
    m_masked_image = clean_image.copy()
    m_masked_image[m_mask.astype(bool)] = 0
    m_masked_image = m_masked_image.astype('float32') / 255
    m_masked_image = np.expand_dims(m_masked_image, axis=0)
    # large mask image
    l_masked_image = clean_image.copy()
    l_masked_image[l_mask.astype(bool)] = 0
    l_masked_image = l_masked_image.astype('float32') / 255
    l_masked_image = np.expand_dims(l_masked_image, axis=0)
    # finish off clean images
    clean_image = clean_image.astype('float32') / 255
    clean_image = np.expand_dims(clean_image, axis=0)
    clean_pred = base_model.predict(clean_image)
    s_pred = base_model.predict(s_masked_image)
    m_pred = base_model.predict(m_masked_image)
    l_pred = base_model.predict(l_masked_image)
    clean_i = np.argmax(clean_pred[0])
    s_i = np.argmax(s_pred[0])
    m_i = np.argmax(m_pred[0])
    l_i = np.argmax(l_pred[0])
    clean_cam = gradcam.GradCAM(base_model, clean_i, 'conv_7b_ac')
    s_cam = gradcam.GradCAM(base_model, s_i, 'conv_7b_ac')
    m_cam = gradcam.GradCAM(base_model, m_i, 'conv_7b_ac')
    l_cam = gradcam.GradCAM(base_model, l_i, 'conv_7b_ac')
    clean_heatmap = clean_cam.compute_heatmap(clean_image)
    s_heatmap = s_cam.compute_heatmap(s_masked_image)
    m_heatmap = m_cam.compute_heatmap(m_masked_image)
    l_heatmap = l_cam.compute_heatmap(l_masked_image)
    clean_image = cv2.imread(r"..\\Data\\train_balanced_224x224\\val\\" + img_name[-3:] + "\\" + img_name + ".jpg")
    clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    s_masked_image = clean_image.copy()
    s_masked_image[s_mask.astype(bool)] = 0
    m_masked_image = clean_image.copy()
    m_masked_image[m_mask.astype(bool)] = 0
    l_masked_image = clean_image.copy()
    l_masked_image[l_mask.astype(bool)] = 0
    (clean_heatmap, clean_output) = clean_cam.overlay_heatmap(clean_heatmap, clean_image, alpha=0.5)
    (s_heatmap, s_output) = s_cam.overlay_heatmap(s_heatmap, s_masked_image, alpha=0.5)
    (m_heatmap, m_output) = m_cam.overlay_heatmap(m_heatmap, m_masked_image, alpha=0.5)
    (l_heatmap, l_output) = l_cam.overlay_heatmap(l_heatmap, l_masked_image, alpha=0.5)
    cv2.rectangle(clean_image, (0,0), (90,20), (0,0,0), -1);
    cv2.putText(clean_image, "GT: " + img_name[-3:], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(s_masked_image, (0,0), (90,20), (0,0,0), -1);
    cv2.putText(s_masked_image, "GT: " + img_name[-3:], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(m_masked_image, (0,0), (90,20), (0,0,0), -1);
    cv2.putText(m_masked_image, "GT: " + img_name[-3:], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(l_masked_image, (0,0), (90,20), (0,0,0), -1);
    cv2.putText(l_masked_image, "GT: " + img_name[-3:], (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(clean_output, (0,0), (100,20), (0,0,0), -1);
    cv2.putText(clean_output, "Pred: mel" if clean_i == 0 else "Pred: oth", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(s_output, (0,0), (100,20), (0,0,0), -1);
    cv2.putText(s_output, "Pred: mel" if s_i == 0 else "Pred: oth", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(m_output, (0,0), (100,20), (0,0,0), -1);
    cv2.putText(m_output, "Pred: mel" if m_i == 0 else "Pred: oth", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    cv2.rectangle(l_output, (0,0), (100,20), (0,0,0), -1);
    cv2.putText(l_output, "Pred: mel" if l_i == 0 else "Pred: oth", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
    fig, axes = plt.subplots(3, 4, figsize = (15,15));
    axes[0][0].imshow(clean_image);
    axes[1][0].imshow(clean_heatmap);
    axes[2][0].imshow(clean_output);
    axes[0][1].imshow(s_masked_image);
    axes[1][1].imshow(s_heatmap);
    axes[2][1].imshow(s_output);
    axes[0][2].imshow(m_masked_image);
    axes[1][2].imshow(m_heatmap);
    axes[2][2].imshow(m_output);
    axes[0][3].imshow(l_masked_image);
    axes[1][3].imshow(l_heatmap);
    axes[2][3].imshow(l_output);
    axes[0][0].set_ylabel('Original', fontsize = 20)
    axes[1][0].set_ylabel('Heatmap', fontsize = 20)
    axes[2][0].set_ylabel('Combined', fontsize = 20)
    axes[0][0].set_title('Original', fontsize = 20)
    axes[0][1].set_title("Small DCA", fontsize = 20)
    axes[0][2].set_title("Medium DCA", fontsize = 20)
    axes[0][3].set_title("Large DCA", fontsize = 20)
    for row in axes:
        for ax in row:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0.04, hspace = -0.45)
    
