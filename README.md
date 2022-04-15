# Dark Corner Artifact Removal

This repository contains all code and experiments created for dark corner artifact removal.

Please read the DISCLAIMER before using any of the methods or code from this repository.

If you use any part of the DCA masking/removal process in this research project, please consider citing this paper:

```
@inproceedings{pewton2022dark,
  title={Dark Corner on Skin Lesion Image Dataset: Does it matter?},
  author={Pewton, Samuel William and Yap, Moi Hoon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={TBC},
  year={2022}
}
```

The main dataset used in this research is the result from the duplicate removal process detailed in [this](https://github.com/mmu-dermatology-research/isic_duplicate_removal_strategy) repository.

If using this dataset please consider citing the following paper:

```
@article{cassidy2021isic,
 title   = {Analysis of the ISIC Image Datasets: Usage, Benchmarks and Recommendations},
 author  = {Bill Cassidy and Connah Kendrick and Andrzej Brodzicki and Joanna Jaworek-Korjakowska and Moi Hoon Yap},
 journal = {Medical Image Analysis},
 year    = {2021},
 issn    = {1361-8415},
 doi     = {https://doi.org/10.1016/j.media.2021.102305},
 url     = {https://www.sciencedirect.com/science/article/pii/S1361841521003509}
} 
```

## File Structure
** - needs to be created by user

```
Dark_Corner_Artifact_Removal
├─ Data
|   └─ Annotations
|   └─ DCA_Masks
|   |      └─ train
|   |	   |    └─ mel
|   |	   |	└─ oth
|   |	   └─ val
|   |	   	└─ mel
|   |		└─ oth
|   └─ Dermofit**
|   └─ Metrics_Dermofit
|   |		└─ generated_metrics
|   |		└─ input**
|   |		|    └─ gt**
|   |		|    └─ large**
|   |		|    └─ medium**
|   |		|    └─ oth**
|   |		|    └─ small**
|   |		└─ output**
|   |		     └─ large**
|   |		     └─ medium**
|   |		     └─ oth**
|   |		     └─ small**
|   └─ train_balanced_224x224
|   |      └─ train
|   |	        └─ mel
|   |		└─ oth
|   |	   └─ val
|   |	   	└─ mel
|   |		└─ oth
|   └─ train_balanced_224x224_inpainted_ns
|   |      └─ train
|   |	   |    └─ mel
|   |	   |	└─ oth
|   |	   └─ val
|   |	   	└─ mel
|   |		└─ oth
|   └─ train_balanced_224x224_inpainted_telea
|          └─ train
|	   |    └─ mel
|	   |	└─ oth
|	   └─ val
|	   	└─ mel
|		└─ oth
├─ Models
|    └─ Baseline
|    |	  └─ .. all model experiments ..
|    └─ Inpaint_NS
|    |    └─ .. all model experiments ..
|    └─ Inpaint_Telea
|         └─ .. all model experiments ..
├─ Modules
└─ Notebooks
     └─ 0 - Preliminary Experiments
     └─ 1 - Dataset
     └─ 2 - Dynamic Masking
     └─ 3 - Image Modifications
     └─ 4 - Results
```

## Project Pre-requisite Setup

1. Download <code>EDSR_x4.pb</code> from <url>https://github.com/Saafke/EDSR_Tensorflow</url> and save inside the <code>Models</code> directory. <code>/Models/EDSR_x4.pb</code>
2. Create Dermofit directory inside Data directory. <code>Data/Dermofit</code>
3. Load Dermofit image library <url>https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library</url> inside the Dermofit directory. This will be split into many sub-folders (AK, ALLBCC, ALLDF, etc...), leave that as it is.
4. Create Metrics_Dermofit file structure as shown above.
5. Modify filepaths and run <code>/Modules/prepare_dermofit.py</code> **only** if using dermofit.

## Requirements

This project requires the following installations:

 * Python 3.x
 * Anaconda
 * [pandas](https://pandas.pydata.org/)
 * [numpy](https://numpy.org/)
 * [scikit-learn](https://scikit-learn.org/stable/)
 * scikit-image
 * [Jupyter Notebook](https://jupyter.org/install)
 * [matplotlib](https://matplotlib.org/)
 * [OpenCV](https://docs.opencv.org/4.5.2/d5/de5/tutorial_py_setup_in_windows.html)
 * [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
 * [Tensorflow](https://www.tensorflow.org/install)
 * Tensorflow-GPU
 * CUDA
 * CuDNN
 * [Keras](https://keras.io/)

## Project Steps

### Generate all DCA masks
Load <code>/Notebooks/2 - Dynamic Masking/Mask All DCA Images.ipynb/</code> in Jupyter Notebook and run

### Modify All DCA Images
Load <code>/Notebooks/3 - Image Modifications/Inpaint Dataset.ipynb/</code> in Jupyter Notebook and run cells as required - recommended to run individually as the removal process is time consuming.

### Generate Dermofit Image Metrics
Run <code>/Modules/generate_dermofit_metrics.py</code>

### Train Models
Modify and run <code>/Modules/AbolatationStudy.py</code> as required

### Generate Model Performance Metrics
Modify and run <code>/Modules/model_performance.py</code> as required

### GradCAM Heatmaps
Load <code>/Notebooks/4 - Results/GradCAM Method Comparison.ipynb/</code> in Jupyter Notebook and run
Load <code>/Notebooks/4 - Results/GradCAM-Inscribed DCAs.ipynb/</code> in Jupyter Notebook and run


## Supplementary Material

Full results for the deep learning experiments:

**Baseline Model Results:**
<table>
	<tr>
		<td>Model</td><td>Settings</td><td colspan="3">Metrics</td><td colspan="3">Micro-Average</td>
	</tr>
	<tr>
		<td> </td>
		<td>Best Epoch</td>
		<td>Acc</td>
		<td>TPR</td>
		<td>TNR</td>
		<td>F1</td>
		<td>AUC</td>
		<td>Precision</td>
	</tr>
	<tr><td>VGG16</td><td>33</td><td>0.78</td><td>0.73</td><td>0.84</td><td>0.77</td><td>0.87</td><td>0.82</td></tr>
	<tr><td>VGG19</td><td>32</td><td>0.78</td><td>0.76</td><td>0.80</td><td>0.78</td><td>0.87</td><td>0.80</td></tr>
	<tr><td>Xception</td><td>20</td><td>0.81</td><td>0.76</td><td>0.86</td><td>0.80</td><td>0.88</td><td>0.84</td></tr>
	<tr><td>ResNet50</td><td>18</td><td>0.79</td><td>0.74</td><td>0.85</td><td>0.78</td><td>0.87</td><td>0.83</td></tr>
	<tr><td>ResNet101</td><td>6</td><td>0.78</td><td>0.70</td><td>0.85</td><td>0.76</td><td>0.85</td><td>0.82</td></tr>
	<tr><td>ResNet152</td><td>19</td><td>0.79</td><td>0.74</td><td>0.84</td><td>0.78</td><td>0.87</td><td>0.82</td></tr>
	<tr><td>ResNet50V2</td><td>14</td><td>0.77</td><td>0.73</td><td>0.82</td><td>0.76</td><td>0.85</td><td>0.80</td></tr>
	<tr><td>ResNet101V2</td><td>41</td><td>0.79</td><td>0.78</td><td>0.79</td><td>0.78</td><td>0.87</td><td>0.79</td></tr>
	<tr><td>ResNet152V2</td><td>25</td><td>0.78</td><td>0.77</td><td>0.78</td><td>0.78</td><td>0.85</td><td>0.78</td></tr>
	<tr><td>InceptionV3</td><td>36</td><td>0.80</td><td>0.81</td><td>0.80</td><td>0.80</td><td>0.88</td><td>0.80</td></tr>
	<tr><td>InceptionResNetV2</td><td>20</td><td>0.82</td><td>0.80</td><td>0.83</td><td>0.81</td><td>0.89</td><td>0.82</td></tr>
	<tr><td>DenseNet121</td><td>5</td><td>0.76</td><td>0.67</td><td>0.84</td><td>0.73</td><td>0.82</td><td>0.81</td></tr>
	<tr><td>DenseNet169</td><td>36</td><td>0.80</td><td>0.72</td><td>0.87</td><td>0.78</td><td>0.88</td><td>0.85</td></tr>
	<tr><td>DenseNet201</td><td>17</td><td>0.79</td><td>0.70</td><td>0.87</td><td>0.77</td><td>0.86</td><td>0.84</td></tr>
	<tr><td>EfficientNetB0</td><td>28</td><td>0.78</td><td>0.69</td><td>0.87</td><td>0.76</td><td>0.87</td><td>0.84</td></tr>
	<tr><td>EfficientNetB1</td><td>19</td><td>0.77</td><td>0.68</td><td>0.86</td><td>0.75</td><td>0.85</td><td>0.83</td></tr>
	<tr><td>EfficientNetB3</td><td>13</td><td>0.75</td><td>0.63</td><td>0.88</td><td>0.72</td><td>0.82</td><td>0.84</td></tr>
	<tr><td>EfficientNetB4</td><td>46</td><td>0.78</td><td>0.71</td><td>0.85</td><td>0.76</td><td>0.86</td><td>0.83</td></tr>
	
</table>


**Inpainting Results (Navier-Stokes based method)**
<table>
	<tr>
		<td>Model</td><td>Settings</td><td colspan="3">Metrics</td><td colspan="3">Micro-Average</td>
	</tr>
	<tr>
		<td> </td>
		<td>Best Epoch</td>
		<td>Acc</td>
		<td>TPR</td>
		<td>TNR</td>
		<td>F1</td>
		<td>AUC</td>
		<td>Precision</td>
	</tr>	
	<tr><td>VGG16</td><td>49</td><td>0.79</td><td>0.72</td><td>0.85</td><td>0.77</td><td>0.87</td><td>0.83</td></tr>
	<tr><td>VGG19</td><td>34</td><td>0.78</td><td>0.72</td><td>0.84</td><td>0.77</td><td>0.86</td><td>0.82</td></tr>
	<tr><td>Xception</td><td>19</td><td>0.80</td><td>0.78</td><td>0.83</td><td>0.80</td><td>0.88</td><td>0.82</td></tr>
	<tr><td>ResNet50</td><td>39</td><td>0.79</td><td>0.75</td><td>0.84</td><td>0.78</td><td>0.88</td><td>0.82</td></tr>
	<tr><td>ResNet101</td><td>33</td><td>0.79</td><td>0.71</td><td>0.87</td><td>0.77</td><td>0.87</td><td>0.85</td></tr>
	<tr><td>ResNet152</td><td>17</td><td>0.79</td><td>0.73</td><td>0.85</td><td>0.77</td><td>0.88</td><td>0.83</td></tr>
	<tr><td>ResNet50V2</td><td>20</td><td>0.79</td><td>0.76</td><td>0.81</td><td>0.78</td><td>0.87</td><td>0.80</td></tr>
	<tr><td>ResNet101V2</td><td>40</td><td>0.79</td><td>0.70</td><td>0.88</td><td>0.77</td><td>0.88</td><td>0.85</td></tr>
	<tr><td>ResNet152V2</td><td>23</td><td>0.78</td><td>0.75</td><td>0.80</td><td>0.77</td><td>0.86</td><td>0.79</td></tr>
	<tr><td>InceptionV3</td><td>22</td><td>0.79</td><td>0.77</td><td>0.80</td><td>0.79</td><td>0.87</td><td>0.80</td></tr>
	<tr><td>InceptionResNetV2</td><td>19</td><td>0.80</td><td>0.81</td><td>0.79</td><td>0.80</td><td>0.88</td><td>0.79</td></tr>
	<tr><td>DenseNet121</td><td>37</td><td>0.80</td><td>0.77</td><td>0.83</td><td>0.79</td><td>0.88</td><td>0.82</td></tr>
	<tr><td>DenseNet169</td><td>12</td><td>0.77</td><td>0.75</td><td>0.78</td><td>0.76</td><td>0.85</td><td>0.77</td></tr>
	<tr><td>DenseNet201</td><td>25</td><td>0.78</td><td>0.75</td><td>0.80</td><td>0.77</td><td>0.86</td><td>0.79</td></tr>
	<tr><td>EfficientNetB0</td><td>20</td><td>0.77</td><td>0.66</td><td>0.88</td><td>0.74</td><td>0.86</td><td>0.85</td></tr>
	<tr><td>EfficientNetB1</td><td>13</td><td>0.76</td><td>0.75</td><td>0.78</td><td>0.76</td><td>0.83</td><td>0.77</td></tr>
	<tr><td>EfficientNetB3</td><td>28</td><td>0.77</td><td>0.73</td><td>0.82</td><td>0.76</td><td>0.86</td><td>0.80</td></tr>
	<tr><td>EfficientNetB4</td><td>37</td><td>0.78</td><td>0.69</td><td>0.88</td><td>0.76</td><td>0.87</td><td>0.85</td></tr>
	
</table>

**Inpainting Results (Telea based method)**
<table>
	<tr>
		<td>Model</td><td>Settings</td><td colspan="3">Metrics</td><td colspan="3">Micro-Average</td>
	</tr>
	<tr>
		<td> </td>
		<td>Best Epoch</td>
		<td>Acc</td>
		<td>TPR</td>
		<td>TNR</td>
		<td>F1</td>
		<td>AUC</td>
		<td>Precision</td>
	</tr>	
	<tr><td>VGG16</td><td>54</td><td>0.79</td><td>0.75</td><td>0.82</td><td>0.78</td><td>0.87</td><td>0.81</td></tr>
	<tr><td>VGG19</td><td>10</td><td>0.71</td><td>0.64</td><td>0.78</td><td>0.69</td><td>0.78</td><td>0.74</td></tr>
	<tr><td>Xception</td><td>10</td><td>0.79</td><td>0.75</td><td>0.84</td><td>0.79</td><td>0.88</td><td>0.82</td></tr>
	<tr><td>ResNet50</td><td>10</td><td>0.77</td><td>0.74</td><td>0.81</td><td>0.77</td><td>0.87</td><td>0.79</td></tr>
	<tr><td>ResNet101</td><td>33</td><td>0.80</td><td>0.79</td><td>0.80</td><td>0.80</td><td>0.88</td><td>0.79</td></tr>
	<tr><td>ResNet152</td><td>23</td><td>0.79</td><td>0.78</td><td>0.80</td><td>0.79</td><td>0.87</td><td>0.80</td></tr>
	<tr><td>ResNet50V2</td><td>23</td><td>0.78</td><td>0.81</td><td>0.76</td><td>0.79</td><td>0.87</td><td>0.77</td></tr>
	<tr><td>ResNet101V2</td><td>25</td><td>0.79</td><td>0.79</td><td>0.78</td><td>0.79</td><td>0.87</td><td>0.78</td></tr>
	<tr><td>ResNet152V2</td><td>29</td><td>0.79</td><td>0.75</td><td>0.83</td><td>0.78</td><td>0.87</td><td>0.81</td></tr>
	<tr><td>InceptionV3</td><td>18</td><td>0.79</td><td>0.76</td><td>0.81</td><td>0.78</td><td>0.86</td><td>0.80</td></tr>
	<tr><td>InceptionResNetV2</td><td>11</td><td>0.79</td><td>0.69</td><td>0.88</td><td>0.76</td><td>0.88</td><td>0.86</td></tr>
	<tr><td>DenseNet121</td><td>61</td><td>0.80</td><td>0.80</td><td>0.80</td><td>0.80</td><td>0.88</td><td>0.80</td></tr>
	<tr><td>DenseNet169</td><td>18</td><td>0.78</td><td>0.80</td><td>0.75</td><td>0.78</td><td>0.87</td><td>0.76</td></tr>
	<tr><td>DenseNet201</td><td>38</td><td>0.79</td><td>0.73</td><td>0.84</td><td>0.77</td><td>0.87</td><td>0.82</td></tr>
	<tr><td>EfficientNetB0</td><td>18</td><td>0.78</td><td>0.72</td><td>0.85</td><td>0.77</td><td>0.87</td><td>0.83</td></tr>
	<tr><td>EfficientNetB1</td><td>51</td><td>0.78</td><td>0.79</td><td>0.86</td><td>0.78</td><td>0.87</td><td>0.77</td></tr>
	<tr><td>EfficientNetB3</td><td>49</td><td>0.79</td><td>0.78</td><td>0.79</td><td>0.78</td><td>0.87</td><td>0.79</td></tr>
	<tr><td>EfficientNetB4</td><td>10</td><td>0.75</td><td>0.64</td><td>0.86</td><td>0.72</td><td>0.82</td><td>0.82</td></tr>
	
</table>

## References
```
@article{cassidy2021isic,
 title   = {Analysis of the ISIC Image Datasets: Usage, Benchmarks and Recommendations},
 author  = {Bill Cassidy and Connah Kendrick and Andrzej Brodzicki and Joanna Jaworek-Korjakowska and Moi Hoon Yap},
 journal = {Medical Image Analysis},
 year    = {2021},
 issn    = {1361-8415},
 doi     = {https://doi.org/10.1016/j.media.2021.102305},
 url     = {https://www.sciencedirect.com/science/article/pii/S1361841521003509}
} 

@misc{rosebrock_2020, 
 title   = {Grad-cam: Visualize class activation maps with Keras, tensorflow, and Deep Learning}, 
 url     = {https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/}, 
 journal = {PyImageSearch}, 
 author  = {Rosebrock, Adrian}, 
 year    = {2020}, 
 month   = {3},
 note    = {[Accessed: 10-03-2022]}
} 

@article{scikit-image,
 title   = {scikit-image: image processing in {P}ython},
 author  = {van der Walt, {S}t\'efan and {S}ch\"onberger, {J}ohannes {L}. and
           {Nunez-Iglesias}, {J}uan and {B}oulogne, {F}ran\c{c}ois and {W}arner,
           {J}oshua {D}. and {Y}ager, {N}eil and {G}ouillart, {E}mmanuelle and
           {Y}u, {T}ony and the scikit-image contributors},
 year    = {2014},
 month   = {6},
 keywords = {Image processing, Reproducible research, Education,
             Visualization, Open source, Python, Scientific programming},
 volume  = {2},
 pages   = {e453},
 journal = {PeerJ},
 issn    = {2167-8359},
 url     = {https://doi.org/10.7717/peerj.453},
 doi     = {10.7717/peerj.453}
}

@article{scikit-learn,
 title   = {Scikit-learn: Machine Learning in {P}ython},
 author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal = {Journal of Machine Learning Research},
 volume  = {12},
 pages   = {2825--2830},
 year    = {2011}
}

@inproceedings{lim2017enhanced,
  title  = {Enhanced deep residual networks for single image super-resolution},
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle= {Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages  = {136--144},
  year   = {2017}
}
```
