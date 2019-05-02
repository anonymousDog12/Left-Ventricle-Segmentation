# Cardiac MR Left Ventricle Segmentation
UCLA-CS269 Project -- Cardiac MR Left Ventricle Segmentation Challenge.

### Abstract
Image segmentation of the left ventricle from cardiac magnetic resonance imaging is a crucial but tedious step for clinical cardiac health diagnosis. In this project, we proposed to use convolutional neural network combined with deformable model to conduct medical image segmentation. A three-step approach is proposed to deal with the low-contrast nature of medical image and relative small size of available data. Finally, the performance of the segmentation algorithm is evaluated from both quantitative and qualitative aspects.  

  
### Members
Qi Qu  
Jingxi Yu  
Changyu Yan  
Sha Liu  

### Dataset
Data could be downloaded on this site after registering http://smial.sri.utoronto.ca/LV_Challenge/Home.html. 
All data is expected to be released to ../Data/

### Codes 
ROI_detection.ipynb shows the process of ROI detection and also how to loads and prepare the data in this challenge.  
StackedAE.ipynb && PCA_autoencoder.ipynb shows the process of shape prior inference.  
SAE+ActiveContour.ipynb shows the last step of computing the contour.  

