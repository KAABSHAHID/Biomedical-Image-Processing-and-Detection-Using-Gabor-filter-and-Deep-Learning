# Biomedical-Image-Processing-and-Detection-Using-Gabor-filter-and-Deep-Learning  

This repository implements a deep learning-based approach for the detection and classification of colon and lung cancer using biomedical images. The methodology involves Gabor filtering for image preprocessing, GhostNet for feature extraction, and a fully connected Artificial Neural Network (ANN) for classification.  

## Table of Contents  
- Introduction  
- Methodology  
- 1. Gabor Filtering  
- 2. GhostNet Feature Extraction  
- 3. Artificial Neural Network (ANN) Classifier  
- Dataset  
- Results  
- How to Run  
- References  
## Introduction  
Cancer detection, particularly for colon and lung cancer, is a critical problem in biomedical research. This project applies a hybrid approach combining Gabor filters for image preprocessing and deep learning models for classification.   
## Methodology  
### 1. Gabor Filtering  
Gabor filters are applied to the input biomedical images to preprocess them by enhancing features such as edges and textures. This step helps in highlighting the frequency and orientation data in the image, making the features more prominent for further analysis.  

### 2. GhostNet Feature Extraction  
The preprocessed images are passed through the GhostNet model, a lightweight neural network designed for efficient feature extraction. GhostNet reduces the computation required while maintaining feature quality. In this step, the images are transformed into a feature vector for classification.  

### 3. Artificial Neural Network (ANN) Classifier  
The extracted features are fed into a fully connected Artificial Neural Network (ANN). The ANN contains multiple hidden layers with ReLU activation functions and a softmax output layer, which classifies the images into one of five categories:  

- Lung Adenocarcinoma  
- Lung Squamous Cell Carcinoma  
- Lung Benign Tissue  
- Colon Adenocarcinoma  
- Colon Benign Tissue  
## Dataset   
The dataset used in this project consists of histopathological images of colon and lung tissues. These images are processed and classified into five categories using Gabor filtering, GhostNet feature extraction, and ANN classification.  

## Results  
The model achieved 99.73% accuracy during training and 96% accuracy on the test set. These results demonstrate the effectiveness of combining Gabor filters, GhostNet feature extraction, and deep learning in cancer detection tasks.  

## How to Run  
```bash
git clone https://github.com/KAABSHAHID/Biomedical-Image-Processing-and-Detection-Using-Gabor-filter-and-Deep-Learning.git
```
install the libraries imported in the `cancer.py`  
update the file paths both input and output  
Then run  
```bash
python cancer.py
```
