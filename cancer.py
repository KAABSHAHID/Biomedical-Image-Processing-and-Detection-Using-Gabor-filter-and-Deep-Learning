# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:20:50 2024

@author: mkaab
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision import models
import timm


def build_gabor_filter(ksize, sigma, theta, lambd, gamma, psi):
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)


ksize = 28
sigma = 0.5
theta_list = [0, np.pi/4, np.pi/2, 3*np.pi/4]
lambd = 5.0
gamma = 0.5
psi = 0


def gabor_filter(input_folder,output_folder):
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')): 
            image_path = os.path.join(input_folder, filename)
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

            if image is not None:
                
                for theta in theta_list:
                    gabor_filter = build_gabor_filter(ksize, sigma, theta, lambd, gamma, psi)
                    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_filter)
                
                output_path = os.path.join(output_folder, f'filtered_{filename}')
                cv2.imwrite(output_path, filtered_image)
                
                #image_attribute.append((image_path,filtered_image)) 


            else:
                print(f"Failed to load image: {filename}")
    


gabor_filter('colon_cancer/lung_colon_image_set/colon_image_sets/colon_aca','colon_cancer/lung_colon_image_set/colon_image_sets/colon_aca_gabor')
gabor_filter('colon_cancer/lung_colon_image_set/colon_image_sets/colon_n','colon_cancer/lung_colon_image_set/colon_image_sets/colon_n_gabor')
gabor_filter('colon_cancer/lung_colon_image_set/lung_image_sets/lung_aca','colon_cancer/lung_colon_image_set/lung_image_sets/lung_aca_gabor')
gabor_filter('colon_cancer/lung_colon_image_set/lung_image_sets/lung_scc','colon_cancer/lung_colon_image_set/lung_image_sets/lung_scc_gabor')
gabor_filter('colon_cancer/lung_colon_image_set/lung_image_sets/lung_n','colon_cancer/lung_colon_image_set/lung_image_sets/lung_n_gabor')





#ghostNet feature extraction

model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
model.eval()

from PIL import Image
from torchvision import transforms

model = torch.nn.Sequential(*list(model.children())[:-1])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def ghostnet_feature_extractor(image_folder, output_path,label):
    
    image_list = os.listdir(image_folder)

    all_features = []

    # Process images in batches
    batch_size = 16
    for i in range(0, len(image_list), batch_size):
        batch_images = image_list[i:i+batch_size]
        
        input_batch = []
        
        for image_name in batch_images:
            image_path = os.path.join(image_folder, image_name)
            input_image = Image.open(image_path).convert("RGB")  # convert grayscale to RGB
            input_tensor = preprocess(input_image)
            input_batch.append(input_tensor)
        
        input_batch = torch.stack(input_batch)
        
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')
        
        with torch.no_grad():
            features = model(input_batch)
        
        features = features.cpu().numpy()
        
        all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)
    all_features = all_features.reshape(all_features.shape[0], -1)


    print(f"Extracted features shape: {all_features.shape}")




    np.save(output_path, all_features)  



    loaded_features = np.load(output_path)


    dfm = []
    dfm = loaded_features
    dfm = pd.DataFrame(dfm)
    dfm["label"] = label
    return dfm


df_c_aca = ghostnet_feature_extractor('colon_cancer/lung_colon_image_set/colon_image_sets/colon_aca_gabor','colon_cancer/lung_colon_image_set/colon_mal_features.npy','colon_adenocarcinoma')
df_c_n = ghostnet_feature_extractor('colon_cancer/lung_colon_image_set/colon_image_sets/colon_n_gabor','colon_cancer/lung_colon_image_set/colon_n_features.npy', 'colon_healthy')
df_l_aca = ghostnet_feature_extractor('colon_cancer/lung_colon_image_set/lung_image_sets/lung_aca_gabor','colon_cancer/lung_colon_image_set/lung_aca_features.npy', 'lung_adenocarcinoma')
df_l_scc = ghostnet_feature_extractor('colon_cancer/lung_colon_image_set/lung_image_sets/lung_scc_gabor','colon_cancer/lung_colon_image_set/lung_scc_features.npy','lung_squamouscarcinoma')
df_l_n = ghostnet_feature_extractor('colon_cancer/lung_colon_image_set/lung_image_sets/lung_n_gabor','colon_cancer/lung_colon_image_set/lung_n_features.npy', 'lung_healthy')




#making the dataset by concatenation
df_concatenated = pd.concat([df_c_aca, df_c_n, df_l_aca, df_l_scc, df_l_n], axis=0)
df_concatenated.reset_index(drop=True, inplace=True)

#one hot encode
one_hot_encoded = pd.get_dummies(df_concatenated.iloc[:, -1], prefix='label')
df_final = pd.concat([df_concatenated.iloc[:, :-1], one_hot_encoded], axis=1)

#shuffle
df_shuffled = df_final.sample(frac=1).reset_index(drop=True)




#model training

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

#df_shuffled.iloc[1,1279]
x = df_shuffled.iloc[:,:1280]
y = df_shuffled.iloc[:,1280:]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(5, activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 
model.fit(X_train, y_train, epochs=50, batch_size=16) # 99%



loss, accuracy = model.evaluate(X_test, y_test) # 96%





