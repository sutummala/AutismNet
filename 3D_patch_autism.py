# Code created by Sudhakar on Nov 2020
# 3D patch based learning for Diagnosis of Autism Spectrum Disorder
# images were initially affine registered

import os
import tensorflow as tf
from tensorflow.keras import backend as K
#import tensorflow_addons as tfa
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import nibabel as nib
import numpy as np
import random

data_dir = '/media/sudhakar/SUDHAKAR/Work/Data/ABIDE/Healthy' # Path to the heatlhy subjects data directory
data_dir1 = '/media/sudhakar/SUDHAKAR/Work/Data/ABIDE/Autism' # Path to the autism subjects data directory

voi_size = [60, 60, 60]

image_tag = 'hrT1'

# preparing 3D patches for training, validation and testing 

def create_dataset(data_for_patches, tag):
    
    Healthy = []
    Autism = []
    
    save_patches = False
    
    for subject in os.listdir(data_for_patches):
        # healthy images
        image_path = os.path.join(data_for_patches, subject, 'mni')
        images = os.listdir(image_path)
            
        for image in images:
            if image_tag in image and image.endswith('reoriented.mni.nii'):
                
                print(f'extracting patches for {subject}')
                
                input_image = nib.load(os.path.join(image_path, image))
                input_image_data = input_image.get_fdata()
                
                input_image_data = (input_image_data-np.min(input_image_data))/(np.max(input_image_data)-np.min(input_image_data))
                
                x, y, z =  np.shape(input_image_data)
                #patch_size = [30, 30, 30]
                
                for i in range (0, x-60, 60):
                    for j in range(0, y-60, 60):
                        for k in range(0, z-60, 60):
                            patch = input_image_data[i:i+60, j:j+60, k:k+60]
                
                            if tag == 'healthy':
                                Healthy.append(np.array(patch))
                                print(f'healthy {np.shape(Healthy)}')
                                if save_patches:
                                    patch_image = nib.Nifti1Image(patch, input_image.affine)
                                    nib.save(patch_image, 'patch'+str(np.shape(Healthy)[0])+'.nii.gz')  
                            elif tag == 'autism':
                                Autism.append(np.array(patch))
                                print(f'diseased {np.shape(Autism)}')
                                if save_patches:
                                    patch_image = nib.Nifti1Image(patch, input_image.affine)
                                    nib.save(patch_image, 'patch'+str(np.shape(Autism)[0])+'.nii.gz')  
                            
    if tag == 'healthy':
        return Healthy
    else:
        return Autism
                
        
healthy_data = create_dataset(data_dir, 'healthy')
autism_data = create_dataset(data_dir1, 'autism')    

y_cor = np.ones(np.shape(healthy_data)[0])
y_incor = np.zeros(np.shape(autism_data)[0])

y_true = np.concatenate((y_cor, y_incor))
#y_true = tf.keras.utils.to_categorical(y_true, 2)
X = np.concatenate((healthy_data, autism_data))
#X = X.reshape(list(X.shape) + [1])

# Saving data

np.save('/home/sudhakar/data', healthy_data)
np.save('/home/sudhakar/data', autism_data)
np.save('/home/sudhakar/data', y_true)

print('data is ready for 3D patch-based learning')
