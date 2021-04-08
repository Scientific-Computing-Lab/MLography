# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import random as rand
import skimage, skimage.morphology
import matplotlib.pyplot as plt
import os

def cluster(in_dir, img_name, out_dir=None, wi_dir=None):
    gray = cv2.imread(os.path.join(in_dir, img_name), 0)
    image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)[1]
    image=image/255
    distance_map = ndimage.distance_transform_edt(image)
    
    
    local_max = peak_local_max(distance_map, indices=False, min_distance=20, labels=image, exclude_border=0)
    markers = ndimage.label(image)[0]
    labels = watershed(-distance_map, markers, mask=image)
    cnts_mask = np.zeros(image.shape, np.uint8)
    
    # Iterate through unique labels
    n = 0
    Areas = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # Create a mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
    
        # Find contours and determine contour area
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area == 0:
            continue
        Areas.append(area)
        n+=1
        print('Grain particle' ,'of number ' , n , 'has area = ',area)
        # total_area += area
        cv2.drawContours(image,[c],-1,(rand.randint(0,255),rand.randint(0,255),rand.randint(0,255)),-1)
        cv2.drawContours(cnts_mask, [c], -1, (255), 0)
    
    cv2.imwrite(os.path.join(out_dir, "contours_" + img_name), cnts_mask)
    in_image = cv2.imread(os.path.join(wi_dir, img_name))
    
    img_with_cnts = np.zeros_like(in_image)
    img_with_cnts[:,:,0] = cnts_mask
    img_with_cnts[:,:,1] = cnts_mask
    img_with_cnts[:,:,2] = cnts_mask
    black_mask = img_with_cnts == 0
    img_with_cnts[black_mask] =  in_image[black_mask] * 0.7
    
    cv2.imwrite(os.path.join(out_dir, "masked_cnts_" + img_name), img_with_cnts)
    cv2.imwrite(os.path.join(out_dir, img_name), image)
        

in_dir = "/home/matanr/MLography/Segmentation/unet/data/full_segmented_edges_binary/"
out_dir = "/home/matanr/MLography/Segmentation/unet/data/clustered"
wi_dir = "/home/matanr/MLography/Segmentation/unet/data/without_impurities/"

cnts_mask = cluster(in_dir, "32.jpg", out_dir, wi_dir)
cnts_mask = cluster(in_dir, "25.jpg", out_dir, wi_dir)


