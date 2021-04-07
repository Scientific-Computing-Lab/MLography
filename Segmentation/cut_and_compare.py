#!/usr/bin/env python3.6
from __future__ import print_function
import sys
print(sys.path)
sys.path.append('/home/yonif/.conda/envs/pca_kmeans_change_detection/lib/python3.6/site-packages')
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import argparse
from matplotlib import pyplot as plt
from keras import backend as K
import tensorflow as tf
from networks.dextr import DEXTR
from mypath import Path
from helpers import helpers as helpers
from scipy.misc import imread, imsave, imresize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import skimage.transform
from skimage import color
from skimage.color import rgb2gray
from skimage.color import gray2rgb
import CNN_Registration.src.Registration as Registration
from CNN_Registration.src.utils.utils import *
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from skimage import filters
from skimage import exposure

################PCA-KMEANS####################################################################

def find_vector_set(diff_image, new_size, window_size):
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / (window_size*window_size)), window_size*window_size))
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block = diff_image[j:j + window_size, k:k + window_size]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + (window_size*window_size)
            j = j + (window_size*window_size)
        i = i + 1

    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec  # mean normalization

    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new, window_size):
    i = window_size//2
    feature_vector_set = []

    while i < new[0] - window_size//2:
        j = window_size//2
        while j < new[1] - window_size//2:
            block = diff_image[i - window_size//2:i + window_size//2+1 , j - window_size//2:j + window_size//2+1]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j + 1
        i = i + 1

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size", FVS.shape)
    return FVS


def clustering(FVS, components, new, window_size):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)

    sort_index = sorted(count, key=count.get)
    least_index = sort_index[0]
    second_least_index = sort_index[1]
    change_map = np.reshape(output, (new[0] - (window_size//2)*2, new[1] - (window_size//2)*2))
    return least_index, second_least_index, change_map


def find_PCAKmeans(image1, image2, window_size):
    new_size = np.asarray(image1.shape) / window_size
    new_size = new_size.astype(int) * window_size
    image1 = imresize(image1, (new_size)).astype(np.int16)
    image2 = imresize(image2, (new_size)).astype(np.int16)
    diff_image = abs(image1 - image2)
    diff_image = color.rgb2gray(diff_image)
    imsave('/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/diff.jpg', diff_image)

    vector_set, mean_vec = find_vector_set(diff_image, new_size, window_size)
    pca = PCA()
    pca.fit(vector_set)
    EVS = pca.components_

    FVS = find_FVS(EVS, diff_image, mean_vec, new_size, window_size)
    components = 3
    least_index, second_least_index, change_map = clustering(FVS, components, new_size, window_size)

    change_map[change_map == least_index] = 255
    #change_map[change_map == second_least_index] = 128
    change_map[change_map != 255] = 0
    #change_map[np.logical_and(change_map != 128, change_map != 255)] = 0
    change_map = change_map.astype(np.uint8)

    kernel = np.asarray(((0, 0, 1, 0, 0),
                         (0, 1, 1, 1, 0),
                         (1, 1, 1, 1, 1),
                         (0, 1, 1, 1, 0),
                         (0, 0, 1, 0, 0)), dtype=np.uint8)

    cleanChangeMap = cv2.erode(change_map, kernel)
    imsave("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results_Homography/changemap_1_3_components_w7.jpg", change_map)
    imsave("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results_Homography/cleanchangemap_1_3_components_w7.jpg", cleanChangeMap)

####################################### Unstructured Change ###################################################

#Function to retrieve features from intermediate layers
def get_activations(model, layer_idx, X_batch):
     get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
     activations = get_activations([X_batch,0])
     return activations

#Function to extract features from intermediate layers
def extra_feat(img):
    #Using a VGG19 as feature extractor
    base_model = VGG19(weights='imagenet',include_top=False)
    img = imresize(img, (224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    block1_pool_features=get_activations(base_model, 3, x)
    block2_pool_features=get_activations(base_model, 6, x)
    block3_pool_features=get_activations(base_model, 10, x)
    block4_pool_features=get_activations(base_model, 14, x)
    block5_pool_features=get_activations(base_model, 18, x)

    x1 = tf.image.resize_images(block1_pool_features[0],[112,112])
    x2 = tf.image.resize_images(block2_pool_features[0],[112,112])
    x3 = tf.image.resize_images(block3_pool_features[0],[112,112])
    x4 = tf.image.resize_images(block4_pool_features[0],[112,112])
    x5 = tf.image.resize_images(block5_pool_features[0],[112,112])

    F = tf.concat([x3,x2,x1,x4,x5],3) #Change to only x1, x1+x2,x1+x2+x3..so on, inorder to visualize features from diffetrrnt blocks
    return F

def unstructured_change(img1, img2):

  sess = tf.InteractiveSession()
  F1=extra_feat(img1) #Features from image patch 1
  F1=tf.square(F1)
  F2=extra_feat(img2) #Features from image patch 2
  F2=tf.square(F2)
  d=tf.subtract(F1,F2)
  d=tf.square(d)
  d=tf.reduce_sum(d,axis=3)
  dis=(d.eval())   #The change map formed showing change at each pixels
  dis=np.resize(dis,[112,112])
  # Calculating threshold using Otsu's Segmentation method
  val = filters.threshold_otsu(dis[:,:])
  hist, bins_center = exposure.histogram(dis[:,:],nbins=256)
  plt.figure()
  plt.title('Unstructured change')
  plt.imshow(dis[:,:] < val, cmap='gray', interpolation='bilinear')
  plt.axis('off')
  plt.tight_layout()
  plt.show()
  plt.savefig("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/unstructured_change.jpg")

################################################################################################################

def CNN_Registration(IX, IY, mask):
    print(IX.shape)
    print(IY.shape)
    print(mask.shape)
    reg = Registration.CNN()
    X, Y, Z = reg.register(IX, IY)
    registered = tps_warp(Y, Z, IY, IX.shape)
    mask_registered = tps_warp(Y, Z, mask, IX.shape)
    imsave("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/aligned.jpg", registered)
    return registered, mask_registered

def homography(img1, img2, mask_img):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)

    # Apply ratio test
    good_draw = []
    good_without_list = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_draw.append([m])
            good_without_list.append(m)

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img2, kp2, img1, kp1, good_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(3)
    plt.axis('off')
    plt.imshow(img3)
    plt.title('matching descriptors')
    plt.savefig("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/matching.png")

    # Extract location of good matches
    points1 = np.zeros((len(good_without_list), 2), dtype=np.float32)
    points2 = np.zeros((len(good_without_list), 2), dtype=np.float32)

    for i, match in enumerate(good_without_list):
        points1[i, :] = kp2[match.queryIdx].pt
        points2[i, :] = kp1[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = img2.shape[:2]
    im2Reg = cv2.warpPerspective(img2, h, (width, height))
    imsave("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/aligned.jpg", im2Reg)
    mask_registered = cv2.warpPerspective(mask_img, h, (width, height))
    return im2Reg, mask_registered


def cut_images(path1, path2):

    modelName = 'dextr_pascal-sbd'
    pad = 50
    thres = 0.8

    # Handle input and output args
    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        net = DEXTR(nb_classes=1, resnet_layers=101, input_shape=(512, 512), weights=modelName,
                    num_input_channels=4, classifier='psp', sigmoid=True)

        #  Read image and click the points
        image_1 = np.array(Image.open(path1))
        image_2 = np.array(Image.open(path2))

        ################## In Case of Resize ##################################
        image_1 = cv2.resize(image_1, (2688, 2016), interpolation=cv2.INTER_AREA)
        image_2 = cv2.resize(image_2, (2688, 2016), interpolation=cv2.INTER_AREA)
             
        plt.figure(1)
        plt.ion()
        plt.axis('off')
        plt.imshow(image_1)
        plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')
        plt.show()        
        #################----image1----##############################################################################
        results_1 = []
        extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image_1, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image_1, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)
    
        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                          pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)
    
        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
    
        # Run a forward pass
        pred = net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
        result_1 = helpers.crop2fullmask(pred, bbox, im_size=image_1.shape[:2], zero_pad=True, relax=pad) > thres
    
        results_1.append(result_1)
    
        # Plot the results
        plt.imshow(helpers.overlay_masks(image_1 / 255, results_1))
        plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
        for i in range(image_1.shape[:2][0]):
            for j in range(image_1.shape[:2][1]):
                if result_1[i][j] == False:
                    image_1[i][j] = (0,0,0)
        imsave("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/cropped_1.jpg", image_1)
        #################----image2----##############################################################################

        plt.figure(2)
        plt.ion()
        plt.axis('off')
        plt.imshow(image_2)
        plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')
        plt.show()

        results_2 = []
        extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image_2, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image_2, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [
            pad,
            pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)

        # Run a forward pass
        pred = net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
        result_2 = helpers.crop2fullmask(pred, bbox, im_size=image_2.shape[:2], zero_pad=True, relax=pad) > thres
        results_2.append(result_2)

        # Plot the results
        plt.imshow(helpers.overlay_masks(image_2 / 255, results_2))
        plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
        for i in range(image_2.shape[:2][0]):
            for j in range(image_2.shape[:2][1]):
                if result_2[i][j] == False:
                    image_2[i][j] = (0,0,0)
        imsave("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/cropped_2.jpg", image_2)
        return image_1, image_2, result_1, result_2

##################---Main---##########################################################################################
"""
#NOTICE: cut images resizes the images
image1, image2, result_1, result_2 = cut_images("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/IMG_6500.jpg", "/home/yonif/ChangeDetectionProject/Yoni_Test_Images/IMG_6512.jpg")
#image1, image2, result_1, result_2 = cut_images("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/circuit1.jpg", "/home/yonif/ChangeDetectionProject/Yoni_Test_Images/circuit2.jpg" )

mask = np.zeros((result_2.shape[0], result_2.shape[1], 3), dtype=np.uint8)
for i in range(image2.shape[:2][0]):
    for j in range(image2.shape[:2][1]):
        if result_2[i][j] == True:
            mask[i][j] = [255, 255, 255]

image2_registered, mask_registered = homography(image1, image2, mask)
#image2_registered, mask_registered = CNN_Registration(image1, image2, mask)

min_width = min(image1.shape[:2][0], image2.shape[:2][0])
min_height = min(image1.shape[:2][1], image2.shape[:2][1])
for i in range(min_width):
    for j in range(min_height):
        if mask_registered[i][j][0] == 0 or result_1[i][j] == False:
            image2_registered[i][j] = 0
            image1[i][j] = 0
imsave("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/FINAL_1.jpg", image1)
imsave("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results/FINAL_2.jpg", image2_registered)
"""
image1 = np.array(Image.open("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results_Homography/FINAL_1.jpg"))
image2_registered = np.array(Image.open("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/Results_Homography/FINAL_2.jpg"))

find_PCAKmeans(image1, image2_registered, 7)

#image1 = np.float32(np.array(Image.open("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/f1_yoni.png")))[:,:,:3]
#image2_registered = np.float32(np.array(Image.open("/home/yonif/ChangeDetectionProject/Yoni_Test_Images/f2_yoni.png")))[:,:,:3]
#unstructured_change(image1, image2_registered)