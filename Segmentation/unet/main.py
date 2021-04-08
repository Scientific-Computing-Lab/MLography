"""
install via pip install
"""
import cv_algorithms


from model import *
from data import *
import os
from absl import flags, app
import time
import glob
import cv2 as cv
import numpy as np
import ray
import matplotlib.pyplot as plt
from scipy import ndimage

import random as rand

import skimage, skimage.morphology
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from skimage.feature import peak_local_max
from skimage.morphology import watershed

import subprocess
import multiprocessing

num_threads = 40

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_name = 'unet_membrane.hdf5'
# keep_training = True

FLAGS = flags.FLAGS

# flags.DEFINE_string('model_name', 'mlography_segment.hdf5', 'Model name')
flags.DEFINE_string('imp_model_name', 'preprocessed_imgs.hdf5', 'impurities model name')
flags.DEFINE_string('gb_model_name', 'grains_128.hdf5', 'grains boundary name')
flags.DEFINE_string('state', 'use', 'use if model should be used. train if the model should be trained, test if it should be tested')
flags.DEFINE_boolean('keep_training', True, 'True if model should be trained')
flags.DEFINE_boolean('prepare_data', False, 'True if lables should be merged and contours should be generated')
flags.DEFINE_string('base_dir', '/dev/shm', 'Base directory for the process of segmentation')
flags.DEFINE_string('base_dir_final', "/home/matanr/MLography/Segmentation/unet/data", 'Base directory for the segmentation and the binarization after it')
flags.DEFINE_string('in_dir', "/home/matanr/MLography/Segmentation/unet/data/metallography/train/image",
                    'directory that holds input image')
flags.DEFINE_string('in_img', None, 'input image name')
flags.DEFINE_integer('stride', 16, 'stride for segmentation windows')
flags.DEFINE_integer('scale_fac', 3, 'Scale factor for zooming the input image.')


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '¦', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def preprocess_labels(path):
    images = glob.glob(path + "*")
    for img_path in images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(img_path)
        img = cv.imread(img_path)
        image = img.copy()
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        ret, thresh = cv.threshold(gray, 140, 255, cv.THRESH_BINARY)

        cv.imwrite(os.path.join(path,img_name+"_thresh.jpg"), thresh)


def merge_labels(in_path, out_dir):
    images = glob.glob(in_path + "*")
    for img_path in images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        end_num = img_name.find("_")
        if end_num == -1:
            img_num = img_name
        else:
            img_num = img_name[:img_name.find("_")]
        print(img_num)
        img = cv.imread(img_path)
        img[img>0] = 255

        cv.imwrite(os.path.join(out_dir, img_num + ".jpg"), img)


def create_contour_labels(in_path, out_dir):
    images = glob.glob(in_path + "*")
    for img_path in images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(img_path)
        img = cv.imread(img_path)
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR to grayscale

        ret, thresh = cv.threshold(imgray, 200, 255, cv.THRESH_BINARY)

        _, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # epsilon = 0.1 * cv.arcLength(contours, True)
        # approx = cv.approxPolyDP(contours, epsilon, True)
        # cv.drawContours(img, approx, -1, (0, 255, 0), 3)

        # cv.drawContours(img, contours, -1, (0, 255, 0), 3)
        copy_img = img.copy()
        copy_img[True] = 0
        cv.drawContours(copy_img, contours, -1, (255, 255, 255), 2)
        # cv.imshow("Contour", img)
        cv.imwrite(os.path.join(out_dir, img_name + ".jpg"), copy_img)
        
        
def preprocess_grains_masks(in_path, out_dir):
    images = glob.glob(in_path + "*")
    for img_path in images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(img_path)
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR to grayscale

        # kernel3 = np.ones((3,3), np.uint8) 
        
        # img = cv.erode(img, kernel3, iterations=1)
        # img = cv.dilate(img, kernel3, iterations=1)
        
        # img = 255 - img
        
        ret, img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
        cv_algorithms.guo_hall(img, inplace=True)
        
        cv.imwrite(os.path.join(out_dir, img_name + ".png"), img)


def preprocess_images(in_path, out_dir):
    if in_path[-1] == "/":
        images = glob.glob(in_path + "*")
    else:
        images = glob.glob(in_path + "/*")
    length = len(images)
    printProgressBar(0, length, prefix='Preprocessing:', suffix='Complete', length=50)
    for i, img_path in enumerate(images):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv.imread(img_path)

        # -----Converting image to LAB Color model-----------------------------------
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv.createCLAHE(clipLimit=0.8, tileGridSize=(9, 9))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

        gamma = 1.3

        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        img = cv.LUT(img, table)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        # lower mask (0-10)
        lower_red = np.array([0, 10, 60])
        # lower_red = np.array([0, 50, 50])
        upper_red = np.array([15, 255, 255])
        mask0 = cv.inRange(hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([165, 10, 60])
        # lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv.inRange(hsv, lower_red, upper_red)

        # join my masks
        mask = mask0 + mask1
        # mask = mask0
        mask_locations = mask == 255

        s[mask_locations] = np.maximum(s[mask_locations] * 0.8, 0)
        v[mask_locations] = np.minimum(v[mask_locations] * 1.2, 255)

        # define range of black color in HSV
        lower_val = np.array([0, 0, 0])
        upper_val = np.array([179, 255, 127])

        # Threshold the HSV image to get only black colors
        mask = cv.inRange(hsv, lower_val, upper_val)

        img = cv.merge((h, s, v))
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

        img = cv.bilateralFilter(img, 5, 75, 30)

        sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv.GaussianBlur(img, (5, 5), 0)
        img = cv.filter2D(img, -1, sharp_kernel)

        img = cv.medianBlur(img, 3)

        
        cv.imwrite(os.path.join(out_dir, img_name + ".jpg"), img)

        # time.sleep(0.1)
        # Update Progress Bar
        printProgressBar(i + 1, length, prefix='Preprocessing:', suffix='Complete', length=50)


@ray.remote
def preprocess_images_before_seg_single(in_chunk, out_dir):
    for i, img_path in enumerate(in_chunk):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv.imread(img_path)

        # img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # -----Converting image to LAB Color model-----------------------------------
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv.createCLAHE(clipLimit=0.8, tileGridSize=(9, 9))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

        gamma = 1.3

        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        img = cv.LUT(img, table)
        img = cv.bilateralFilter(img, 5, 75, 30)
        img = cv.GaussianBlur(img, (5, 5), 0)

        cv.imwrite(os.path.join(out_dir, img_name + ".jpg"), img)



def preprocess_images_before_seg_parallel(in_path, out_dir):
    start = time.time()
    if in_path[-1] == "/":
        images = glob.glob(in_path + "*")
    else:
        images = glob.glob(in_path + "/*")

    in_chunks = np.array_split(images, num_threads)
    tasks = list()
    for i in range(num_threads):
        tasks.append(preprocess_images_before_seg_single.remote(in_chunks[i], out_dir))
    length = len(tasks)
    printProgressBar(0, length, prefix='Preprocessing:', suffix='Threads Completed', length=50)
    for i in range(num_threads):
        ray.get(tasks[i])
        printProgressBar(i + 1, length, prefix='Preprocessing:', suffix='Threads Completed', length=50)
    end = time.time()
    print("Parallel preprocessing time: " + str(end - start))


def preprocess_images_before_seg(in_path, out_dir):
    if in_path[-1] == "/":
        images = glob.glob(in_path + "*")
    else:
        images = glob.glob(in_path + "/*")
    length = len(images)
    printProgressBar(0, length, prefix='Preprocessing:', suffix='Complete', length=50)
    for i, img_path in enumerate(images):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv.imread(img_path)

        # img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # -----Converting image to LAB Color model-----------------------------------
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv.createCLAHE(clipLimit=0.8, tileGridSize=(9, 9))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

        gamma = 1.3

        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        img = cv.LUT(img, table)
        img = cv.bilateralFilter(img, 5, 75, 30)

        # sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv.GaussianBlur(img, (5, 5), 0)
        # img = cv.filter2D(img, -1, sharp_kernel)

        # img = cv.medianBlur(img, 3)

        cv.imwrite(os.path.join(out_dir, img_name + ".jpg"), img)

        printProgressBar(i + 1, length, prefix='Preprocessing:', suffix='Complete', length=50)


def binarization(in_dir, in_img, out_dir):
    img_path = os.path.join(in_dir, in_img)
    img = cv.imread(img_path, 0)  # pass 0 to convert into gray level
#    ret, thr = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    # img = cv.bilateralFilter(img, 9, 150, 5)
    img = cv.GaussianBlur(img, (3, 3), 0)
    # img = cv.medianBlur(img,5)
    ret3, th3 = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

    whites = np.count_nonzero(th3 > 100)
    print("% of black pixels from the picture: {}".format(100 * whites / th3.size))

    # cv.imshow('win1', thr)
    cv.imwrite(os.path.join(out_dir, in_img), th3)


def edges_binarization(in_dir, in_img, out_dir, without_impurities_dir):
    without_impurities = os.path.join(without_impurities_dir, in_img)
    wi_img = cv.imread(without_impurities)  # pass 0 to convert into gray level
    wi_img = cv.cvtColor(wi_img, cv.COLOR_BGR2RGB)
    
    
    kernel3 = np.ones((3,3), np.uint8) 
    kernel5 = np.ones((5,5), np.uint8) 
    kernel7 = np.ones((7,7), np.uint8) 
    elipse3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    
    img_path = os.path.join(in_dir, in_img)
    seg_img = cv.imread(img_path, 0)  # pass 0 to convert into gray level
    
    
    img = seg_img
    
    ret, img = cv.threshold(img, 55, 255, cv.THRESH_BINARY)
    
    
    # Perform thinning out-of-place
    # guo_hall = cv_algorithms.guo_hall(imgThresh)
    
    # ... or allow the library to modify the original image (= faster):
    # uncomment for thin threshold
    cv_algorithms.guo_hall(img, inplace=True)
    
    # Alternate algorithm (but very similar)
    # Only slight differences in the output!
    # zhang_suen = cv_algorithms.zhang_suen(imgThresh)
    
    
    
    # manual_img_path = "/home/matanr/Pictures/32_grains_cropped.png"
    # manual_img = cv.imread(manual_img_path, 0)  # pass 0 to convert into gray level
    
    
    # img_with_grains_boundary = np.zeros_like(wi_img)
    # img_with_grains_boundary[:,:,0] = img
    # img_with_grains_boundary[:,:,1] = img
    # img_with_grains_boundary[:,:,2] = img
    # black_mask = img_with_grains_boundary == 0
    # img_with_grains_boundary[black_mask] =  wi_img[black_mask] * 0.7
    
    
    # Now, we need to apply threshold, meaning convert uint8 image to boolean.
    mask = img == 255  #Sets TRUE for all 255 valued pixels and FALSE for 0
    
    cv.imwrite(os.path.join(out_dir, in_img), img)
    
    # img_with_grains_boundary = cv.cvtColor(img_with_grains_boundary, cv.COLOR_BGR2RGB)
    # cv.imwrite(os.path.join(out_dir, "masked_"+in_img), img_with_grains_boundary)
    
    
def edges_postprocess(in_dir, in_img, out_binary_dir, out_masked_dir, without_impurities_dir):
    without_impurities = os.path.join(without_impurities_dir, in_img)
    wi_img = cv.imread(without_impurities)
    # wi_img = cv.cvtColor(wi_img, cv.COLOR_BGR2RGB)
    
    gray = cv.imread(os.path.join(in_dir, in_img), 0)
    image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+ cv.THRESH_OTSU)[1]
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
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)
        if area == 0:
            continue
        Areas.append(area)
        n+=1
        # print('Grain particle' ,'of number ' , n , 'has area = ',area)
        ## total_area += area
        cv.drawContours(image,[c],-1,(rand.randint(0,255),rand.randint(0,255),rand.randint(0,255)),-1)
        cv.drawContours(cnts_mask, [c], -1, (255), 2)
    
    # cv.imwrite(os.path.join(out_dir, "contours_" + img_name), cnts_mask)
    # in_image = cv2.imread(os.path.join(wi_dir, img_name))
    cv_algorithms.guo_hall(cnts_mask, inplace=True)
    
    img_with_cnts = np.zeros_like(wi_img)
    img_with_cnts[:,:,0] = cnts_mask
    img_with_cnts[:,:,1] = cnts_mask
    img_with_cnts[:,:,2] = cnts_mask
    black_mask = img_with_cnts == 0
    img_with_cnts[black_mask] =  wi_img[black_mask] * 0.7
    
    cv.imwrite(os.path.join(out_masked_dir, in_img), img_with_cnts)
    cv.imwrite(os.path.join(out_binary_dir, in_img), cnts_mask)


def save_mask(in_dir, in_img, out_dir):
    # black as transparent:
    img_path = os.path.join(in_dir, in_img)

    # Read image in BGR format
    img = cv.imread(img_path)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv.dilate(img, kernel, iterations=3)

    # Coinvert from BGR to BGRA
    bgra = cv.cvtColor(dilation, cv.COLOR_BGR2BGRA)

    tmp = bgra[:, :, 0:3]
    bgra[:, :, 0:3] = np.where(tmp < 255, 0, 255)

    # Slice of alpha channel
    alpha = bgra[:, :, 3]

    # Use logical indexing to set alpha channel to 0 where BGR=0
    alpha[np.all(bgra[:, :, 0:3] == (0, 0, 0), 2)] = 0
    bgra[:, :, 3] = alpha

    pre, ext = os.path.splitext(in_img)
    cv.imwrite(os.path.join(out_dir, pre + ".png"), bgra)


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged


def divide_to_squares(in_dir, in_img, out_dim, stride, out_dir, scale_fac, detailEnhance=False, smooth=False, gray_scale=False):
    img_path = os.path.join(in_dir, in_img)
    if gray_scale:
        img = cv.imread(img_path, 0)  # pass 0 to convert into gray level
    else:
        img = cv.imread(img_path)
    original_img_shape = img.shape
    
    scale_fac_str = str(scale_fac).replace(".", "_")
    
    if smooth:
        for i in range(4):
            img = cv.GaussianBlur(img, (9, 9), 1.5)

    if detailEnhance:
        for i in range(7):
            img = cv.bilateralFilter(img, 5, 50, 7)
            img = cv.detailEnhance(img, sigma_s=0.6, sigma_r=0.15)
        for i in range(4):
            img = cv.detailEnhance(img, sigma_s=0.6, sigma_r=0.15)

    if scale_fac != 1:
        width = int(img.shape[1] * scale_fac)
        height = int(img.shape[0] * scale_fac)
        dim = (width, height)
        img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    shape = img.shape

    imgheight, imgwidth  = shape[0], shape[1]
            
    for i in range(0, imgheight, stride):
        for j in range(0, imgwidth, stride):
            if i+out_dim > imgheight or j+out_dim > imgwidth:
                continue
            square = img[i:i+out_dim, j:j+out_dim]
            # # delete this 2 lines and uncomment all lines below
            # img_wo_ext = os.path.splitext(os.path.basename(in_img))[0]
            # cv.imwrite(os.path.join(out_dir, "{}-{}-{}.png".format(img_wo_ext, i, j)), square)
            cv.imwrite(os.path.join(out_dir, "{}-{}-{}.png".format(scale_fac_str, i, j)), square)
    for i in range(0, imgheight, stride):
        if i+out_dim > imgheight:
                continue
        square = img[i:i+out_dim, imgwidth-out_dim:imgwidth]
        cv.imwrite(os.path.join(out_dir, "{}-{}-{}.png".format(scale_fac_str, i, j)), square)
    for j in range(0, imgwidth, stride):
        if j+out_dim > imgwidth:
                continue
        square = img[imgheight-out_dim:imgheight, j:j+out_dim]
        cv.imwrite(os.path.join(out_dir, "{}-{}-{}.png".format(scale_fac_str, i, j)), square)
        
    square = img[imgheight-out_dim:imgheight, imgwidth-out_dim:imgwidth]
    cv.imwrite(os.path.join(out_dir, "{}-{}-{}.png".format(scale_fac_str, i, j)), square)
    
    return shape, original_img_shape


def divide_to_hierarchy_of_squares(in_dir, in_img, out_dim, stride, out_dir, scales, detailEnhance=False, smooth=False, gray_scale=False):
    shapes = []
    img_path = os.path.join(in_dir, in_img)
    if gray_scale:
        img = cv.imread(img_path, 0)  # pass 0 to convert into gray level
    else:
        img = cv.imread(img_path)
    shape = img.shape
    shapes.append(shape)
    
    for scale_fac in scales:
        shape,_ = divide_to_squares(in_dir, in_img, out_dim, stride, out_dir, scale_fac, detailEnhance, smooth, gray_scale)
        shapes.append(shape)
    return shapes




def union_from_squares(in_dir, original_img_shape, img_shape, square_dim, stride, out_dir, img_name, scale_fac, img_format="jpg", write_img=True, gray_scale=False):
    scale_fac_str = str(scale_fac).replace(".", "_")
    imgheight, imgwidth  = img_shape[0], img_shape[1]
    out_img = np.zeros(shape=img_shape, dtype="float32")
    squares_number_per_pixel = np.zeros(shape=img_shape, dtype="float32")
    for i in range(0, imgheight, stride):
        for j in range(0, imgwidth, stride):
            if i+square_dim > imgheight or j+square_dim > imgwidth:
                continue
            # # # delete these 9 lines and uncomment all lines below
            # img_wo_ext = os.path.splitext(os.path.basename(img_name))[0]
            # square_name = os.path.join(in_dir, "{}-{}-{}.{}".format(img_wo_ext, i, j, img_format))
            # if os.path.exists(square_name):
            #     print ("FOUND {}".format(square_name))
            #     img = cv.imread(square_name)
            #     out_img[i:i + square_dim, j:j + square_dim] = img[:, :]
            # else:
            #     print ("NOT FOUND {}".format(square_name))
            #     out_img[i:i + square_dim, j:j + square_dim] = 255
            
            if gray_scale:
                img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)), 0)
            else:
                img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)))
            cur_sum = out_img[i:i+square_dim, j:j+square_dim] * squares_number_per_pixel[i:i+square_dim, j:j+square_dim]
            cur_sum[:, :] = cur_sum[:, :] + img[:, :]
            squares_number_per_pixel[i:i + square_dim, j:j + square_dim] += 1
            out_img[i:i + square_dim, j:j + square_dim] = cur_sum / squares_number_per_pixel[i:i + square_dim, j:j + square_dim]
    for i in range(0, imgheight, stride):
        if i+square_dim > imgheight:
                continue
        if gray_scale:
            img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)), 0)
        else:
            img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)))
        cur_sum = out_img[i:i+square_dim, imgwidth-square_dim:imgwidth] * squares_number_per_pixel[i:i+square_dim, imgwidth-square_dim:imgwidth]
        cur_sum[:, :] = cur_sum[:, :] + img[:, :]
        squares_number_per_pixel[i:i+square_dim, imgwidth-square_dim:imgwidth] += 1
        out_img[i:i+square_dim, imgwidth-square_dim:imgwidth] = cur_sum / squares_number_per_pixel[i:i+square_dim, imgwidth-square_dim:imgwidth]
    for j in range(0, imgwidth, stride):
        if j+square_dim > imgwidth:
                continue
        if gray_scale:
            img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)), 0)
        else:
            img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)))
        cur_sum = out_img[imgheight-square_dim:imgheight, j:j+square_dim] * squares_number_per_pixel[imgheight-square_dim:imgheight, j:j+square_dim]
        cur_sum[:, :] = cur_sum[:, :] + img[:, :]
        squares_number_per_pixel[imgheight-square_dim:imgheight, j:j+square_dim] += 1
        out_img[imgheight-square_dim:imgheight, j:j+square_dim] = cur_sum / squares_number_per_pixel[imgheight-square_dim:imgheight, j:j+square_dim]
    
    if gray_scale:
        img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)), 0)
    else:
        img = cv.imread(os.path.join(in_dir, "{}-{}-{}.{}".format(scale_fac_str, i, j, img_format)))
    cur_sum = out_img[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth] * squares_number_per_pixel[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth]
    cur_sum[:, :] = cur_sum[:, :] + img[:, :]
    squares_number_per_pixel[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth] += 1
    out_img[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth] = cur_sum / squares_number_per_pixel[imgheight-square_dim:imgheight, imgwidth-square_dim:imgwidth]

    if scale_fac != 1:
        # width = int(out_img.shape[1] / scale_fac)
        # height = int(out_img.shape[0] / scale_fac)
        width = original_img_shape[1]
        height = original_img_shape[0]
        dim = (width, height)
        out_img = cv.resize(out_img, dim, interpolation=cv.INTER_AREA)
    if write_img:
        cv.imwrite(os.path.join(out_dir, img_name), out_img)
    return out_img


def union_from_hierarchy_of_squares(in_dir, img_shapes, square_dim, stride, out_dir, img_name, scales, img_format="jpg", gray_scale=False):
    
    final_img = np.zeros(shape=img_shapes[0], dtype="float32")
    
    weights_sum = 0
    for img_shape, scale_fac in zip(img_shapes[1:], scales):
        out_img = union_from_squares(in_dir, img_shapes[0], img_shape, square_dim, stride, out_dir, img_name, scale_fac, img_format, write_img=False, gray_scale=gray_scale)
        cv.imwrite(os.path.join(out_dir, "{}_{}".format(scale_fac,img_name)), out_img)
        weight = 1/scale_fac
        # print("w: {}".format(weight))
        final_img += (out_img * weight)
        weights_sum += weight

    final_img /= weights_sum
    # print("ws: {}".format(weights_sum))
    cv.imwrite(os.path.join(out_dir, img_name), final_img)


def evaluate_segmentation(test_dir, segmentation_dir, gt_dir, model_name):
    seg_list = []
    gt_mask_list = []
    
    loss_func = binary_focal_loss(alpha=0.2)
    model = load_model(model_name, custom_objects={'binary_focal_loss_fixed': loss_func})
    test_model(model, test_dir + "/*", 128, 128, out_path=segmentation_dir)
    
    seg_files = glob.glob(segmentation_dir + "/*")
    for seg_file in seg_files:
        seg_file_base_name = os.path.basename(seg_file)
        seg_img = cv.imread(os.path.join(segmentation_dir, seg_file_base_name), 0)
        seg_img = cv.GaussianBlur(seg_img, (3, 3), 0)
        cv.imwrite(os.path.join(segmentation_dir, seg_file_base_name), seg_img)
        
        gt_img = cv.imread(os.path.join(gt_dir, seg_file_base_name), 0)
        seg_img = seg_img / 255
        gt_img[gt_img <= 50] = 0
        gt_img[gt_img > 50] = 1
        seg_list.extend(seg_img)
        gt_mask_list.extend(gt_img)
    
    flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
    flatten_score_map_list = np.concatenate(seg_list).ravel()

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
    per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
    print('Pixel ROCAUC: %.3f' % (per_pixel_rocauc))
    plt.figure('ROCAUC: %.3f' % (per_pixel_rocauc))
    plt.plot(fpr, tpr, label='ROCAUC: %.3f' % (per_pixel_rocauc))
    plt.show()
    
    
def impurities_segmentation(base_dir, prep_dir, in_img):
    assert(os.path.exists(FLAGS.imp_model_name))
    loss_func = binary_focal_loss(alpha=0.2)
    model = load_model(FLAGS.imp_model_name, custom_objects={'binary_focal_loss_fixed': loss_func})
    # parallel_model = multi_gpu_model(model, gpus=gpus_num)
    segment_out_base_dir = os.path.join(base_dir, "segmented_squares")
    if not os.path.exists(segment_out_base_dir):
        os.makedirs(segment_out_base_dir)
    segment_out_dir = os.path.join(segment_out_base_dir, in_img)
    if not os.path.exists(segment_out_dir):
        os.makedirs(segment_out_dir)
    test_model(model, prep_dir + "/*", 128, 128, out_path=segment_out_dir)
    # test_model_parallel(parallel_model, prep_dir + "/*", 128, 128, out_path=segment_out_dir, gpus_num=gpus_num)
    
    
def impurities_inpainting(base_dir_final, in_dir, in_img, png_file):
    full_base_dir_final = os.path.abspath(base_dir_final)
    full_in_dir = os.path.abspath(in_dir)
    without_impurities_dir = os.path.join(base_dir_final, "without_impurities")
    if not os.path.exists(without_impurities_dir):
        os.makedirs(without_impurities_dir)
    p = subprocess.Popen(('./generative_inpainting_runner.sh {} {} {}'.
                          format(os.path.join(full_in_dir, in_img), 
                                  os.path.join(full_base_dir_final, "impurities_masks", png_file), 
                                  os.path.join(full_base_dir_final, "without_impurities", png_file))), 
                          shell=True)
    p.wait()


def gb_segmentation(base_dir, png_file, squares_without_impurities_dir):
    squares_without_impurities_edges_base = os.path.join(base_dir, "without_impurities_squares_edges")
    if not os.path.exists(squares_without_impurities_edges_base):
        os.makedirs(squares_without_impurities_edges_base)
    squares_without_impurities_edges = os.path.join(squares_without_impurities_edges_base, png_file)
    if not os.path.exists(squares_without_impurities_edges):
        os.makedirs(squares_without_impurities_edges)
    assert(os.path.exists(FLAGS.gb_model_name))
    loss_func = binary_focal_loss(alpha=0.2)
    model = load_model(FLAGS.gb_model_name, custom_objects={'binary_focal_loss_fixed': loss_func})
    test_model(model, squares_without_impurities_dir + "/*", 128, 128, out_path=squares_without_impurities_edges)


def divide_and_conquer(in_dir, in_img, stride, base_dir, base_dir_final, scale_fac, gpus_num=1):
    filename, _ = os.path.splitext(in_img)
    png_file = filename + ".png"

    if os.path.exists(os.path.join(base_dir_final, "full_segmented_edges_binary", png_file)):
        print("GB segmentation already exist. Skipping.")
        return
    
    # divide to squares for impurities segmentation
    out_base_dir = os.path.join(base_dir, "divided")
    if not os.path.exists(out_base_dir):
        os.makedirs(out_base_dir)
    squares_dir = os.path.join(out_base_dir, in_img)
    if not os.path.exists(squares_dir):
        os.makedirs(squares_dir)
    impurities_scales = [scale_fac]
    in_img_shapes = divide_to_hierarchy_of_squares(in_dir, in_img, 128, stride, squares_dir, impurities_scales)
    print("Divided to squares in: " + squares_dir)

    # preprocess squares for impurities segmentation
    prep_out_base_dir = os.path.join(base_dir, "preprocessed_squares")
    if not os.path.exists(prep_out_base_dir):
        os.makedirs(prep_out_base_dir)
    prep_dir = os.path.join(prep_out_base_dir, in_img)
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)
    preprocess_images_before_seg(squares_dir, prep_dir)
    # ray.init()
    # preprocess_images_before_seg_parallel(squares_dir, prep_dir)
    print("Preprocessed squares in: " + prep_dir)

    # impurities segmentation
    is_p = multiprocessing.Process(target=impurities_segmentation, args=(base_dir, prep_dir, in_img)) 
    is_p.start() 
    is_p.join()
    segment_out_base_dir = os.path.join(base_dir, "segmented_squares")
    segment_out_dir = os.path.join(segment_out_base_dir, in_img)
    print("Impurities Segmented in: " + segment_out_dir)

    # combine impurities segmentation squares to full image
    full_segment_out_dir = os.path.join(base_dir_final, "full_segmented")
    if not os.path.exists(full_segment_out_dir):
        os.makedirs(full_segment_out_dir)
    union_from_hierarchy_of_squares(segment_out_dir, in_img_shapes, 128, stride, full_segment_out_dir, in_img, impurities_scales)
    print("Constructed full segmentation in: " + full_segment_out_dir)

    # apply threshold on the impurities segmentation
    binaryfull_segment_out_dir = os.path.join(base_dir_final, "full_segmented_binary")
    if not os.path.exists(binaryfull_segment_out_dir):
        os.makedirs(binaryfull_segment_out_dir)
    # binarization(full_segment_out_dir, "stride{}_avg_".format(stride) +in_img, binaryfull_segment_out_dir)
    binarization(full_segment_out_dir, in_img, binaryfull_segment_out_dir)
    print("Constructed binarization in: " + binaryfull_segment_out_dir)
    
    # adapt the thresholded image into a mask for impurities inpainting
    imps_mask_dir = os.path.join(base_dir_final, "impurities_masks")
    if not os.path.exists(imps_mask_dir):
        os.makedirs(imps_mask_dir)
    save_mask(binaryfull_segment_out_dir, in_img, imps_mask_dir)
    print("Constructed mask in: " + imps_mask_dir)
    
    # impurities inpainting
    ii_p = multiprocessing.Process(target=impurities_inpainting, args=(base_dir_final, in_dir, in_img, png_file)) 
    ii_p.start() 
    ii_p.join()
    without_impurities_dir = os.path.join(base_dir_final, "without_impurities")
    print("Inpainted impurities in: " + without_impurities_dir)
    
    # divide to squares for grains boundary segmentation
    without_impurities_dir = os.path.join(base_dir_final, "without_impurities")
    squares_without_impurities_dir = os.path.join(base_dir, "without_impurities_squares", png_file)
    if not os.path.exists(squares_without_impurities_dir):
        os.makedirs(squares_without_impurities_dir)
    # scales = [0.7, 0.8, 0.9, 1]
    scales = [0.7, 1, 1.3, 1.5]
    in_img_without_impurities_shapes = divide_to_hierarchy_of_squares(without_impurities_dir, png_file, 128, 16, squares_without_impurities_dir, scales)
    print("Divided to squares wihtout impurities in: " + squares_without_impurities_dir)
    
    # grains boundary segmentation
    gbs_p = multiprocessing.Process(target=gb_segmentation, args=(base_dir, png_file, squares_without_impurities_dir)) 
    gbs_p.start() 
    gbs_p.join()
    squares_without_impurities_edges_base = os.path.join(base_dir, "without_impurities_squares_edges")
    squares_without_impurities_edges = os.path.join(squares_without_impurities_edges_base, png_file)
    print("GB Segmented in: " + squares_without_impurities_edges)

    # combine grains boundary segmentation squares to full image
    full_segment_edges_out_dir = os.path.join(base_dir_final, "full_segmented_edges")
    if not os.path.exists(full_segment_edges_out_dir):
        os.makedirs(full_segment_edges_out_dir)
    union_from_hierarchy_of_squares(squares_without_impurities_edges, in_img_without_impurities_shapes, 128, 16, full_segment_edges_out_dir, png_file, scales, "png")
    print("Constructed full edge segmentation in: " + full_segment_edges_out_dir)

    # apply threshold on the grains boundary segmentation
    binaryfull_segment_edges_out_dir = os.path.join(base_dir_final, "full_segmented_edges_binary")
    if not os.path.exists(binaryfull_segment_edges_out_dir):
        os.makedirs(binaryfull_segment_edges_out_dir)
    edges_binarization(full_segment_edges_out_dir, png_file, binaryfull_segment_edges_out_dir, without_impurities_dir)
    print("Constructed edge binarization in: " + binaryfull_segment_edges_out_dir)
    
    # apply watershed post-process on the binary grains boundary
    postprocess_segment_edges_out_dir = os.path.join(base_dir_final, "post_segmented_edges_binary")
    postprocess_binary_out_dir = os.path.join(postprocess_segment_edges_out_dir, "binary")
    postprocess_masked_out_dir = os.path.join(postprocess_segment_edges_out_dir, "masked")
    if not os.path.exists(postprocess_segment_edges_out_dir):
        os.makedirs(postprocess_segment_edges_out_dir)
    if not os.path.exists(postprocess_binary_out_dir):
        os.makedirs(postprocess_binary_out_dir)
    if not os.path.exists(postprocess_masked_out_dir):
        os.makedirs(postprocess_masked_out_dir)
    edges_postprocess(binaryfull_segment_edges_out_dir, png_file, 
                      postprocess_binary_out_dir, postprocess_masked_out_dir, 
                      without_impurities_dir)
    print("Post-processed edge binarization in: " + postprocess_segment_edges_out_dir)
    return



    # # #### DexiNed for GB segmentation
    # without_impurities_dir = os.path.join(base_dir_final, "without_impurities")
    # squares_without_impurities_dir = os.path.join(base_dir, "without_impurities_squares", in_img)
    # if not os.path.exists(squares_without_impurities_dir):
    #     os.makedirs(squares_without_impurities_dir)
    # # in_img_without_impurities_shape = divide_to_squares(without_impurities_dir, in_img, 128, 16, squares_without_impurities_dir, 0.25, smooth=True)
    # scales = [0.7, 0.8, 0.9, 1]
    # # in_img_without_impurities_shapes = divide_to_hierarchy_of_squares(without_impurities_dir, in_img, 128, 16, squares_without_impurities_dir, scales, detailEnhance=True, smooth=True)
    # in_img_without_impurities_shapes = divide_to_hierarchy_of_squares(without_impurities_dir, in_img, 128, 16, squares_without_impurities_dir, scales)
    # print("Divided to squares wihtout impurities in: " + squares_without_impurities_dir)
    
    # # # without_impurities_prep_out_base_dir = os.path.join(base_dir, "preprocessed_squares_without_impurities")
    # # # if not os.path.exists(without_impurities_prep_out_base_dir):
    # # #     os.makedirs(without_impurities_prep_out_base_dir)
    # # # without_impurities_prep_dir = os.path.join(without_impurities_prep_out_base_dir, in_img)
    # # # if not os.path.exists(without_impurities_prep_dir):
    # # #     os.makedirs(without_impurities_prep_dir)
    # # # preprocess_images_before_seg(squares_without_impurities_dir, without_impurities_prep_dir)
    # # # print("Preprocessed squares in: " + without_impurities_prep_dir)
    
    # squares_without_impurities_edges = os.path.join(base_dir, "without_impurities_squares_edges", in_img)
    # p = subprocess.Popen(('./dexined_runner.sh {} {}'.
    #                       format(squares_without_impurities_dir, 
    #                               squares_without_impurities_edges)), shell=True)
    # # # p = subprocess.Popen(('./dexined_runner.sh {} {}'.
    # # #                       format(without_impurities_prep_dir, 
    # # #                               squares_without_impurities_edges)), shell=True)
    # p.wait()

    # squares_without_impurities_edges_dir = os.path.join(squares_without_impurities_edges, "pred-a")
    # full_segment_edges_out_dir = os.path.join(base_dir_final, "full_segmented_edges")
    # if not os.path.exists(full_segment_edges_out_dir):
    #     os.makedirs(full_segment_edges_out_dir)
    # # # union_from_squares(squares_without_impurities_edges_dir, in_img_without_impurities_shape, 128, 16, full_segment_edges_out_dir, in_img, 0.25, "png")
    # union_from_hierarchy_of_squares(squares_without_impurities_edges_dir, in_img_without_impurities_shapes, 128, 16, full_segment_edges_out_dir, in_img, scales, "png")
    # print("Constructed full edge segmentation in: " + full_segment_edges_out_dir)

    # binaryfull_segment_edges_out_dir = os.path.join(base_dir_final, "full_segmented_edges_binary")
    # if not os.path.exists(binaryfull_segment_edges_out_dir):
    #     os.makedirs(binaryfull_segment_edges_out_dir)
    
    # # plt.figure("input")
    # # img_path = os.path.join(without_impurities_dir, in_img)
    # # wi_img = cv.imread(img_path)
    # # plt_wi_img = cv.cvtColor(wi_img, cv.COLOR_BGR2RGB)
    # # plt.imshow(plt_wi_img)
    
    # # plt.figure("input blurring")
    # # for i in range(7):
    # #     wi_img = cv.bilateralFilter(wi_img, 5, 60, 7)
    # #     wi_img = cv.detailEnhance(wi_img, sigma_s=0.6, sigma_r=0.15)
    # # for i in range(4):
    # #     wi_img = cv.detailEnhance(wi_img, sigma_s=0.6, sigma_r=0.15)
    # # plt_wi_img = cv.cvtColor(wi_img, cv.COLOR_BGR2RGB)
    # # plt.imshow(plt_wi_img)
    
    # # plt.figure("edges")
    # # gray_wi_img = cv.cvtColor(wi_img, cv.COLOR_BGR2GRAY)
    # # edges = cv.Canny(gray_wi_img,30,60)
    # # # edges = cv.Canny(gray_wi_img,10,200)
    # # # edges = auto_canny(gray_wi_img)
    # # plt.imshow(edges, cmap='gray', vmin=0, vmax=255)
    
    
    # edges_binarization(full_segment_edges_out_dir, in_img, binaryfull_segment_edges_out_dir)
    # print("Constructed edge binarization in: " + binaryfull_segment_edges_out_dir)
    # # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # # plt.show()
    # return

    
    # # Just divide
    # out_base_dir = os.path.join(base_dir_final, "squares_128/train/inv_label")
    # if not os.path.exists(out_base_dir):
    #     os.makedirs(out_base_dir)
    # impurities_scales = [1]
    # in_img_shapes = divide_to_hierarchy_of_squares(in_dir, in_img, 128, 128, out_base_dir, impurities_scales)
    # print("Divided to squares in: " + out_base_dir)
    # return
    
    # Just combine labels
    # combined_base_dir = os.path.join(base_dir_final, "combined_labels_256")
    # if not os.path.exists(combined_base_dir):
    #     os.makedirs(combined_base_dir)
    # shapes = []
    # img_path = os.path.join(in_dir, in_img)
    # img = cv.imread(img_path)
    # shapes.append(img.shape)
    # shapes.append(img.shape)
    # # in_img_shapes = divide_to_hierarchy_of_squares(in_dir, in_img, 256, 256, out_base_dir, impurities_scales)
    # labels_dir = os.path.join(base_dir_final, "65_squares", "train", "label")
    # union_from_hierarchy_of_squares(labels_dir, shapes, 256, 256, combined_base_dir, in_img, [1], "png")
    # print("Combined labels in: " + combined_base_dir)
    # return


def main(_):
    if FLAGS.state == 'use':
        if FLAGS.in_img is None:
            dir = FLAGS.in_dir
            if dir[-2:] != "/*":
                dir += "/*"
            imgs = glob.glob(dir)
            for in_img in imgs:
                in_img_base_name = os.path.basename(in_img)
                print(in_img_base_name)
                divide_and_conquer(FLAGS.in_dir, in_img_base_name, FLAGS.stride, FLAGS.base_dir, FLAGS.base_dir_final,
                                   FLAGS.scale_fac,
                                   gpus_num=2)
        else:
            divide_and_conquer(FLAGS.in_dir, FLAGS.in_img, FLAGS.stride, FLAGS.base_dir, FLAGS.base_dir_final, FLAGS.scale_fac,
                               gpus_num=2)
    elif FLAGS.state == 'test':
        evaluate_segmentation('data/squares_128/train/image', 
                              'data/squares_128/train/predictions', 
                              'data/squares_128/train/inv_label', 
                              FLAGS.model_name)
    
    elif FLAGS.state == 'train':
        if FLAGS.prepare_data:
            # merge_labels('data/small/train/dark_label/', 'data/small/train/dark_label_fixed/')
            # create_contour_labels('data/small/train/dark_label_fixed/', 'data/small/train/dark_label_contours/')
            # preprocess_grains_masks('data/80_squares/train/label/', 'data/80_squares/train/inv_label/')
            # preprocess_grains_masks('data/full_segmented_edges/25.jpg', 'data/full_segmented_edges_thin/')
            preprocess_grains_masks('data/squares_128/train/inv_label/', 'data/squares_128/train/inv_label_thin')
            return
        
        # make sure to preprocess the training and testing data
        # in the same way before using unet 
        
        # preprocess_images("data/small/train/image/", "data/small/train/image_preprocess/")
        # preprocess_images("data/small/test/", "data/small/test_preprocess/")
        # return
        
        data_gen_args = dict(rotation_range=0.4,
                             width_shift_range=0.15,  # width_shift_range=0.05,
                             height_shift_range=0.15,  # height_shift_range=0.05,
                             shear_range=0.25, # shear_range=0.05,
                             zoom_range=0.25,  # zoom_range=0.05,
                             horizontal_flip=True,
                             fill_mode='nearest',
                             brightness_range=[0.5, 1.4],
                             zca_whitening=True,
                             rescale=1. / 255)
    
        # impurities
        # myGene = trainGenerator(2, 'data/small/train', 'image_preprocess_cons', 'label_fixed_cons', data_gen_args,
        #                         save_to_dir=None, target_size=(128, 128))
        
    
        # grains
        # myGene = trainGenerator(2, 'data/65_squares/train', 'image', 'inv_label', data_gen_args,
        #                         save_to_dir=None, target_size=(256, 256), image_color_mode='grayscale')
        myGene = trainGenerator(2, 'data/squares_128/train', 'image', 'inv_label', data_gen_args,
                                # image_color_mode='grayscale',
                                save_to_dir=None, target_size=(128, 128))
    
        # focus_param = 2
        # class_weights = np.array([1/ 0.01, 1/ 0.99])
        # loss_func = focal_loss(alpha=1)
        # loss_func = SigmoidFocalCrossEntropy()
        # class_weights = None
        loss_func = binary_focal_loss(alpha=0.2)
        # loss_func = binary_focal_loss(alpha=0.99)
        # loss_func = 'binary_crossentropy'
        if not os.path.exists(FLAGS.model_name):
            # model = unet(loss_func=focal_crossentropy_loss(focus_param=focus_param, class_weights=class_weights))
            # impurities
            model = unet16(input_size=(128,128,3), loss_func=loss_func)
            # model = unet16(input_size=(128,128,1), loss_func=loss_func)
            # grains
            # model = unet16(input_size=(256,256,3), loss_func=loss_func)
        else:
            # model = load_model(FLAGS.model_name)
            model = load_model(FLAGS.model_name, custom_objects={'binary_focal_loss_fixed': loss_func})
    
        if FLAGS.keep_training:
            model_checkpoint = ModelCheckpoint(FLAGS.model_name, monitor='loss', verbose=1, save_best_only=True)
            # model.fit_generator(myGene, steps_per_epoch=300, epochs=100, callbacks=[model_checkpoint])
            model.fit_generator(myGene, steps_per_epoch=300, epochs=300, callbacks=[model_checkpoint])
            # model.fit_generator(myGene, steps_per_epoch=300, epochs=150, callbacks=[model_checkpoint])
    
        print("Finished training")
        # # testGene = testGenerator("data/membrane/test", num_image=30)
        # # results = model.predict_generator(testGene, 30, verbose=1)
        # testGene = testGenerator("data/80_squares/test", 10, target_size = (256, 256))
        # results = model.predict_generator(testGene, 10, verbose=1)
        # saveResult("data/65_squares/test", results)
        
        # this was uncommented
        # test_model(model, "data/small/test_preprocess/*", 128, 128)
    else:
        print("state is not recognized: nothing is done")


if __name__ == "__main__":
   app.run(main)

