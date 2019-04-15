import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage


def get_markers(img):
    
    image = img.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV)
    
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
    #opening = cv.morphologyEx(opening, cv.MORPH_ERODE, kernel, iterations=1)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    
    # dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    # ret, sure_fg = cv.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
    
    sure_fg = opening
    
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    markers = cv.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]
    
    #plt.imshow(image, cmap='gray')
    #plt.show()
    
    print(ret)
    return ret, markers
    
    
def write_impurities(img, markers, impurities_num):
    for impurity in range(2, impurities_num+1):
        blank_image = np.zeros(img.shape, np.uint8)
        blank_image[:, :] = (255, 255, 255)
        blank_image[markers == impurity] = img[markers == impurity]
    
        cv.imwrite("./scan1tag0_cropped_impurities/impurity_" + str(impurity) + 
                   ".png", blank_image)

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cmin -= 1
    rmin -= 1
    cmax += 1
    rmax += 1

    return rmin, rmax, cmin, cmax


def normalize_boxes(img, markers, impurities_num):
    dr_max = 0
    dc_max = 0

    for impurity in range(2, impurities_num+1):
        indx = markers == impurity
        rmin, rmax, cmin, cmax = bbox(indx)
        dr = rmax - rmin
        dc = cmax - cmin
        if dr > dr_max:
            dr_max = dr
        if dc > dc_max:
            dc_max = dc
            
    dr_max *= 2
    dc_max *= 2
    
    for impurity in range(2, impurities_num+1):
        indx = markers == impurity
        rmin, rmax, cmin, cmax = bbox(indx)
        dr = rmax - rmin
        dc = cmax - cmin
        
        pad_r = (dr_max - dr) // 2
        pad_c = (dc_max - dc) // 2
        
        blank_image = np.zeros((dr_max, dc_max, 3), np.uint8)
        blank_image[:, :] = (255, 255, 255)
        
        image = np.zeros(img.shape, np.uint8)
        image[:, :] = (255, 255, 255)
        image[markers == impurity] = img[markers == impurity]
        blank_image[pad_r:pad_r+dr, pad_c:pad_c+dc] = image[rmin:rmax, cmin:cmax]
# =============================================================================
#         plt.imshow(blank_image, cmap='gray')
#         plt.show()
#         plt.waitforbuttonpress()
# =============================================================================
        cv.imwrite("./scan1tag0_cropped_impurities_reguralized/impurity_" + str(impurity) + 
                   ".png", blank_image)
        
        
    
    # Show image:
# =============================================================================
#     plt.imshow(img, cmap='gray')
#     plt.show()
# =============================================================================
    
img = cv.imread('./tags_png_cropped/scan1tag0_cropped.png')
ret, markers = get_markers(img)
normalize_boxes(img, markers, ret)