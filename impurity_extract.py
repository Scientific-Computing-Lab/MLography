import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import matplotlib.colorbar as clbr
from scipy import ndimage
import scipy.spatial.distance as dist
# from pyod.models.knn import KNN




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
        
def save_boxes(img, markers, impurities_num):
    
    """
    boxes[i-2] := (rmin, rmax, cmin, cmax) of impurity i
    """
#    boxes = [(-1,-1,-1,-1)] * (impurities_num-1)
    boxes = np.zeros((impurities_num-1, 4)) # impurities_num-1 elements, each with 4 features
    
    for impurity in range(2, impurities_num+1):
        indx = markers == impurity
        rmin, rmax, cmin, cmax = bbox(indx)
        boxes[impurity-2,:] = rmin, rmax, cmin, cmax
        
    return boxes



"""
columns = x, rows = y
"""
def impurity_dist(imp1, imp2):
# =============================================================================
#     global imp_boxes
#     rmin1, rmax1, cmin1, cmax1 = imp_boxes[imp1]
#     rmin2, rmax2, cmin2, cmax2 = imp_boxes[imp2]
# =============================================================================
    
    rmin1, rmax1, cmin1, cmax1 = imp1[:]
    rmin2, rmax2, cmin2, cmax2 = imp2[:]
    
    left = cmax2 < cmin1
    right = cmax1 < cmin2
    bottom = rmax2 < rmin1
    top = rmax1 < rmin2
    if top and left:
        return dist.euclidean((cmin1, rmax1), (cmax2, rmin2))
    elif left and bottom:
        return dist.euclidean((cmin1, rmin1), (cmax2, rmax2))
    elif bottom and right:
        return dist.euclidean((cmax1, rmin1), (cmin2, rmax2))
    elif right and top:
        return dist.euclidean((cmax1, rmax1), (cmin2, rmin2))
    elif left:
        return cmin1 - cmax2
    elif right:
        return cmin2 - cmax1
    elif bottom:
        return rmin1 - rmax2
    elif top:
        return rmin2 - rmax1
    else:             # rectangles intersect
        return 0.
    
    
def check_box_dist():
    img = cv.imread('./tags_png_cropped/scan1tag0_cropped.png')
    ret, markers = get_markers(img)
    imp_boxes = save_boxes(img, markers, ret)
    
    # # First subplot
    # plt.figure(1)  # declare the figure
    #
    # plt.subplot(131)  # 221 -> 1 rows, 2 columns, 1st subplot
    # blank_image = np.zeros(img.shape, np.uint8)
    # blank_image[:, :] = (255, 255, 255)
    # blank_image[markers == 7+2] = [0, 0, 255]
    # blank_image[markers == 6+2] = [255, 0, 0]
    # plt.imshow(blank_image)
    # plt.title("dist between 6 and 7 " + str(impurity_dist(imp_boxes[6], imp_boxes[7])))
    #
    # plt.subplot(132)  # 221 -> 1 rows, 2 columns, 2nd subplot
    # blank_image = np.zeros(img.shape, np.uint8)
    # blank_image[:, :] = (255, 255, 255)
    # blank_image[markers == 15+2] = [0, 0, 255]
    # blank_image[markers == 13+2] = [255, 0, 0]
    # plt.imshow(blank_image)
    # plt.title("dist between 15 and 13 " + str(impurity_dist(imp_boxes[15], imp_boxes[13])))
    #
    # plt.subplot(133)  # 221 -> 1 rows, 2 columns, 2nd subplot
    # blank_image = np.zeros(img.shape, np.uint8)
    # blank_image[:, :] = (255, 255, 255)
    # blank_image[markers == 253+2] = [0, 0, 255]
    # blank_image[markers == 940+2] = [255, 0, 0]
    # plt.imshow(blank_image)
    # plt.title("dist between 253 and 940 " + str(impurity_dist(imp_boxes[253], imp_boxes[940])))
    #
    #
    # # Plotting
    # plt.subplots_adjust(hspace=0.4)  # make subplots farther from each other.
    # plt.show()

    k = 15

    impurity_kth_neighbor = kth_nn(imp_boxes, img, markers, k)


def kth_nn(imp_boxes, img, markers, k):
    # data structure that holds for each impurity it's k nearest neighbor
    impurity_kth_neighbor = []
    impurity_kth_neighbor_and_area = []
    for impurity in range(imp_boxes.shape[0]):
        k_nn = [impurity_dist(imp_boxes[impurity], imp_boxes[x]) for x in range(imp_boxes.shape[0]) if x != impurity]
        k_nn.sort()
        impurity_kth_neighbor.append(k_nn[k-1])

        impurity_shape = np.argwhere(markers == impurity + 2)
        imp_area = impurity_shape.shape[0]
        impurity_kth_neighbor_and_area.append(np.square(imp_area) * k_nn[k-1]**2)

    max_val1 = max(impurity_kth_neighbor)
    impurity_kth_neighbor = list(map(lambda x: x / max_val1, impurity_kth_neighbor))

    max_val2 = max(impurity_kth_neighbor_and_area)
    impurity_kth_neighbor_and_area = list(map(lambda x: x / max_val2, impurity_kth_neighbor_and_area))

    # fig = plt.figure(1)
    blank_image1 = np.zeros(img.shape, np.uint8)
    blank_image2 = np.zeros(img.shape, np.uint8)
    blank_image1[:, :] = (255, 255, 255)
    blank_image2[:, :] = (255, 255, 255)
    jet = plt.get_cmap('jet')
    for impurity in range(imp_boxes.shape[0]):

        color1 = jet(impurity_kth_neighbor[impurity])
        # impurity + 2 because first is background, and because it is 1-based and then switched to 0-based.
        blank_image1[markers == impurity + 2] = (color1[0] * 255, color1[1] * 255, color1[2] * 255)

        color2 = jet(impurity_kth_neighbor_and_area[impurity])
        # impurity + 2 because first is background, and because it is 1-based and then switched to 0-based.
        blank_image2[markers == impurity + 2] = (color2[0] * 255, color2[1] * 255, color2[2] * 255)

    plt.subplot(121)
    plt.imshow(blank_image1)
    plt.colorbar()
    plt.clim(0, 1)
    plt.title("impurity colored according to their anomalies, with k = {}".format(k))

    # cmap = cm.jet
    # norm = clrs.Normalize(vmin=5, vmax=10)
    #
    # cb1 = clbr.ColorbarBase(ax, cmap=cmap, norm=norm)
    # cb1.set_label('Some Units')

    plt.subplot(122)
    plt.imshow(blank_image2)
    plt.colorbar()
    plt.clim(0, 1)
    plt.title("anomaly score ^ 2 * impurity_area, with k = {}".format(k))

    plt.show()
    return impurity_kth_neighbor

    
    
def main():
    img = cv.imread('./tags_png_cropped/scan1tag0_cropped.png')
    ret, markers = get_markers(img)
    imp_boxes = save_boxes(img, markers, ret)
    
    # train kNN detector
    # clf_name = 'KNN'
    # clf = KNN(metric=impurity_dist )
    # clf.fit(imp_boxes)
    
    # get the prediction label and outlier scores of the training data
    # y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf.decision_scores_  # raw outlier scores
    
    
    # evaluate and print the results
    print("\nOn Training Data:")

    
    
if __name__== "__main__":
  check_box_dist()
