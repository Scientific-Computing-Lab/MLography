import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from smallestenclosingcircle import make_circle
from data_preparation import normalize_circle_boxes
from data_preparation import rescale_and_write_normalized_impurity
from use_model import predict


def get_markers(img):
    """
    Get the impurities arranged with unique indices from an image (img).
    Applies image processing.
    """
    image = img.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV)
    
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
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

    print(ret)
    return ret, markers
    
    
def write_impurities(img, markers, impurities_num):
    """
    Writes the impurities such that all the image is blank but the impurity itself
    """
    for impurity in range(2, impurities_num+1):
        blank_image = np.zeros(img.shape, np.uint8)
        blank_image[:, :] = (255, 255, 255)
        blank_image[markers == impurity] = img[markers == impurity]
    
        cv.imwrite("./scan1tag0_cropped_impurities/impurity_" + str(impurity) + 
                   ".png", blank_image)


def bbox(img):
    """
    Get the bounding box of an impurity in an image such that all of it is blank but the impurity itself.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    imp_rows = np.where(rows)[0]
    imp_cols = np.where(cols)[0]
    if len(imp_rows) > 0 and len(imp_cols):
        rmin, rmax = imp_rows[[0, -1]]
        cmin, cmax = imp_cols[[0, -1]]
        cmin -= 1
        rmin -= 1
        cmax += 1
        rmax += 1
    else:
        rmin, rmax, cmin, cmax = 0, 0, 0, 0

    return int(rmin), int(rmax), int(cmin), int(cmax)







def save_boxes(markers, impurities_num):
    """
    Saves the bounding boxes
    boxes[i-2] := (rmin, rmax, cmin, cmax) of impurity i
    """
#    boxes = [(-1,-1,-1,-1)] * (impurities_num-1)
    boxes = np.zeros((impurities_num-1, 4)) # impurities_num-1 elements, each with 4 features
    
    for impurity in range(2, impurities_num+1):
        indx = markers == impurity
        rmin, rmax, cmin, cmax = bbox(indx)
        boxes[impurity-2,:] = rmin, rmax, cmin, cmax
        
    return boxes


def impurity_dist(imp1, imp2):
    """
    Calculates the distance between two bounding boxes of impurities
    columns = x, rows = y
    """
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
    """
    a test for bounding box distance calculation
    """
    img = cv.imread('./tags_png_cropped/scan1tag0_cropped.png')
    ret, markers = get_markers(img)
    imp_boxes = save_boxes(markers, ret)
    
    # First subplot
    plt.figure(1)  # declare the figure

    plt.subplot(131)  # 221 -> 1 rows, 2 columns, 1st subplot
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image[:, :] = (255, 255, 255)
    blank_image[markers == 7+2] = [0, 0, 255]
    blank_image[markers == 6+2] = [255, 0, 0]
    plt.imshow(blank_image)
    plt.title("dist between 6 and 7 " + str(impurity_dist(imp_boxes[6], imp_boxes[7])))

    plt.subplot(132)  # 221 -> 1 rows, 2 columns, 2nd subplot
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image[:, :] = (255, 255, 255)
    blank_image[markers == 15+2] = [0, 0, 255]
    blank_image[markers == 13+2] = [255, 0, 0]
    plt.imshow(blank_image)
    plt.title("dist between 15 and 13 " + str(impurity_dist(imp_boxes[15], imp_boxes[13])))

    plt.subplot(133)  # 221 -> 1 rows, 2 columns, 2nd subplot
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image[:, :] = (255, 255, 255)
    blank_image[markers == 253+2] = [0, 0, 255]
    blank_image[markers == 940+2] = [255, 0, 0]
    plt.imshow(blank_image)
    plt.title("dist between 253 and 940 " + str(impurity_dist(imp_boxes[253], imp_boxes[940])))

    # Plotting
    plt.subplots_adjust(hspace=0.4)  # make subplots farther from each other.
    plt.show()


# !!!!!!! Should not be used. Use weighted_kth_nn instead. !!!!!!!
def kth_nn(imp_boxes, img, markers, k_list):
    # data structure that holds for each impurity it's k nearest neighbor
    # it looks like this: first index: the k nearest neighbor (corresponding to k_list), second index is the impurity.

    impurity_neighbors = {}
    impurity_neighbors_and_area = {}

    for k in k_list:
        impurity_neighbors[k] = []
        impurity_neighbors_and_area[k] = []

    # impurity_kth_neighbor = []
    # impurity_kth_neighbor_and_area = []
    for impurity in range(imp_boxes.shape[0]):
        k_nn = [impurity_dist(imp_boxes[impurity], imp_boxes[x]) for x in range(imp_boxes.shape[0]) if x != impurity]
        k_nn.sort()

        impurity_shape = np.argwhere(markers == impurity + 2)
        imp_area = impurity_shape.shape[0]

        for k in k_list:
            impurity_neighbors[k].append(k_nn[k-1])
        # impurity_kth_neighbor.append(k_nn[k-1])
            impurity_neighbors_and_area[k].append(imp_area ** 2 * k_nn[k - 1] ** 2)
            # impurity_neighbors_and_area[k].append(imp_area * k_nn[k - 1] ** 2)

    print("finished calculating ktn_nn")

        # impurity_kth_neighbor_and_area.append(np.square(imp_area) * k_nn[k-1]**2)

    for k in k_list:
        max_val1 = max(impurity_neighbors[k])
        impurity_neighbors[k] = list(map(lambda x: x / max_val1, impurity_neighbors[k]))

        print("For k={}, Median before normalization: {}".format(k, np.median(impurity_neighbors_and_area[k])))
        print("For k={}, Mean before normalization: {}".format(k, np.mean(impurity_neighbors_and_area[k])))
        max_val2 = max(impurity_neighbors_and_area[k])
        impurity_neighbors_and_area[k] = list(map(lambda x: x / max_val2, impurity_neighbors_and_area[k]))
        print("For k={}, Median after normalization: {}".format(k, np.median(impurity_neighbors_and_area[k])))
        print("For k={}, Mean before normalization: {}".format(k, np.mean(impurity_neighbors_and_area[k])))

    # fig = plt.figure(1)
    blank_image2 = {}

    for k in k_list:
        blank_image1 = np.zeros(img.shape, np.uint8)
        blank_image2[k] = np.zeros(img.shape, np.uint8)
        # blank_image1[:, :] = (255, 255, 255)
        blank_image2[k][:, :] = (255, 255, 255)
    jet = plt.get_cmap('jet')
    for impurity in range(imp_boxes.shape[0]):
        for k in k_list:

            # color1 = jet(impurity_kth_neighbor[impurity])
            # # impurity + 2 because first is background, and because it is 1-based and then switched to 0-based.
            # blank_image1[markers == impurity + 2] = (color1[0] * 255, color1[1] * 255, color1[2] * 255)

            color2 = jet(impurity_neighbors_and_area[k][impurity])
            # impurity + 2 because first is background, and because it is 1-based and then switched to 0-based.
            blank_image2[k][markers == impurity + 2] = (color2[0] * 255, color2[1] * 255, color2[2] * 255)
            # blank_image2[k][markers == impurity + 2] = clrs.hsv_to_rgb((color2[0], color2[1], color2[2]))

    # plt.subplot(121)
    # plt.imshow(blank_image1)
    # plt.colorbar()
    # plt.clim(0, 1)
    # plt.title("impurity colored according to their anomalies, with k = {}".format(k))

    # cmap = cm.jet
    # norm = clrs.Normalize(vmin=5, vmax=10)
    #
    # cb1 = clbr.ColorbarBase(ax, cmap=cmap, norm=norm)
    # cb1.set_label('Some Units')

    # plt.subplot(122)
    for i in range(len(k_list)):
        plt.figure(i)
        plt.imshow(blank_image2[k_list[i]])
        plt.colorbar()
        plt.clim(0, 2)
        plt.title("anomaly score ^ 2 * impurity_area, with k = {}".format(k_list[i]))

    plt.show()
    # return impurity_kth_neighbor


def weighted_kth_nn(imp_boxes, img, markers, k_list, imp_area, indices, need_plot=False):
    # data structure that holds for each impurity it's k nearest neighbor
    # it looks like this: first index: the k nearest neighbor (corresponding to k_list), second index is the impurity.

    impurity_neighbors_and_area = {}

    # weighted kth nn calculation
    for k in k_list:
        impurity_neighbors_and_area[k] = np.zeros(imp_boxes.shape[0])

    for impurity in indices:
        k_nn = [(imp_area[impurity] / imp_area[x]) ** 2 * impurity_dist(imp_boxes[impurity], imp_boxes[x])
                for x in indices if x != impurity]
        k_nn.sort()

        for k in k_list:
            impurity_neighbors_and_area[k][impurity] = imp_area[impurity] * k_nn[k - 1] ** 2
    print("finished calculating ktn_nn")

    for k in k_list:

        impurity_neighbors_and_area[k][indices] = np.maximum(np.log(impurity_neighbors_and_area[k][indices]), 0.00001)

        scores = impurity_neighbors_and_area[k][indices]
        scores = (scores - np.min(scores)) / np.ptp(scores)
        scores = np.maximum(scores - 2 * np.std(scores), 0.00001)

        impurity_neighbors_and_area[k][indices] = (scores - np.min(scores)) / np.ptp(scores)

        plt.figure(k)
        plt.hist(impurity_neighbors_and_area[k][indices])

        max_val2 = max(impurity_neighbors_and_area[k])
        impurity_neighbors_and_area[k] = list(map(lambda x: x / max_val2, impurity_neighbors_and_area[k]))

    # fig = plt.figure(1)
    plt.show()

    if need_plot:
        blank_image2 = {}

        for k in k_list:
            blank_image2[k] = np.zeros(img.shape, np.uint8)
            blank_image2[k][:, :] = (255, 255, 255)
        jet = plt.get_cmap('jet')
        for impurity in indices:
            for k in k_list:
                score = impurity_neighbors_and_area[k][impurity]
                color = jet(score)
                blank_image2[k][markers == impurity + 2] = (color[0] * 255, color[1] * 255, color[2] * 255)

        for i in range(len(k_list)):
            plt.figure(i)
            plt.imshow(blank_image2[k_list[i]], cmap='jet')
            plt.colorbar()
            plt.clim(0, 1)
            plt.title("the kthNN is taken from" + r"$imp$" + " , when the distance to each other impurity" + r"$oth$" +
                      "is calculated in the following manner: " + r"$\log ((\frac{S(imp)}{S(oth)})^2 * box-dist(imp, oth))$"
                      + ", with k = {}".format(k_list[i]))

        plt.show()

    return impurity_neighbors_and_area


def get_impurity_areas_and_significant_indices(imp_boxes, markers, min_area=3):
    imp_area = []
    indices = []
    for impurity in range(imp_boxes.shape[0]):
        impurity_shape = np.argwhere(markers == impurity + 2)
        area = impurity_shape.shape[0]
        imp_area.append(area)
        if area > min_area:
            indices.append(impurity)
    return imp_area, indices


def get_circle_impurity_score(markers, imp_boxes, areas, indices):
    scores = np.full(imp_boxes.shape[0], np.infty)
    for impurity in indices:
        impurity_shape = np.argwhere(markers == impurity + 2)
        circle = make_circle(impurity_shape)
        circle_area = np.pi * circle[2] ** 2
        scores[impurity] = (circle_area - areas[impurity]) / circle_area
    # plt.figure("Circle scores")
    # plt.hist(scores[indices])
    # plt.show()

    return scores


def color_close_to_cirlce(img, markers, indices, scores, areas):
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image[:, :] = (255, 255, 255)
    jet = plt.get_cmap('jet')

    num_under_thresh = 0

    for impurity in indices:
        #color = jet(scores[impurity])
        #blank_image[markers == impurity + 2] = (color[0] * 255, color[1] * 255, color[2] * 255)

        #show only under threshold:
        if scores[impurity] <= 0.3 and areas[impurity] > 50:
            num_under_thresh += 1
            color = jet(scores[impurity])
            blank_image[markers == impurity + 2] = (color[0] * 255, color[1] * 255, color[2] * 255)
        else:
            blank_image[markers == impurity + 2] = (0, 0, 0)
    print("under threshold: {}".format(num_under_thresh))


    plt.figure("Colored Circles")
    plt.imshow(blank_image, cmap='jet')
    plt.colorbar()
    plt.clim(0, 1)
    plt.title("The color is determined by " + r"$S(circle) - S(impurity)$" + " , where circle is the minimal circle "
                                                                             "that covers the impurity")

    plt.show()


def color_shape_anomaly(img, markers, indices, scores):
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image[:, :] = (255, 255, 255)
    jet = plt.get_cmap('jet')

    for impurity in indices:
        color = jet(scores[impurity])
        blank_image[markers == impurity + 2] = (color[0] * 255, color[1] * 255, color[2] * 255)

    plt.figure("Colored shape anomaly")
    plt.imshow(blank_image, cmap='jet')
    plt.colorbar()
    plt.clim(0, 1)
    plt.title("The color is determined by the neural network")

    plt.show()
    plt.savefig('colored_shape_anomaly.png')


def color_shape_and_spatial_anomaly(imp_boxes, img, markers, k_list, areas, indices, shape_scores):
    impurity_neighbors_and_area = weighted_kth_nn(imp_boxes, img, markers, k_list, areas, indices)

    shape_scores_no_zero = shape_scores
    # shape_scores_no_zero[shape_scores_no_zero == 0] = 1

    blank_image = {}
    blank_image_s = {}
    blank_image_l = {}
    norm_combined_scores = {}

    for k in k_list:
        blank_image[k] = np.zeros(img.shape, np.uint8)
        blank_image[k][:, :] = (255, 255, 255)

        blank_image_s[k] = np.zeros(img.shape, np.uint8)
        blank_image_s[k][:, :] = (255, 255, 255)

        blank_image_l[k] = np.zeros(img.shape, np.uint8)
        blank_image_l[k][:, :] = (255, 255, 255)

        combined_scores = impurity_neighbors_and_area[k][:] * shape_scores[:]
        norm_combined_scores[k] = (combined_scores - np.min(combined_scores)) / np.ptp(combined_scores)
    jet = plt.get_cmap('jet')
    for impurity in indices:
        # if areas[impurity] <= 100:
        #     continue
        for k in k_list:
            # score = impurity_neighbors_and_area[k][impurity] * shape_scores[impurity]
            # color = jet(max(score, impurity_neighbors_and_area[k][impurity], shape_scores_no_zero[impurity]))
            color = jet(norm_combined_scores[k][impurity])
            blank_image[k][markers == impurity + 2] = (color[0] * 255, color[1] * 255, color[2] * 255)

            color_s = jet(shape_scores[impurity])
            blank_image_s[k][markers == impurity + 2] = (color_s[0] * 255, color_s[1] * 255, color_s[2] * 255)

            color_l = jet(impurity_neighbors_and_area[k][impurity])
            blank_image_l[k][markers == impurity + 2] = (color_l[0] * 255, color_l[1] * 255, color_l[2] * 255)

    for i in range(len(k_list)):
        plt.figure("k = " + str(k_list[i]) + ", Shape and Spatial anomalies combined")
        plt.imshow(blank_image[k_list[i]], cmap='jet')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title("k = " + str(k_list[i]) + ", Shape and Spatial anomalies combined")

        plt.figure("k = " + str(k_list[i]) + ", Shape anomaly")
        plt.imshow(blank_image_s[k_list[i]], cmap='jet')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title("k = " + str(k_list[i]) + ", Shape anomaly")

        plt.figure("k = " + str(k_list[i]) + ", Spatial anomaly")
        plt.imshow(blank_image_l[k_list[i]], cmap='jet')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title("k = " + str(k_list[i]) + ", Spatial anomaly")

    plt.show()

    cv.imwrite('anomaly_detection.png', blank_image[k_list[0]])
    cv.imwrite('SHAPE_anomaly_detection.png', blank_image_s[k_list[0]])
    cv.imwrite('LOCAL_anomaly_detection.png', blank_image_l[k_list[0]])


def spatial_anomaly_detection(img_path):
    img = cv.imread(img_path)
    ret, markers = get_markers(img)
    imp_boxes = save_boxes(markers, ret)

    areas, indices = get_impurity_areas_and_significant_indices(imp_boxes, markers)

    # k = [5, 10, 15, 20, 40, 50]
    k = [50]
    weighted_kth_nn(imp_boxes, img, markers, k, areas, indices, need_plot=True)


# change name to calc self anomaly

def shape_anomaly_detection(img_path, dest_path, need_to_write=False):
    img = cv.imread(img_path)
    ret, markers = get_markers(img)
    imp_boxes = save_boxes(markers, ret)
    areas, indices = get_impurity_areas_and_significant_indices(imp_boxes, markers)
    #normalized_impurities = normalize_boxes(img, markers, imp_boxes, indices)

    # color_close_to_cirlce(img, markers, indices, scores, areas)
    if need_to_write:
        scores = get_circle_impurity_score(markers, imp_boxes, areas, indices)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        #### need to move all files into another sub folder for the DirectoryIterator !!!!
        rescale_and_write_normalized_impurity(img, markers, imp_boxes, areas, indices, scores, scan_name=img_name,
                                              write_all=True, dest_path_all=dest_path+"scan1tag-47/")
    shape_reconstruct_loss = predict(path=dest_path, impurities_num=imp_boxes.shape[0])
    # small impurities are not anomalous, thus the loss is 0
    shape_reconstruct_loss[np.where(np.isinf(shape_reconstruct_loss))] = 0
    norm_reconstruct_loss = (shape_reconstruct_loss - np.min(shape_reconstruct_loss)) / np.ptp(shape_reconstruct_loss)
    print("normalized impurity 717 loss:"+str(norm_reconstruct_loss[717]))
    # np.savetxt('shape_reconstruct_loss.out', shape_reconstruct_loss, delimiter=',')
    # np.savetxt('norm_reconstruct_loss.out', norm_reconstruct_loss, delimiter=',')
    # plt.figure("normalized reconstruct")
    # plt.hist(norm_reconstruct_loss)
    # plt.show()
    # plt.savefig('norm_reconstruct_loss.png')

    # plt.figure("regular reconstruct")
    # plt.hist(shape_reconstruct_loss)
    # plt.show()
    # plt.savefig('shape_reconstruct_loss.png')

    # color_shape_anomaly(img, markers, indices, norm_reconstruct_loss)
    k_list = [50]
    color_shape_and_spatial_anomaly(imp_boxes, img, markers, k_list, areas, indices, norm_reconstruct_loss)


def find_max_dims_from_dir(dir_path):
    scans_dir = os.listdir(dir_path)

    total_dr_max = 0
    total_dc_max = 0

    for img_path in scans_dir:
        img = cv.imread(dir_path + img_path)
        ret, markers = get_markers(img)
        imp_boxes = save_boxes(markers, ret)
        areas, indices = get_impurity_areas_and_significant_indices(imp_boxes, markers)

        imp_dr_max = 0
        imp_dc_max = 0
        for impurity in indices:
            rmin, rmax, cmin, cmax = imp_boxes[impurity]
            dr = rmax - rmin
            dc = cmax - cmin
            if dr > imp_dr_max:
                imp_dr_max = dr
            if dc > imp_dc_max:
                imp_dc_max = dc
        if img_path != "scan1tag-12.png":  # ignore this scan, because dr_max is too big for some reason
            total_dr_max = max(total_dr_max, imp_dr_max)
            total_dc_max = max(total_dc_max, imp_dc_max)
        print("imp_dr_max= {}, imp_dc_max= {}, total_dr_max= {}, total_dc_max= {} finished: {}".format(imp_dr_max,
                                                                                                       imp_dc_max,
                                                                                                       total_dr_max,
                                                                                                       total_dc_max,
                                                                                                       img_path))

    dr_max = int(total_dr_max * 2)
    dc_max = int(total_dc_max * 2)

    print("dr_max= {}, dc_max= {} (multiplied by 2), finished!".format(dr_max, dc_max))

    # total_dr_max= 3237.0, total_dc_max= 494.0


def normalize_all_impurities(dir_path):

    scans_dir = os.listdir(dir_path)
    for img_path in scans_dir:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv.imread(dir_path + img_path)
        ret, markers = get_markers(img)
        imp_boxes = save_boxes(markers, ret)
        areas, indices = get_impurity_areas_and_significant_indices(imp_boxes, markers)
        scores = get_circle_impurity_score(markers, imp_boxes, areas, indices)
        rescale_and_write_normalized_impurity(img, markers, imp_boxes, areas, indices, scores, scan_name=img_name,
                                              dest_path_normal="./data/rescaled_extended/normal/",
                                              dest_path_anomaly="./data/rescaled_extended/anomaly/"
                                              )



def main(img_path):
    shape_anomaly_detection(img_path, "./data/test_scan1tag-47/")


if __name__ == "__main__":
    # spatial_anomaly_detection('./tags_png_cropped/scan1tag-47.png')

    shape_anomaly_detection('./tags_png_cropped/scan1tag-47.png', "./data/test_scan1tag-47/", need_to_write=False)

    # normalize_all_impurities("./tags_png_cropped/")


