import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    import scipy.spatial.distance as dist
    import operator
    from smallestenclosingcircle import make_circle
    from data_preparation import normalize_circle_boxes
    from data_preparation import rescale_and_write_normalized_impurity
    from use_model import predict, predict_not_parallel
    from utils import num_threads, impurity_dist
    import ray
    import time
    from area_anomaly import divide_impurities_to_clusters_by_anomaly, fit_all_pixels_and_color_area_anomaly



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

    sure_fg = opening

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]

    print("number of impurities: " + str(ret))
    return ret, markers




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


@ray.remote
def save_boxes_single(markers, impurities_list):
    boxes = np.zeros((len(impurities_list), 4))  # impurities_num-1 elements, each with 4 features
    for i in range(len(impurities_list)):
        indx = markers == impurities_list[i]
        rmin, rmax, cmin, cmax = bbox(indx)
        boxes[i, :] = rmin, rmax, cmin, cmax
    return boxes



def save_boxes(markers, impurities_num):
    """
    Saves the bounding boxes
    boxes[i-2] := (rmin, rmax, cmin, cmax) of impurity i
    """
    start = time.time()

    boxes = np.zeros((impurities_num - 1, 4))  # impurities_num-1 elements, each with 4 features

    impurities_list = range(2, impurities_num + 1)
    impurities_chunks = np.array_split(impurities_list, num_threads)

    tasks = list()
    for i in range(num_threads):
        tasks.append(save_boxes_single.remote(markers, impurities_chunks[i]))
    for i in range(num_threads):
        task_out = ray.get(tasks[i])
        boxes[impurities_chunks[i] - 2, :] = task_out[:, :]

    end = time.time()
    print("time save_boxes parallel: " + str(end - start))

    return boxes


def save_boxes_not_parallel(markers, impurities_num):
    """
    Saves the bounding boxes
    boxes[i-2] := (rmin, rmax, cmin, cmax) of impurity i
    """
    start = time.time()

    boxes = np.zeros((impurities_num - 1, 4))  # impurities_num-1 elements, each with 4 features

    for impurity in range(2, impurities_num + 1):
        indx = markers == impurity
        rmin, rmax, cmin, cmax = bbox(indx)
        boxes[impurity - 2, :] = rmin, rmax, cmin, cmax

    end = time.time()
    print("time save_boxes not parallel: " + str(end - start))
    return boxes





@ray.remote
def weighted_kth_nn_single(imp_boxes, k_list, imp_area, indices, impurities_chunks):
    impurity_neighbors_and_area = {}

    # weighted kth nn calculation
    for k in k_list:
        impurity_neighbors_and_area[k] = np.zeros(len(impurities_chunks))

    for i in range(len(impurities_chunks)):
        impurity = impurities_chunks[i]
        k_nn = [(imp_area[impurity] / imp_area[x]) ** 2 * impurity_dist(imp_boxes[impurity], imp_boxes[x])
                for x in indices if x != impurity]
        k_nn.sort()

        for k in k_list:
            # print("i: "+str(i))
            # print("impurity: " + str(impurity))
            impurity_neighbors_and_area[k][i] = imp_area[impurity] * k_nn[k - 1] ** 2
    return impurity_neighbors_and_area


def weighted_kth_nn(imp_boxes, img, markers, k_list, imp_area, indices, need_plot=False):
    # data structure that holds for each impurity it's k nearest neighbor
    # it looks like this: first index: the k nearest neighbor (corresponding to k_list), second index is the impurity.
    start = time.time()
    impurity_neighbors_and_area = {}

    for k in k_list:
        impurity_neighbors_and_area[k] = np.zeros(imp_boxes.shape[0])

    # weighted kth nn calculation
    impurities_chunks = np.array_split(indices, num_threads)

    tasks = list()
    for i in range(num_threads):
        tasks.append(weighted_kth_nn_single.remote(imp_boxes, k_list, imp_area, indices, impurities_chunks[i]))
    for i in range(num_threads):
        task_out = ray.get(tasks[i])
        for k in k_list:
            impurity_neighbors_and_area[k][impurities_chunks[i]] = task_out[k][:]
    end = time.time()
    print("time weighted_kth_nn parallel: " + str(end - start))

    for k in k_list:
        impurity_neighbors_and_area[k][indices] = np.maximum(np.log(impurity_neighbors_and_area[k][indices]), 0.00001)

        scores = impurity_neighbors_and_area[k][indices]
        scores = (scores - np.min(scores)) / np.ptp(scores)
        scores = np.maximum(scores - 2 * np.std(scores), 0.00001)

        impurity_neighbors_and_area[k][indices] = (scores - np.min(scores)) / np.ptp(scores)

        # uncomment to see histogram (hope for normal distribution)
        # plt.figure(k)
        # plt.hist(impurity_neighbors_and_area[k][indices])

        max_val2 = max(impurity_neighbors_and_area[k])
        impurity_neighbors_and_area[k] = list(map(lambda x: x / max_val2, impurity_neighbors_and_area[k]))

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

def weighted_kth_nn_not_parallel(imp_boxes, img, markers, k_list, imp_area, indices, need_plot=False):
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

    # plt.show()

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
    start = time.time()
    imp_area = []
    indices = []
    for impurity in range(imp_boxes.shape[0]):
        impurity_shape = np.argwhere(markers == impurity + 2)
        area = impurity_shape.shape[0]
        imp_area.append(area)
        if area > min_area:
            indices.append(impurity)
    end = time.time()
    print("time get_impurity_areas_and_significant_indices: " + str(end - start))
    return imp_area, indices


def get_circle_impurity_score(markers, imp_boxes, areas, indices):
    scores = np.full(imp_boxes.shape[0], np.infty)
    for impurity in indices:
        impurity_shape = np.argwhere(markers == impurity + 2)
        circle = make_circle(impurity_shape)
        circle_area = np.pi * circle[2] ** 2
        scores[impurity] = (circle_area - areas[impurity]) / circle_area
    return scores


def color_close_to_cirlce(img, markers, indices, scores, areas):
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image[:, :] = (255, 255, 255)
    jet = plt.get_cmap('jet')

    num_under_thresh = 0

    for impurity in indices:

        # show only under threshold:
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


def color_shape_anomaly(img, markers, indices, scores, imp_boxes):
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


def color_area_anomaly(img, markers, indices, imp_boxes, prototype_impurities):

    blank_image_condensed_nn = np.zeros(img.shape, np.uint8)
    blank_image_condensed_nn[:, :] = (255, 255, 255)

    impurity_area_scores = np.full(imp_boxes.shape[0], 9 / 10)
    for (prototype_impurity, anomaly_class) in prototype_impurities:
        impurity_area_scores[prototype_impurity] = anomaly_class / 10

    tab10 = plt.get_cmap('tab10')

    for impurity in indices:

        color_condensed_nn = tab10(impurity_area_scores[impurity])
        blank_image_condensed_nn[markers == impurity + 2] = \
            (color_condensed_nn[0] * 255, color_condensed_nn[1] * 255, color_condensed_nn[2] * 255)

    plt.figure("k = 9, Area anomaly")
    plt.imshow(blank_image_condensed_nn, cmap='tab10')
    plt.colorbar()
    plt.clim(0, 1)
    plt.title("k = 9, Area anomaly")

    plt.show()
    plt.savefig('colored_area_anomaly.png')


def color_shape_and_spatial_anomaly(imp_boxes, img, markers, k_list, areas, indices, shape_scores):
    impurity_neighbors_and_area = weighted_kth_nn(imp_boxes, img, markers, k_list, areas, indices)

    blank_image = {}
    blank_image_s = {}
    blank_image_l = {}
    blank_image_condensed_nn = {}

    norm_combined_scores = {}
    impurity_area_scores = {}

    for k in k_list:
        blank_image[k] = np.zeros(img.shape, np.uint8)
        blank_image[k][:, :] = (255, 255, 255)

        blank_image_s[k] = np.zeros(img.shape, np.uint8)
        blank_image_s[k][:, :] = (255, 255, 255)

        blank_image_l[k] = np.zeros(img.shape, np.uint8)
        blank_image_l[k][:, :] = (255, 255, 255)

        blank_image_condensed_nn[k] = np.zeros(img.shape, np.uint8)
        blank_image_condensed_nn[k][:, :] = (255, 255, 255)

        combined_scores = impurity_neighbors_and_area[k][:] * shape_scores[:]
        norm_combined_scores[k] = (combined_scores - np.min(combined_scores)) / np.ptp(combined_scores)
        prototype_impurities = divide_impurities_to_clusters_by_anomaly(indices, imp_boxes, norm_combined_scores[k][:],
                                                                        k=9)
        impurity_area_scores[k] = np.full(imp_boxes.shape[0], 1/9)
        for (prototype_impurity, anomaly_class) in prototype_impurities:
            impurity_area_scores[k][prototype_impurity] = 1/anomaly_class

    jet = plt.get_cmap('jet')
    tab10 = plt.get_cmap('tab10')
    for impurity in indices:
        for k in k_list:
            color = jet(norm_combined_scores[k][impurity])
            blank_image[k][markers == impurity + 2] = (color[0] * 255, color[1] * 255, color[2] * 255)

            color_s = jet(shape_scores[impurity])
            blank_image_s[k][markers == impurity + 2] = (color_s[0] * 255, color_s[1] * 255, color_s[2] * 255)

            color_l = jet(impurity_neighbors_and_area[k][impurity])
            blank_image_l[k][markers == impurity + 2] = (color_l[0] * 255, color_l[1] * 255, color_l[2] * 255)

            color_condensed_nn = tab10(impurity_area_scores[k][impurity])
            blank_image_condensed_nn[k][markers == impurity + 2] = \
                (color_condensed_nn[0] * 255, color_condensed_nn[1] * 255, color_condensed_nn[2] * 255)

    for i in range(len(k_list)):
        plt.figure("k = " + str(k_list[i]) + ", Shape and Spatial anomalies combined")
        plt.imshow(blank_image[k_list[i]], cmap='jet')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title("k = " + str(k_list[i]) + ", Shape and Spatial anomalies combined")

        plt.figure("Shape anomaly")
        plt.imshow(blank_image_s[k_list[i]], cmap='jet')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title("Shape anomaly")

        plt.figure("k = " + str(k_list[i]) + ", Spatial anomaly")
        plt.imshow(blank_image_l[k_list[i]], cmap='jet')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title("k = " + str(k_list[i]) + ", Spatial anomaly")

        plt.figure("k = " + str(k_list[i]) + ", Area anomaly")
        plt.imshow(blank_image_condensed_nn[k_list[i]], cmap='tab10')
        plt.colorbar()
        plt.clim(0, 1)
        plt.title("k = " + str(k_list[i]) + ", Area anomaly")

        # plt.figure("Input")
        # plt.imshow(img)
        # plt.title("Input")

    plt.show()

    cv.imwrite('anomaly_detection.png', blank_image[k_list[0]])
    cv.imwrite('SHAPE_anomaly_detection.png', blank_image_s[k_list[0]])
    cv.imwrite('LOCAL_anomaly_detection.png', blank_image_l[k_list[0]])


    #
    #
    # fig, axs = plt.subplots(2, 2, constrained_layout=True)
    # # plt.colorbar()
    # # plt.clim(0, 1)
    # plt.title("Anomaly Detection")
    # # fig.suptitle('This is a somewhat long figure title', fontsize=16)
    #
    # plt.imshow(img)
    # axs[0].set_title('Input')
    #
    # plt.imshow(blank_image_s[k_list[i]], cmap='jet')
    # axs[1].set_title('Shape Anomaly')
    #
    # plt.imshow(blank_image_l[k_list[i]], cmap='jet')
    # axs[2].set_title('Spatial Anomaly with k = 50')
    #
    # plt.imshow(blank_image[k_list[0]], cmap='jet')
    # axs[3].set_title('Shape with k = 50 and Spatial Anomalies combined')
    #
    # plt.show()


def spatial_anomaly_detection(img_path):
    img = cv.imread(img_path)
    ret, markers = get_markers(img)
    imp_boxes = save_boxes(markers, ret)

    areas, indices = get_impurity_areas_and_significant_indices(imp_boxes, markers)

    # k = [5, 10, 15, 20, 40, 50]
    k = [50]
    weighted_kth_nn(imp_boxes, img, markers, k, areas, indices, need_plot=True)


def area_anomaly_detection(img_path):
    img = cv.imread(img_path)
    ret, markers = get_markers(img)
    imp_boxes = save_boxes(markers, ret)

    areas, indices = get_impurity_areas_and_significant_indices(imp_boxes, markers)

    # k = [5, 10, 15, 20, 40, 50]
    k = [50]
    impurity_neighbors_and_area = weighted_kth_nn(imp_boxes, img, markers, k, areas, indices, need_plot=False)
    prototype_impurities = divide_impurities_to_clusters_by_anomaly(indices, imp_boxes,
                                                                    impurity_neighbors_and_area[50][:], k=2)
    # color_area_anomaly(img, markers, indices, imp_boxes, prototype_impurities)
    fit_all_pixels_and_color_area_anomaly(img, markers, indices, imp_boxes, prototype_impurities)


# change name to calc self anomaly

def shape_and_spatial_anomaly_detection(img_path, dest_path, scan_name="scan1tag-47/", need_to_write=False):
    img = cv.imread(img_path)
    ret, markers = get_markers(img)
    imp_boxes = save_boxes(markers, ret)  # this is the parallel version, there is a non-parallel version too
    areas, indices = get_impurity_areas_and_significant_indices(imp_boxes, markers)

    if need_to_write:
        scores = get_circle_impurity_score(markers, imp_boxes, areas, indices)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # need to move all files into another sub folder for the DirectoryIterator !!!!
        # this is the parallel version, there is a non-parallel version too
        rescale_and_write_normalized_impurity(img, markers, imp_boxes, areas, indices, scores, scan_name=img_name,
                                              write_all=True, dest_path_all=dest_path + scan_name)

    # this is the parallel version, there is a non-parallel version too
    shape_reconstruct_loss = predict(path=dest_path, impurities_num=imp_boxes.shape[0])

    shape_reconstruct_loss = shape_reconstruct_loss - np.min(shape_reconstruct_loss)
    # small impurities are not anomalous, thus the loss is 0
    shape_reconstruct_loss[np.where(np.isinf(shape_reconstruct_loss))] = 0
    norm_reconstruct_loss = shape_reconstruct_loss / np.ptp(shape_reconstruct_loss)

    k_list = [50]
    color_shape_and_spatial_anomaly(imp_boxes, img, markers, k_list, areas, indices, norm_reconstruct_loss)


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
    shape_and_spatial_anomaly_detection(img_path, "./data/test_scan1tag-47/")


if __name__ == "__main__":
    ray.init()

    area_anomaly_detection('./tags_png_cropped/scan1tag-47.png')

    # only weighted_kth_nn
    # spatial_anomaly_detection('./tags_png_cropped/scan1tag-47.png')

    # different examples of shape and spatial anomaly
    # shape_and_spatial_anomaly_detection('./tags_png_cropped/scan1tag-47.png', "./data/test_scan1tag-47/", scan_name="scan1tag-47/",
    #                                     need_to_write=False)
    # shape_anomaly_detection('./tags_png_cropped/scan2tag-39.png', "./data/test_scan2tag-39/", scan_name="scan2tag-39/",
    #                         need_to_write=False)

    # shape_anomaly_detection('./tags_png_cropped/scan3tag-55.png', "./data/test_scan3tag-55/", scan_name="scan3tag-55/",
    #                         need_to_write=False)
    # shape_anomaly_detection('./tags_png_cropped/scan4tag-11.png', "./data/test_scan4tag-11/", scan_name="scan4tag-11/",
    #                         need_to_write=False)

    # prepare all data
    # normalize_all_impurities("./tags_png_cropped/")


