import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    import numpy as np
    import cv2 as cv
    from data_preparation import rescale_and_write_normalized_impurity
    from shape_anomaly import get_circle_impurity_score
    from utils import num_threads
    import ray
    import time
    from absl import app

def get_markers(img, min_threshold, img_name):
    """
    Get the impurities arranged with unique indices from an image (img).
    Applies image processing.
    """
    image = img.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, min_threshold, 255, cv.THRESH_BINARY_INV)

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

    print(img_name + ", number of impurities: " + str(ret))
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
def get_impurity_areas_and_significant_indices_single(markers, impurities_chunks, min_area):
    imp_area = np.zeros(len(impurities_chunks))
    indices = []
    for i in range(len(impurities_chunks)):
        impurity = impurities_chunks[i]
        impurity_shape = np.argwhere(markers == impurity + 2)
        area = impurity_shape.shape[0]
        imp_area[i] = area
        if area > min_area:
            indices.append(impurity)
    return imp_area, indices


def get_impurity_areas_and_significant_indices(imp_boxes, markers, min_area=3):
    start = time.time()
    imp_area = np.zeros(imp_boxes.shape[0])
    indices = []

    impurities_chunks = np.array_split(range(imp_boxes.shape[0]), num_threads)

    tasks = list()
    for i in range(num_threads):
        tasks.append(get_impurity_areas_and_significant_indices_single.remote(markers, impurities_chunks[i], min_area))
    for i in range(num_threads):
        task_out_areas, task_out_indices = ray.get(tasks[i])
        imp_area[impurities_chunks[i]] = task_out_areas[:]   # order is important
        indices.extend(task_out_indices)   # order is not important
    end = time.time()
    print("time get_impurity_areas_and_significant_indices parallel: " + str(end - start))
    return imp_area, indices


def get_impurity_areas_and_significant_indices_not_parallel(imp_boxes, markers, min_area=3):
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
    print("time get_impurity_areas_and_significant_indices not parallel: " + str(end - start))
    return imp_area, indices


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


def extract_impurities(img_path, use_ray, min_threshold=0, black_background=True):
    img = cv.imread(img_path)
    if black_background:
        img = 255 - img
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    ret, markers = get_markers(img, min_threshold, img_name)
    if use_ray:
        imp_boxes = save_boxes(markers, ret)
        areas, indices = get_impurity_areas_and_significant_indices(imp_boxes, markers)
    else:
        imp_boxes = save_boxes_not_parallel(markers, ret)
        areas, indices = get_impurity_areas_and_significant_indices_not_parallel(imp_boxes, markers)
    return img, ret, markers, imp_boxes, areas, indices
