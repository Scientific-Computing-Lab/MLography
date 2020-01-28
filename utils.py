import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import ray
    import numpy as np
    import scipy.spatial.distance as dist
    import math

num_threads = 50


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
    else:  # rectangles intersect
        return 0.


@ray.remote
def find_diameter_single(imp_boxes_chunk, start_index,  imp_boxes):
    max_dist = 0
    for i, imp in enumerate(imp_boxes_chunk):
        global_i = start_index + i
        for other_imp in imp_boxes[global_i+1:]:
            max_dist = max(max_dist, impurity_dist(imp, other_imp))
    return max_dist


def find_diameter(imp_boxes):
    imp_boxes_chunks = np.array_split(imp_boxes, num_threads)
    tasks = list()
    for i in range(num_threads):
        start_index = i * len(imp_boxes_chunks[i-1])  # index offset from the whole array
        tasks.append(find_diameter_single.remote(imp_boxes_chunks[i], start_index, imp_boxes))
    max_dist = 0
    for i in range(num_threads):
        max_dist = max(max_dist, ray.get(tasks[i]))
    return max_dist


def find_diameter_not_parallel(imp_boxes):
    max_dist = 0
    for i, imp in enumerate(imp_boxes):
        for other_imp in imp_boxes[i+1:]:
            max_dist = max(max_dist, impurity_dist(imp, other_imp))
    return max_dist

