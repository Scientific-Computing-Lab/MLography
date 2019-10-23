import numpy as np
import scipy.spatial.distance as dist

num_threads = 50


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
