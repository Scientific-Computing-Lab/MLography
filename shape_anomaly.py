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
    from data_preparation import rescale_and_write_normalized_impurity, \
        rescale_and_write_normalized_impurity_not_parallel
    from use_model import predict, predict_not_parallel
    from utils import num_threads, impurity_dist
    import ray
    import time
    from absl import flags
    from absl import app

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
    plt.title("The color is determined by " + r"$\tfrac{(S(circle) - S(impurity))}{S(circle)}$" + " , where circle is the minimal circle "
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