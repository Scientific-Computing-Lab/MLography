import numpy as np
import random
from utils import impurity_dist
import matplotlib.pyplot as plt
import time
from sklearn import neighbors



def divide_impurities_to_clusters_by_anomaly(indices, imp_boxes, anomaly_scores, k=10):
    """
    Based on Condensed-nearest-neighbor
    :param indices:
    :param anomaly_scores:
    :param k:
    :return:
    """
    start = time.time()
    prototype_impurities = list()
    remaining_indices = indices.copy()
    converged = False
    while not converged:
        converged = True
        random_indices = remaining_indices.copy()
        random.shuffle(random_indices)
        for impurity in random_indices:
            nn = [(impurity_dist(imp_boxes[impurity], imp_boxes[x]), x) for x in indices if x != impurity]
            nn.sort()
            nearest_impurity = nn[0][1]
            self_anomaly_class = int(anomaly_scores[impurity] / (1 / k))
            nearest_anomaly_class = int(anomaly_scores[nearest_impurity] / (1 / k))
            if self_anomaly_class != nearest_anomaly_class:
                prototype_impurities.append((impurity, self_anomaly_class))
                remaining_indices.remove(impurity)
                converged = False
    end = time.time()
    print("time divide_impurities_to_clusters_by_anomaly: " + str(end - start))
    return prototype_impurities


def fit_all_pixels_and_color_area_anomaly(img, markers, indices, imp_boxes, prototype_impurities):

    print ("started to fit all pixels according to prototype impurities")
    blank_image_condensed_nn = np.zeros(img.shape, np.uint8)
    blank_image_condensed_nn[:, :] = (255, 255, 255)
    blank_image_condensed_nn[markers == -1] = (0, 0, 0)

    data = list()
    labels = list()
    for (prototype_impurity, anomaly_class) in prototype_impurities:
        impurity_pixels = np.argwhere(markers == prototype_impurity + 2)
        for impurity_pixel in impurity_pixels:
            data.append(impurity_pixel)
            labels.append(anomaly_class / 10)

    knn = neighbors.KNeighborsClassifier()

    # we create an instance of Neighbours Classifier and fit the data.
    knn.fit(data, labels)

    h = .02  # step size in the mesh

    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    pixel_predictions = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    pixel_predictions = pixel_predictions.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    tab10 = plt.get_cmap('tab10')
    plt.set_cmap(tab10)
    plt.pcolormesh(xx, yy, pixel_predictions)



    for (prototype_impurity, anomaly_class) in prototype_impurities:
        color_condensed_nn = tab10(anomaly_class / 10)
        blank_image_condensed_nn[markers == prototype_impurity + 2] = \
            (color_condensed_nn[0] * 255, color_condensed_nn[1] * 255, color_condensed_nn[2] * 255)






    plt.figure("Area anomaly")
    plt.imshow(blank_image_condensed_nn, cmap='tab10')
    plt.colorbar()
    plt.clim(0, 1)
    plt.title("Area anomaly")

    plt.show()
    plt.savefig('colored_area_anomaly.png')