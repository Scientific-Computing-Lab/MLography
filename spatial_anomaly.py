import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    # import os
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    import scipy.spatial.distance as dist
    from utils import num_threads, impurity_dist
    import ray
    import time


@ray.remote
def weighted_kth_nn_single(imp_boxes, k_list, imp_area, indices, impurities_chunks):
    impurity_neighbors_and_area = {}

    # weighted kth nn calculation
    for k in k_list:
        impurity_neighbors_and_area[k] = np.zeros(len(impurities_chunks))

    for i in range(len(impurities_chunks)):
        impurity = impurities_chunks[i]
        k_nn = [(imp_area[impurity] / imp_area[x]) ** 4 *
                np.maximum(impurity_dist(imp_boxes[impurity], imp_boxes[x]), 0.00001)
                for x in indices if x != impurity]
        # k_nn = [(imp_area[impurity] ** 6 + imp_area[x] ** 6) *
        #         np.maximum(impurity_dist(imp_boxes[impurity], imp_boxes[x]), 0.00001)
        #         for x in indices if x != impurity]
        # k_nn = [
        #         np.maximum(impurity_dist(imp_boxes[impurity], imp_boxes[x]), 0.00001)
        #         for x in indices if x != impurity]
        k_nn.sort()

        for k in k_list:
            # print("i: "+str(i))
            # print("impurity: " + str(impurity))
            impurity_neighbors_and_area[k][i] = imp_area[impurity] * k_nn[k - 1] ** 2
            # impurity_neighbors_and_area[k][i] = k_nn[k - 1] ** 2
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
        data = impurity_neighbors_and_area[k][indices]
        data[data == 0] = 0.00001
        impurity_neighbors_and_area[k][indices] = np.log(data)
        # impurity_neighbors_and_area[k][indices] = np.maximum(np.log(impurity_neighbors_and_area[k][indices]), 0.00001)

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
        k_nn = [(imp_area[impurity] / imp_area[x]) ** 4 * impurity_dist(imp_boxes[impurity], imp_boxes[x])
                for x in indices if x != impurity]
        # k_nn = [impurity_dist(imp_boxes[impurity], imp_boxes[x]) for x in indices if x != impurity]
        k_nn.sort()

        for k in k_list:
            impurity_neighbors_and_area[k][impurity] = imp_area[impurity] * k_nn[k - 1] ** 2
            # impurity_neighbors_and_area[k][impurity] = k_nn[k - 1] ** 2
    print("finished calculating ktn_nn")

    for k in k_list:
        impurity_neighbors_and_area[k][indices] = np.maximum(np.log(impurity_neighbors_and_area[k][indices]), 0.00001)

        scores = impurity_neighbors_and_area[k][indices]
        scores = (scores - np.min(scores)) / np.ptp(scores)
        scores = np.maximum(scores - 2 * np.std(scores), 0.00001)

        impurity_neighbors_and_area[k][indices] = (scores - np.min(scores)) / np.ptp(scores)

        if need_plot:
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