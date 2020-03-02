import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from smallestenclosingcircle import make_circle


def get_circle_impurity_score(markers, imp_boxes, areas, indices):
    scores = np.full(imp_boxes.shape[0], np.infty)
    for impurity in indices:
        impurity_shape = np.argwhere(markers == impurity + 2)
        circle = make_circle(impurity_shape)
        circle_area = np.pi * circle[2] ** 2
        scores[impurity] = (circle_area - areas[impurity]) / circle_area
    return scores


def color_close_to_cirlce(img, markers, indices, scores, areas, save_dir_path):
    blank_image = np.zeros(img.shape, np.uint8)
    blank_image[:, :] = (255, 255, 255)
    jet = plt.get_cmap('jet')

    num_under_thresh = 0

    for impurity in indices:

        # show only under threshold:
        # if scores[impurity] <= 0.3 and areas[impurity] > 50:
        if areas[impurity] > 50:
            num_under_thresh += 1
            color = jet(scores[impurity])
            blank_image[markers == impurity + 2] = (color[0] * 255, color[1] * 255, color[2] * 255)
        else:
            blank_image[markers == impurity + 2] = (0, 0, 0)
    print("under threshold: {}".format(num_under_thresh))

    figure = plt.figure("Colored Circles")
    plt.imshow(blank_image, cmap='jet')
    plt.colorbar()
    plt.clim(0, 1)
    plt.title("The color is determined by " + r"$\frac{(S(circle) - S(impurity))}{S(circle)}$" + " , where circle is the minimal circle "
                                                                             "that covers the impurity")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    # matplotlib.rcParams.update({'font.size': 100})
    figure.set_size_inches(120, 80)
    plt.savefig(save_dir_path + "/" + "circle_area_diff.png")

    plt.show()


def color_circle_diff_all_impurities(img, markers, imp_boxes, areas, indices, save_dir_path):
    scores = get_circle_impurity_score(markers, imp_boxes, areas, indices)
    indx = np.argwhere(areas>50)
    # scores = np.minimum(scores, 1e6)
    # scores = np.maximum(scores, 1e-5)
    # normalized_scores = (scores - np.min(scores)) / np.ptp(scores)
    fig = plt.figure("circle_diff")
    step = 0.02
    refined_bins = np.arange(0, 1 + step, step)
    plt.hist(scores[indx], bins=refined_bins)
    plt.title("circle diff")
    plt.savefig(save_dir_path + "/circle_differences.png" , dpi=fig.dpi)
    color_close_to_cirlce(img, markers, indices, scores, areas, save_dir_path)


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