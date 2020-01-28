import numpy as np
import cv2 as cv
from utils import num_threads
import ray

""" not used anymore """
def normalize_circle_boxes(img, markers, imp_boxes, areas, indices, scores, dr_max=300, dc_max=300,
                           write_to_files=False, scan_name="", number_of_impurities_to_write=None, write_circles=True,
                           write_all=True, dest_path="./data/all_regularized_impurities_anomaly/"):
    """
    normalize the impurity images into a fixed size, and standardize the impurities to be in the center.
    :param img: original image
    :param markers: the markers from get_markers function
    :param imp_boxes: the bounding boxes of the impurities
    :param areas: the ares of the impurities
    :param indices: the indices of the significant impurities (the ones with a not too-small size)
    :param scores: the anomaly scores of the impurities. used for writing the score to the name of the file
    :param dr_max: optional, the maximum difference of rows (height) of the impurity that is tolerated
    :param dc_max: optional, the maximum difference of columns (width) of the impurity that is tolerated
    :param write_to_files: True - write the impurities to files, False - do not write.
    :param scan_name: optional - the name of the scan.
    :param number_of_impurities_to_write: optional - maximum impurities allowed to be written
    :param write_circles: True - if writing impurities that are closed to circles is desired.
                          False - for writing anomaly impurities (not closed to circles)
    :param write_all: True only if writing all significant impurities from a specific scan is intended
    :param dest_path: The base destination path of the directory in which the output should be written to
    """
    if dr_max is None or dc_max is None:
        dr_max = 0
        dc_max = 0

        for impurity in indices:
            rmin, rmax, cmin, cmax = imp_boxes[impurity]
            dr = rmax - rmin
            dc = cmax - cmin
            if dr > dr_max:
                dr_max = dr
            if dc > dc_max:
                dc_max = dc

    dr_max = int(dr_max * 2)
    dc_max = int(dc_max * 2)

    too_big_counter = 0

    print("Starting to write normalized impurities")
    normalized = np.zeros(imp_boxes.shape[0])

    number_of_written_impurities = 0
    for impurity in indices:
        # take only circle impurities OR
        # take only non-circle impurities as anomalies OR
        # take all significant impurities
        if (write_circles and scores[impurity] <= 0.2 and areas[impurity] > 50) or \
                (not write_circles and scores[impurity] > 0.6 and areas[impurity] > 50) or write_all:
            rmin, rmax, cmin, cmax = imp_boxes[impurity]
            dr = int(rmax - rmin)
            dc = int(cmax - cmin)
            if 2*dr > dr_max or 2*dc > dc_max:
                # skip too big impurities
                too_big_counter += 1
                continue

            pad_r = int((dr_max - dr) // 2)
            pad_c = int((dc_max - dc) // 2)

            blank_image = np.zeros((dr_max, dc_max, 3), np.uint8)
            blank_image[:, :] = (255, 255, 255)

            image = np.zeros(img.shape, np.uint8)
            image[:, :] = (255, 255, 255)
            image[markers == impurity+2] = img[markers == impurity+2]
            blank_image[pad_r:pad_r+dr, pad_c:pad_c+dc] = image[int(rmin):int(rmax), int(cmin):int(cmax)]
            # normalized[impurity] = blank_image
            if write_to_files:
                string_score = str(scores[impurity])
                string_score.replace('.', '_')
                cv.imwrite(dest_path + string_score +
                           scan_name + "_impurity_" + str(impurity) +".png", blank_image)

            number_of_written_impurities += 1
            if number_of_impurities_to_write is not None and \
                    number_of_written_impurities >= number_of_impurities_to_write:
                return normalized
        print ("too big impurites: " + str(too_big_counter))
    return normalized

@ray.remote
def rescale_and_write_normalized_impurity_single(img, markers, imp_boxes, areas, impurities_chunk,
                                                 scores, height, width,
                                                 proportion_impurity_of_image, scan_name, dest_path_normal,
                                                 dest_path_anomaly, write_all, dest_path_all):
    for i in range(len(impurities_chunk)):
        impurity = impurities_chunk[i]
        # if impurity == 717:
        #     print("in imp 717")
        # take only circle impurities OR
        # take only non-circle impurities as anomalies OR
        # take all significant impurities

        rmin, rmax, cmin, cmax = imp_boxes[impurity]
        dr = int(rmax - rmin)
        dc = int(cmax - cmin)
        blank_image = np.zeros((dr, dc, 3), np.uint8)
        blank_image[:, :] = (255, 255, 255)

        image = np.zeros(img.shape, np.uint8)
        image[:, :] = (255, 255, 255)
        # take only the indices of the impurity
        image[markers == impurity + 2] = img[markers == impurity + 2]
        # take the bounding box of the impurity
        blank_image[:, :] = image[int(rmin):int(rmax), int(cmin):int(cmax)]
        # blank_image = blank_image / 255.0  # conversion for opencv images
        scale_factor_r = height * proportion_impurity_of_image / dr
        scale_factor_c = width * proportion_impurity_of_image / dc
        scale_factor = min(scale_factor_r, scale_factor_c)

        h = int(dr * scale_factor)
        w = int(dc * scale_factor)
        dim = (w, h)

        if h == 0 or w == 0:
            continue

        scaled_image = cv.resize(blank_image, dim)

        normalized_scaled_image = np.zeros((height, width, 3), np.uint8)
        normalized_scaled_image[:, :] = (255, 255, 255)

        pad_r = int((height - h) // 2)
        pad_c = int((width - w) // 2)
        normalized_scaled_image[pad_r:pad_r + h, pad_c:pad_c + w] = scaled_image[:, :]

        string_score = str(scores[impurity])
        string_score.replace('.', '_')
        # normal impurity
        if write_all is False:
            if scores[impurity] <= 0.3 and areas[impurity] > 50:
                cv.imwrite(dest_path_normal + string_score +
                           scan_name + "_impurity_" + str(impurity) + ".png", normalized_scaled_image)
            # anomalous impurity
            elif scores[impurity] > 0.55 and areas[impurity] > 50:
                cv.imwrite(dest_path_anomaly + string_score +
                           scan_name + "_impurity_" + str(impurity) + ".png", normalized_scaled_image)
        else:
            cv.imwrite(dest_path_all + string_score +
                       scan_name + "_impurity_" + str(impurity) + ".png", normalized_scaled_image)


def rescale_and_write_normalized_impurity(img, markers, imp_boxes, areas, indices, scores, height=100, width=100,
                                          proportion_impurity_of_image=0.8,
                                          scan_name="",
                                          dest_path_normal="./data/rescaled/normal/",
                                          dest_path_anomaly="./data/rescaled/anomaly/",
                                          write_all=False,
                                          dest_path_all="./data/rescaled/all/"):
    """
    rescale the impurity images into a fixed size, and standardize the impurities to be in the center.
    :param img: original image
    :param markers: the markers from get_markers function
    :param imp_boxes: the bounding boxes of the impurities
    :param areas: the ares of the impurities
    :param indices: the indices of the significant impurities (the ones with a not too-small size)
    :param scores: the anomaly scores of the impurities. used for writing the score to the name of the file
    :param dr_max: optional, the maximum difference of rows (height) of the impurity that is tolerated
    :param dc_max: optional, the maximum difference of columns (width) of the impurity that is tolerated
    :param write_to_files: True - write the impurities to files, False - do not write.
    :param scan_name: optional - the name of the scan.
    :param number_of_impurities_to_write: optional - maximum impurities allowed to be written
    :param write_circles: True - if writing impurities that are closed to circles is desired.
                          False - for writing anomaly impurities (not closed to circles)
    :param write_all: True only if writing all significant impurities from a specific scan is intended
    :param dest_path: The base destination path of the directory in which the output should be written to
    """

    print("Starting to write normalized impurities of ", scan_name)
    # normalized = np.zeros(imp_boxes.shape[0])

    chunk_size = int(np.ceil(len(indices) / num_threads))
    impurities_chunks = np.array_split(indices, num_threads)

    tasks = list()
    for i in range(num_threads):
        tasks.append(rescale_and_write_normalized_impurity_single.remote(img, markers, imp_boxes, areas,
                                                                         impurities_chunks[i], scores, height, width,
                                                                         proportion_impurity_of_image, scan_name,
                                                                         dest_path_normal, dest_path_anomaly,
                                                                         write_all, dest_path_all))
    for i in range(num_threads):
        ray.get(tasks[i])



def rescale_and_write_normalized_impurity_not_parallel(img, markers, imp_boxes, areas, indices, scores,
                                                       height=100, width=100, proportion_impurity_of_image=0.8,
                                                       scan_name="",
                                                       dest_path_normal="./data/rescaled/normal/",
                                                       dest_path_anomaly="./data/rescaled/anomaly/",
                                                       write_all=False,
                                                       dest_path_all="./data/rescaled/all/"):
    """
    rescale the impurity images into a fixed size, and standardize the impurities to be in the center.
    :param img: original image
    :param markers: the markers from get_markers function
    :param imp_boxes: the bounding boxes of the impurities
    :param areas: the ares of the impurities
    :param indices: the indices of the significant impurities (the ones with a not too-small size)
    :param scores: the anomaly scores of the impurities. used for writing the score to the name of the file
    :param dr_max: optional, the maximum difference of rows (height) of the impurity that is tolerated
    :param dc_max: optional, the maximum difference of columns (width) of the impurity that is tolerated
    :param write_to_files: True - write the impurities to files, False - do not write.
    :param scan_name: optional - the name of the scan.
    :param number_of_impurities_to_write: optional - maximum impurities allowed to be written
    :param write_circles: True - if writing impurities that are closed to circles is desired.
                          False - for writing anomaly impurities (not closed to circles)
    :param write_all: True only if writing all significant impurities from a specific scan is intended
    :param dest_path: The base destination path of the directory in which the output should be written to
    """

    print("Starting to write normalized impurities of ", scan_name)
    # normalized = np.zeros(imp_boxes.shape[0])

    number_of_written_impurities = 0
    for impurity in indices:
        # if impurity == 717:
        #     print("in imp 717")
        # take only circle impurities OR
        # take only non-circle impurities as anomalies OR
        # take all significant impurities

        rmin, rmax, cmin, cmax = imp_boxes[impurity]
        dr = int(rmax - rmin)
        dc = int(cmax - cmin)
        blank_image = np.zeros((dr, dc, 3), np.uint8)
        blank_image[:, :] = (255, 255, 255)

        image = np.zeros(img.shape, np.uint8)
        image[:, :] = (255, 255, 255)
        # take only the indices of the impurity
        image[markers == impurity + 2] = img[markers == impurity + 2]
        # take the bounding box of the impurity
        blank_image[:, :] = image[int(rmin):int(rmax), int(cmin):int(cmax)]
        # blank_image = blank_image / 255.0  # conversion for opencv images
        scale_factor_r = height * proportion_impurity_of_image / dr
        scale_factor_c = width * proportion_impurity_of_image / dc
        scale_factor = min(scale_factor_r, scale_factor_c)

        h = int(dr * scale_factor)
        w = int(dc * scale_factor)
        dim = (w, h)

        if h == 0 or w == 0:
            continue

        scaled_image = cv.resize(blank_image, dim)

        normalized_scaled_image = np.zeros((height, width, 3), np.uint8)
        normalized_scaled_image[:, :] = (255, 255, 255)

        pad_r = int((height - h) // 2)
        pad_c = int((width - w) // 2)
        normalized_scaled_image[pad_r:pad_r + h, pad_c:pad_c + w] = scaled_image[:, :]

        string_score = str(scores[impurity])
        string_score.replace('.', '_')
        # normal impurity
        if write_all is False:
            if scores[impurity] <= 0.3 and areas[impurity] > 50:
                cv.imwrite(dest_path_normal + string_score +
                           scan_name + "_impurity_" + str(impurity) + ".png", normalized_scaled_image)
                number_of_written_impurities += 1
            # anomalous impurity
            elif scores[impurity] > 0.55 and areas[impurity] > 50:
                cv.imwrite(dest_path_anomaly + string_score +
                           scan_name + "_impurity_" + str(impurity) + ".png", normalized_scaled_image)
                number_of_written_impurities += 1
        else:
            cv.imwrite(dest_path_all + string_score +
                       scan_name + "_impurity_" + str(impurity) + ".png", normalized_scaled_image)
