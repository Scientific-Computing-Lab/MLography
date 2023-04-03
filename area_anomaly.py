import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import numpy as np
    import statistics
    from utils import impurity_dist, num_threads, find_diameter
    import ray
    import time
    import json
    import matplotlib.pyplot as plt
    import matplotlib
    import cv2 as cv
    import os
    import gc


class CheapImpCouple:
    def __init__(self, containing_cluster_inside):
        self.cheapest_impurity_outside = None
        self.containing_cluster_outside = None
        self.cheapest_impurity_inside = None
        self.containing_cluster_inside = containing_cluster_inside
        self.lowest_price = np.inf

    def update_cheapest_couple(self, cheap_imp_in, cheap_imp_out, containing_cluster_out, cheap_price):
        if cheap_price < self.lowest_price:
            self.cheapest_impurity_inside = cheap_imp_in
            self.cheapest_impurity_outside = cheap_imp_out
            self.containing_cluster_outside = containing_cluster_out
            self.lowest_price = cheap_price

    def merge_cheapest_couples(self, couples_list):
        for couple in couples_list:
            self.update_cheapest_couple(couple.cheapest_impurity_inside, couple.cheapest_impurity_outside,
                                        couple.containing_cluster_outside, couple.lowest_price)


class MarketClustering:

    def __init__(self, img_shape, indices, markers, imp_boxes, anomaly_scores, k=10):
        self.img_shape = img_shape
        self.indices = indices
        self.markers = markers
        self.imp_boxes = imp_boxes
        self.anomaly_scores = anomaly_scores
        self.k = k
        self.anomaly_clusters = [None] * self.k  # create k clusters
        self.sorted_impurities = []
        self.auction_impurities = {}
        self.init_clusters()

    def init_clusters(self):
        dtype = [('id', int), ('score', float)]
        scores_with_impurity_id = np.array([(i, self.anomaly_scores[i]) for i in range(len(self.anomaly_scores))
                                            if self.anomaly_scores[i] > 0], dtype=dtype)  # ignore impurities with score 0
        sorted_impurities = np.sort(scores_with_impurity_id, order='score')  # order the impurities by their scores
        self.sorted_impurities = [impurity for (impurity, score) in sorted_impurities]

        for cluster in range(self.k):
            imp_id = 1 + cluster
            self.anomaly_clusters[cluster] = {}
            core_impurity = self.sorted_impurities[-imp_id]
            # set the core impurities with highest impurities
            self.anomaly_clusters[cluster]["core_impurities"] = [core_impurity]
            # set initial clusters with highest impurities
            self.anomaly_clusters[cluster]["impurities_inside"] = [core_impurity]
            # set initial wallet for each cluster
            # self.anomaly_clusters[cluster]["wallet"] = (self.anomaly_scores[core_impurity] * 1e4) ** 2.7
            self.anomaly_clusters[cluster]["wallet"] = np.exp(np.sqrt(self.anomaly_scores[core_impurity] * 1e2)) ** 2.8
            # self.anomaly_clusters[cluster]["wallet"] = (self.anomaly_scores[core_impurity] * 1e2) ** 5
            # set initial anomaly score for the cluster. updated only  in update_clusters_scores
            self.anomaly_clusters[cluster]["order_keys"] = []

    def find_containing_cluster(self, impurity):
        """
        Returns the index of the cluster that currently contains the impurity, together with a boolean value that is True if
        the given impurity is a core impurity of that cluster, or False otherwise. Note that there may be only one cluster
        containing each impurity in a given time
        """
        for cluster in self.anomaly_clusters:
            if impurity in cluster["core_impurities"]:
                return cluster, True
            if impurity in cluster["impurities_inside"]:
                return cluster, False
        return -1, False

    def find_cheapest_imp_in_cluster(self, cluster, impurity, is_core_impurity_out):
        """

        :param cluster: cluster in which the cheapest impurity is being searched
        :param impurity: the impurity outside the cluster that searches for cheapest impurity inside the cluster
        :return: the cheapest impurity inside the cluster, and its price
        """
        lowest_price = np.inf
        cheapest_impurity = None
        for impurity_inside in cluster["impurities_inside"]:
            is_core_impurity_inside = True if impurity_inside in cluster["core_impurities"] \
                else False

            distance = impurity_dist(self.imp_boxes[impurity], self.imp_boxes[impurity_inside])
            f = 0.95
            scores_part = (1 - (self.anomaly_scores[impurity] * f) ** 0.5 *
                           (self.anomaly_scores[impurity_inside] * f) ** 0.5) ** 1.6
            distance_part = np.exp(np.sqrt(distance)) ** 1.7
            price = distance_part * scores_part

            # penalty = (2 - np.abs(self.anomaly_scores[impurity] - self.anomaly_scores[impurity_inside])) ** 8
            # price *= penalty

            # if is_core_impurity_out and is_core_impurity_inside:
            #     # discount for cluster combining
            #     discount_part = (1 - (self.anomaly_scores[impurity] * f) ** 0.05 *
            #                      (self.anomaly_scores[impurity_inside] * f) ** 0.05) ** 2
            #     price *= discount_part
            if is_core_impurity_out:
                # discount for cluster combining
                discount_part = (1 - (self.anomaly_scores[impurity] * f) ** 0.05 *
                                 (self.anomaly_scores[impurity_inside] * f) ** 0.05) ** 2.5
                price *= discount_part
                penalty = (2 - np.abs(self.anomaly_scores[impurity] - self.anomaly_scores[impurity_inside])) ** 8
                price *= penalty


            if price < lowest_price:
                #  ignore impurities of bigger bidders
                if impurity not in self.auction_impurities or self.auction_impurities[impurity] < cluster["wallet"]:
                    lowest_price = price
                    cheapest_impurity = impurity_inside
            return cheapest_impurity, lowest_price

    def attempt_to_expand(self, containing_cluster, impurity, cheapest_impurity, lowest_price, cluster):
        """
        Attempts to expand given cluster with the cheapest_impurity
        :param containing_cluster: the containing cluster of the impurity that is being added to the cluster
        :param impurity: the impurity that is being added to the cluster
        :param cheapest_impurity: the cheapest impurity for the impurity in the cluster that is being expanded
        :param lowest_price: the price of the cheapest impurity
        :param cluster: the cluster that is being expanded
        :return: a status code: 0 - nothing has changed (the cluster can't afford addind the cheapest impurity),
        1 - the cluster added the impurity and the impurity is not the core_impurity of the cluster
        2 - the cluster added the impurity and the impurity is the core_impurity of the cluster (both clusters are combined into one)
        """
        if containing_cluster != -1 and impurity in containing_cluster["core_impurities"]:
            self.auction_impurities[impurity] = cluster["wallet"]
            cluster["wallet"] += containing_cluster["wallet"]
            cluster["core_impurities"].extend(containing_cluster["core_impurities"])
            cluster["impurities_inside"].extend(containing_cluster["impurities_inside"])
            self.anomaly_clusters.remove(containing_cluster)
            return 2
        else:
            if cluster["wallet"] >= lowest_price:
                self.auction_impurities[impurity] = cluster["wallet"]
                cluster["wallet"] -= lowest_price
                cluster["impurities_inside"].append(impurity)
                if containing_cluster != -1:
                    containing_cluster["impurities_inside"].remove(impurity)
                return 1
        return 0

    @ray.remote
    def make_clusters_single(self, cluster, impurities_not_in_cluster_chunk):
        cheapest_impurity_couple = CheapImpCouple(cluster)
        for impurity in impurities_not_in_cluster_chunk:
            containing_cluster, is_core_impurity = self.find_containing_cluster(impurity)
            #  calculate prices for all impurities in cluster to all impurities not in cluster,
            #  choose to add best one.

            cheap_impurity_inside, cheap_price_inside = self.find_cheapest_imp_in_cluster(cluster, impurity,
                                                                                          is_core_impurity)
            cheapest_impurity_couple.update_cheapest_couple(cheap_impurity_inside, impurity, containing_cluster,
                                                            cheap_price_inside)
        return cheapest_impurity_couple

    def make_clusters(self):
        start = time.time()
        # converged = False
        status = -1
        while status != 0:
            # converged = True
            status = 0
            self.anomaly_clusters.sort(key=lambda x: x["wallet"], reverse=True)
            for cluster in self.anomaly_clusters:
                if status == 2:  # clusters where combined, need to sort the clusters in the outer loop
                    break
                cheapest_impurity_couple = CheapImpCouple(cluster)
                impurities_not_in_cluster = list(set(list(self.sorted_impurities)) - set(cluster["impurities_inside"]))
                impurities_not_in_cluster_chunks = np.array_split(impurities_not_in_cluster, num_threads)

                tasks = list()
                for i in range(num_threads):
                    tasks.append(self.make_clusters_single.remote(self, cluster, impurities_not_in_cluster_chunks[i]))
                couples_list = list()
                for i in range(num_threads):
                    couples_list.append(ray.get(tasks[i]))

                cheapest_impurity_couple.merge_cheapest_couples(couples_list)

                status = self.attempt_to_expand(
                    cheapest_impurity_couple.containing_cluster_outside,
                    cheapest_impurity_couple.cheapest_impurity_outside,
                    cheapest_impurity_couple.cheapest_impurity_inside,
                    cheapest_impurity_couple.lowest_price,
                    cluster)
        end = time.time()
        print("time make_clusters parallel: " + str(end - start))

    def make_clusters_not_parallel(self):
        # converged = False
        status = -1
        while status != 0:
            # converged = True
            status = 0
            # self.color_clusters()
            self.anomaly_clusters.sort(key=lambda x: x["wallet"], reverse=True)
            for cluster in self.anomaly_clusters:
                if status == 2:   # clusters where combined, need to sort the clusters in the outer loop
                    break
                impurities_not_in_cluster = set(list(self.sorted_impurities)) - set(cluster["impurities_inside"])
                cheapest_impurity_outside = None
                containing_cluster_outside = None
                cheapest_impurity_inside = None
                lowest_price = np.inf
                for impurity in impurities_not_in_cluster:
                    containing_cluster, is_core_impurity = self.find_containing_cluster(impurity)
                    #  calculate prices for all impurities in cluster to all impurities not in cluster,
                    #  choose to add best one.

                    cheap_impurity_inside, lowest_price_inside = self.find_cheapest_imp_in_cluster(cluster, impurity)
                    if lowest_price_inside < lowest_price:
                        cheapest_impurity_inside = cheap_impurity_inside
                        cheapest_impurity_outside = impurity
                        containing_cluster_outside = containing_cluster
                        lowest_price = lowest_price_inside

                status = self.attempt_to_expand(
                    containing_cluster_outside, cheapest_impurity_outside, cheapest_impurity_inside, lowest_price,
                    cluster)

    def update_clusters_score(self, areas=None, imp_boxes=None):
        clusters_order_in_scan = []
        for cluster in self.anomaly_clusters:
            cluster_anomaly_scores = [self.anomaly_scores[i] for i in cluster["impurities_inside"]]

            cluster["order_keys"].append({"name": "median", "score": statistics.median(cluster_anomaly_scores)})
            cluster["order_keys"].append({"name": "mean", "score": statistics.mean(cluster_anomaly_scores)})
            cluster["order_keys"].append({"name": "sum", "score": sum(cluster_anomaly_scores)})
            amount = len(cluster_anomaly_scores)
            cluster["order_keys"].append({"name": "amount", "score": amount})

            if areas is not None:
                areas_inside = [areas[i] for i in cluster["impurities_inside"]]
                cluster["order_keys"].append({"name": "areas_sum", "score": sum(areas_inside)})

            if imp_boxes is not None:
                boxes_inside = [imp_boxes[i] for i in cluster["impurities_inside"]]
                diameter = find_diameter(boxes_inside)
                cluster["order_keys"].append({"name": "diameter", "score": diameter})
                if diameter != 0:
                    cluster["order_keys"].append({"name": "amount_div_diameter", "score": amount / diameter})
                    cluster["order_keys"].append({"name": "amount_mult_diameter", "score": amount * diameter})
                    cluster["order_keys"].append({"name": "sum_div_diameter", "score": sum(cluster_anomaly_scores)
                                                                                       / diameter})
                else:
                    cluster["order_keys"].append({"name": "amount_div_diameter", "score": -1})
                    cluster["order_keys"].append({"name": "amount_mult_diameter", "score": -1})
                    cluster["order_keys"].append({"name": "sum_div_diameter", "score": -1})
                # clusters_order_in_scan.append(amount * diameter)


            if areas is not None and imp_boxes is not None:
                if diameter != 0:
                    cluster["order_keys"].append({"name": "area_sum_div_diameter", "score": sum(areas_inside)/diameter})
                else:
                    cluster["order_keys"].append({"name": "area_sum_div_diameter", "score": -1})
                cluster["order_keys"].append({"name": "area_sum_mult_diameter", "score": sum(areas_inside) * diameter})
                anomaly_areas_scores = [self.anomaly_scores[i] * areas[i] for i in cluster["impurities_inside"]]
                cluster["order_keys"].append({"name": "weighted_area_sum_mult_diameter",
                                              "score": sum(anomaly_areas_scores) * diameter})
                weighted_area_sum_mult_diameter_mult_amount = sum(anomaly_areas_scores) * diameter * amount
                cluster["order_keys"].append({"name": "weighted_area_sum_mult_diameter_mult_amount",
                                              "score": weighted_area_sum_mult_diameter_mult_amount})
                clusters_order_in_scan.append(weighted_area_sum_mult_diameter_mult_amount)
                anomaly_areas_scores = [self.anomaly_scores[i] ** 2 * areas[i] for i in cluster["impurities_inside"]]
                # cluster["order_keys"].append({"name": "weighted2_area_sum_mult_diameter",
                #                               "score": sum(anomaly_areas_scores) * diameter})
                anomaly_areas_scores = [self.anomaly_scores[i] * areas[i] ** 2 for i in cluster["impurities_inside"]]
                cluster["order_keys"].append({"name": "weighted_area2_sum_mult_diameter",
                                              "score": sum(anomaly_areas_scores) * diameter})
                weighted_area2_sum_mult_diameter_mult_amount = sum(anomaly_areas_scores) * diameter * amount
                cluster["order_keys"].append({"name": "weighted_area2_sum_mult_diameter_mult_amount",
                                              "score": weighted_area2_sum_mult_diameter_mult_amount})
                # clusters_order_in_scan.append(weighted_area2_sum_mult_diameter_mult_amount)

                anomaly_areas_scores = [self.anomaly_scores[i] * areas[i] for i in cluster["impurities_inside"]]
                # cluster["order_keys"].append({"name": "weighted2_area2_sum_mult_diameter",
                #                               "score": sum(np.array(anomaly_areas_scores) ** 2) * diameter})
                # cluster["order_keys"].append({"name": "weighted_area_sum2_mult_diameter",
                #                               "score": sum(np.array(anomaly_areas_scores)) ** 2 * diameter})
                # cluster["order_keys"].append({"name": "weighted_area_sum_mult_diameter2",
                #                               "score": sum(np.array(anomaly_areas_scores)) * diameter ** 2})
        indices = np.argsort(clusters_order_in_scan)
        self.anomaly_clusters = [self.anomaly_clusters[indices[i]] for i in range(len(self.anomaly_clusters))]

    def write_clusters_score(self, scan_name, log_path, plots_dir):
        if not os.path.exists(log_path):
            os.mknod(log_path)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        with open(log_path, "r") as json_file:
            try:
                data = json.load(json_file)
            except ValueError:
                data = []
        with open(log_path, "w") as json_file:
            scan_json = {}
            scan_json["scan_name"] = scan_name
            plot_path = plots_dir + "/" + scan_name
            self.color_clusters(show_fig=False, save_plot_path=plot_path)
            scan_json["plot_path"] = plot_path
            scan_json["clusters"] = []
            for cluster_num in range(len(self.anomaly_clusters)):
                # cluster_name = "cluster_" + str(cluster_num)
                # cluster_name = "id_{}_color_{}".format(str(cluster_num),
                #                                        str(round(cluster_num / (len(self.anomaly_clusters) - 1), 2)))
                if (len(self.anomaly_clusters) == 1):
                    cluster_name = "color_{}".format(str(1))
                else:
                    cluster_name = "color_{}".format(str(round(cluster_num / (len(self.anomaly_clusters) - 1), 3)))
                cluster_json = {}
                cluster_json["cluster_name"] = cluster_name
                cluster = self.anomaly_clusters[cluster_num]
                cluster_json["order_keys"] = cluster["order_keys"]
                cluster_json["core_impurities"] = [int(core_imp) for core_imp in cluster["core_impurities"]]
                impurities_and_anomalies = []
                for i in cluster["impurities_inside"]:
                    impurities_and_anomalies.append({"id": int(i), "score": self.anomaly_scores[i]})
                cluster_json["impurities"] = impurities_and_anomalies
                scan_json["clusters"].append(cluster_json)
            data.append(scan_json)
            json.dump(data, json_file)
            json_file.flush()

            pre, ext = os.path.splitext(scan_name)
            self.impurities_pixels_info(plots_dir + "/impurities_info_" + pre + ".npy")

    def impurities_pixels_info(self, impurities_pixels_info_path):
        # 2 values: impurity id, impurity score
        pixels_out = np.full((self.img_shape[0], self.img_shape[1], 2), -1., dtype=float)
        for i in range(len(self.anomaly_scores)):
            imp_id = int(i)
            imp_score = self.anomaly_scores[i]
            argw = np.argwhere(self.markers == imp_id + 2)
            argws = np.split(argw, 2, 1)
            pixels_out[argws[0][:, 0], argws[1][:, 0], :] = [imp_id, imp_score]

        # with open(impurities_pixels_info_path, 'w') as f:
        np.save(impurities_pixels_info_path, pixels_out)



    def color_clusters(self, show_fig=True, save_plot_path=None):
        blank_image = np.zeros(self.img_shape, np.uint8)
        blank_image[:, :] = (255, 255, 255)

        # tab10 = plt.get_cmap('tab10')
        jet = plt.cm.get_cmap('jet', len(self.anomaly_clusters))
        for impurity in self.indices:
            blank_image[self.markers == impurity + 2] = (0, 0, 0)
        for cluster_id, cluster in enumerate(self.anomaly_clusters):
            if len(self.anomaly_clusters) == 1:
                cluster_color = jet(1)
            else:
                cluster_color = jet(cluster_id / (len(self.anomaly_clusters) - 1))
            for impurity in cluster["impurities_inside"]:
                blank_image[self.markers == impurity + 2] = \
                    (cluster_color[0] * 255, cluster_color[1] * 255, cluster_color[2] * 255)
            # print("cluster id: " + str(cluster_id) + ", mean:" + str(cluster["score"]["mean"]) + ", median:" +
            #       str(cluster["score"]["median"]))

        plt.close()
        matplotlib.rcParams.update({'font.size': 22})
        fig = plt.figure("Area anomaly")
        fig.set_size_inches(30, 20)
        img = plt.imshow(blank_image, cmap='jet')
        if len(self.anomaly_clusters) == 1:
            ticks = [0, 1]
            delta = 0.5
        else:
            ticks = list(np.array(range(len(self.anomaly_clusters))) / (len(self.anomaly_clusters) - 1))
            delta = 0.5 * (1 / (len(self.anomaly_clusters) - 1))

        # bounds = ticks
        # bounds = ticks
        # np.append(bounds, 1)
        # plt.colorbar(img, cmap=jet, boundaries=bounds, ticks=ticks)
        plt.colorbar(img, cmap=jet, ticks=ticks)

        # plt.clim(-delta, 1 + delta)
        plt.clim(0, 1)
        plt.title("Area anomaly")

        if show_fig:
            plt.show()
        elif save_plot_path is not None:
            # plt.savefig(save_plot_path, dpi=fig.dpi)
            plt.savefig(save_plot_path)


def create_sub_histogram(histograms_sub_dir, name, scores):
    max_minus_min = np.ptp(scores)
    if max_minus_min != 0:
        normalized_scores = (scores - np.min(scores)) / max_minus_min
    else:
        normalized_scores = np.ones(scores.shape)
    fig = plt.figure(name)
    plt.hist(normalized_scores)
    plt.title(name)
    plt.savefig(histograms_sub_dir + "/" + name + ".png", dpi=fig.dpi)
    plt.close()


def cluster_impurities_info(cluster_json, sorted_clusters_json, clusters_info_path,
                            order_name="weighted_area2_sum_mult_diameter_mult_amount"):

    if os.path.exists(clusters_info_path):
        with open(clusters_info_path, "r") as clusters_info_json_file:
            cluster_info = json.load(clusters_info_json_file)
    else:
        cluster_info = {}

    for order in sorted_clusters_json:
        if order["key_name"] == order_name:
            sorted_clusters_json_by_key = order

    for scan in cluster_json:
        impurities_json = {}
        for cluster in scan['clusters']:
            for impurity in cluster['impurities']:
                impurities_info = {}
                impurities_info['score'] = impurity['score']
                impurities_info['cluster_name'] = cluster['cluster_name']
                for cluster_id, sorted_cluster in enumerate(sorted_clusters_json_by_key['sorted_clusters']):
                    if sorted_cluster['cluster_name'] == cluster['cluster_name']:
                        impurities_info['cluster_num_in_order'] = cluster_id
                        impurities_info['cluster_score'] = sorted_cluster['score']
                        impurities_info['cluster_norm_score'] = sorted_cluster['norm_score']
                        for perc in sorted_clusters_json_by_key["percentiles"]:
                            if (sorted_cluster["norm_score"] >= perc["lower"]) and (
                                    sorted_cluster["norm_score"] < perc["upper"] + np.finfo(float).eps):
                                impurities_info['cluster_perc'] = perc["value"]
                impurities_json[impurity['id']] = impurities_info

            cluster_info[scan['scan_name']] = impurities_json

    with open(clusters_info_path, "w") as clusters_info_file:
        json.dump(cluster_info, clusters_info_file)
        clusters_info_file.flush()

def order_clusters(anomaly_clusters_json_file, ordered_clusters_json_file, order_histograms_path=None, order_keys=None,
                   save_ordered_dir="./logs/area/ordered_clusters", clusters_info_path="./logs/area/clusters_impurities_info.txt"):
    if not os.path.exists(order_histograms_path):
        os.makedirs(order_histograms_path)
    if not os.path.exists(save_ordered_dir):
        os.makedirs(save_ordered_dir)
    sorted_clusters_json = []
    with open(anomaly_clusters_json_file, "r") as anomaly_clusters_json:
        data = json.load(anomaly_clusters_json)
    if len(data) == 0 or len(data[0]["clusters"]) == 0:
        return
    if order_keys is None:
        order_keys = [order_key["name"] for order_key in data[0]["clusters"][0]["order_keys"]]
    for i, order_key in enumerate(order_keys):
        clusters_scores = []
        for scan in data:
            for cluster in scan["clusters"]:
                clusters_scores.append([scan["plot_path"], cluster["cluster_name"],
                                        cluster["order_keys"][i]["score"]])
        # dtype = [('path', str), ('name', str), ('score', float)]
        # clusters_scores = np.array(clusters_scores, dtype=dtype)
        ordered_key = {}
        ordered_key["key_name"] = order_key
        ordered_key["sorted_clusters"] = []
        sorted_clusters = sorted(clusters_scores, key=lambda x: x[2], reverse=True)
        scores_only = np.array(sorted_clusters)[:, 2]
        scores_only = scores_only.astype(np.float)
        # scores_only = (scores_only - np.mean(scores_only)) / np.std(scores_only)
        # scores_only = np.abs(scores_only - np.median(scores_only))
        normalized_scores = (scores_only - np.min(scores_only)) / np.ptp(scores_only)
        for cluster_id, cluster in enumerate(sorted_clusters):
            cluster_json = {}
            cluster_json["path"] = cluster[0]
            cluster_json["cluster_name"] = cluster[1]
            cluster_json["score"] = cluster[2]
            cluster_json["norm_score"] = normalized_scores[cluster_id]
            ordered_key["sorted_clusters"].append(cluster_json)

        # order_histograms + percentiles
        ordered_key["percentiles"] = []

        if order_histograms_path is not None:
            histograms_sub_dir = order_histograms_path+"/"+order_key
            if not os.path.exists(histograms_sub_dir):
                os.makedirs(histograms_sub_dir)
            plt.close()
            order_scores = normalized_scores
            fig = plt.figure(order_key)
            plt.hist(order_scores, log=True)
            plt.title(order_key)
            plt.savefig(histograms_sub_dir+"/all.png", dpi=fig.dpi)
            plt.close()
        lower = 0
        for i in range(1, 11):
            upper = np.percentile(normalized_scores, i * 10 + np.finfo(float).eps)
            sub_arr = normalized_scores[(normalized_scores >= lower) & (normalized_scores < upper)]
            if sub_arr.size > 0:
                # print(sub_arr)
                if order_histograms_path is not None:
                    create_sub_histogram(histograms_sub_dir, "percentile:{}_{}_{}".format(i*10, lower, upper), sub_arr)
            percentile_json = {}
            percentile_json["value"] = "{}%-{}%".format((i-1)*10, i*10)
            percentile_json["lower"] = lower
            percentile_json["upper"] = upper
            ordered_key["percentiles"].append(percentile_json)
            lower = upper

        sorted_clusters_json.append(ordered_key)
    for order in sorted_clusters_json:
        if not os.path.exists(save_ordered_dir + "/" + order["key_name"]):
            # check-point: color and save order keys with no existing directory (in case of OOM errors)
            color_sorted_clusters(order["sorted_clusters"], show_fig=False, save_ordered_dir=save_ordered_dir + "/"
                                                                                             + order["key_name"])
        gc.collect()
    with open(ordered_clusters_json_file, "w") as ordered_json_file:
        json.dump(sorted_clusters_json, ordered_json_file)

    cluster_impurities_info(data, sorted_clusters_json, clusters_info_path,
                            order_name="weighted_area2_sum_mult_diameter_mult_amount")
    return sorted_clusters


@ray.remote
def color_sorted_clusters_single(clusters_to_plot, indices_chunk, show_fig, save_ordered_dir):
    for cluster_id, cluster in zip(indices_chunk, clusters_to_plot):
        bgr_img = cv.imread(cluster["path"])
        img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        plt.imshow(img, cmap='jet')
        cluster_name_id = cluster["cluster_name"][cluster["cluster_name"].find("_")+1:]
        plt.title("#" + str(cluster_id) + ": " + cluster_name_id)
                  # + "\n" + str(cluster["score"]))

        if show_fig:
            plt.show()
        elif save_ordered_dir is not None:
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(30, 20)
            plt.savefig(save_ordered_dir+"/"+str(cluster_id)+".png")
            plt.close()


def color_sorted_clusters(sorted_clusters, top_to_show=50, show_fig=True, save_ordered_dir=None):
    if save_ordered_dir is not None:
        if not os.path.exists(save_ordered_dir):
            os.makedirs(save_ordered_dir)
    plt.close()
    clusters_to_plot = sorted_clusters[:top_to_show]
    indices = range(1, top_to_show+1)
    clusters_to_plot_chunks = np.array_split(clusters_to_plot, num_threads)
    indices_chunks = np.array_split(indices, num_threads)

    tasks = list()
    for i in range(num_threads):
        tasks.append(color_sorted_clusters_single.remote(clusters_to_plot_chunks[i], indices_chunks[i],
                                                         show_fig, save_ordered_dir))
    for i in range(num_threads):
        ray.get(tasks[i])


def color_sorted_clusters_not_parallel(sorted_clusters, top_to_show=150, show_fig=True, save_ordered_dir=None):
    if save_ordered_dir is not None:
        if not os.path.exists(save_ordered_dir):
            os.makedirs(save_ordered_dir)
    plt.close()
    clusters_to_plot = sorted_clusters[:top_to_show]
    for cluster_id, cluster in enumerate(clusters_to_plot, start=1):

        bgr_img = cv.imread(cluster["path"])
        img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        plt.imshow(img, cmap='jet')
        cluster_name_id = cluster["cluster_name"][cluster["cluster_name"].find("_")+1:]
        plt.title("#" + str(cluster_id) + ": " + cluster_name_id)
                  # + "\n" + str(cluster["score"]))

        if show_fig:
            plt.show()
        elif save_ordered_dir is not None:
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(30, 20)
            plt.savefig(save_ordered_dir+"/"+str(cluster_id)+".png")

def clusters_pixels_info(ordered_clusters_json, order_name, scan_file_name, clusters_pixels_info_path):
    input_scan_name = os.path.splitext(os.path.basename(scan_file_name))[0]
    pixels_to_clusters_info = {}
    for order in ordered_clusters_json:
        if order["key_name"] == order_name:
            for cluster_id, cluster in enumerate(order["sorted_clusters"]):
                if os.path.splitext(os.path.basename(cluster["path"]))[0] == input_scan_name:
                    for perc in order["percentiles"]:
                        if (cluster["norm_score"] >= perc["lower"]) and (cluster["norm_score"] < perc["upper"] + np.finfo(float).eps):
                            score = {}
                            score["rank"] = cluster_id
                            score["score"] = cluster["norm_score"]
                            score["range"] = perc["value"]
                            score["name"] = cluster["cluster_name"]
                            for impurity in cluster["impurities_inside"]:
                                pixels_to_clusters_info[impurity] = score

    with open(clusters_pixels_info_path, 'wb') as f:
        json.dump(pixels_to_clusters_info, f)
        f.flush()

def print_clusters_of_img_in_order(ordered_clusters_json_file, order_name, scan_file_name):
    input_scan_name = os.path.splitext(os.path.basename(scan_file_name))[0]
    with open(ordered_clusters_json_file, "r") as ordered_json_file:
        data = json.load(ordered_json_file)
    scores = []
    for order in data:
        if order["key_name"] == order_name:
            for cluster_id, cluster in enumerate(order["sorted_clusters"]):
                if os.path.splitext(os.path.basename(cluster["path"]))[0] == input_scan_name:
                    for perc in order["percentiles"]:
                        if (cluster["norm_score"] >= perc["lower"]) and (cluster["norm_score"] < perc["upper"] + np.finfo(float).eps):
                            print("num in order: {}\nname in scan: {}\nscore: {}\nnormalized score: {}\n"
                                  "percentile range: {}\n".format(cluster_id, cluster["cluster_name"], cluster["score"],
                                                                cluster["norm_score"], perc["value"]))
                            score = {}
                            score["rank"] = cluster_id
                            score["score"] = cluster["norm_score"]
                            score["range"] = perc["value"]
                            score["name"] = cluster["cluster_name"]
                            scores.append(score)
            return scores
