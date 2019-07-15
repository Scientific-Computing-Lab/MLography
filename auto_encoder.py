import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import matplotlib.colorbar as clbr
from scipy import ndimage
import scipy.spatial.distance as dist
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import evaluate_print
from smallestenclosingcircle import make_circle
from load_data import *







def main():
    load_train("./ae_data/all_regularized_impurities_train_normal/",
               "./ae_data/all_regularized_impurities_train_anomaly/")
    load_test("./ae_data/all_regularized_impurities_test_normal/",
              "./ae_data/all_regularized_impurities_test_anomaly/")


if __name__ == "__main__":
    main()



