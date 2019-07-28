import glob
import numpy as np
import os
from shutil import copyfile


#Train data
def split_to_train_test_val(class_data_path, train_frac=0.6, test_frac=0.2, val_frac=0.2):
    """

    :param class_data_path:
    :param train:
    :param test:
    :param val:
    :return:
    """

    train_path = class_data_path + "/train"
    try:
        os.mkdir(train_path)
    except OSError:
        print("Creation of the directory %s failed" % train_path)
    else:
        print("Successfully created the directory %s " % train_path)

    test_path = class_data_path + "/test"
    try:
        os.mkdir(test_path)
    except OSError:
        print("Creation of the directory %s failed" % test_path)
    else:
        print("Successfully created the directory %s " % test_path)

    val_path = class_data_path + "/validation"
    try:
        os.mkdir(val_path)
    except OSError:
        print("Creation of the directory %s failed" % val_path)
    else:
        print("Successfully created the directory %s " % val_path)

    files = glob.glob(class_data_path + "/*.png")  # normal impurities
    random = np.random.permutation(files)
    num_files = len(files)
    train_files = random[0:int(num_files*train_frac)]
    test_files = random[int(num_files * train_frac): int(num_files * train_frac + num_files * test_frac)]
    val_files = random[-int(num_files*val_frac):]
    for train_file in train_files:
        base = os.path.basename(train_file)
        copyfile(train_file, train_path + "/" + base)
    for test_file in test_files:
        base = os.path.basename(test_file)
        copyfile(test_file, test_path + "/" + base)
    for val_file in val_files:
        base = os.path.basename(val_file)
        copyfile(val_file, val_path + "/" + base)




split_to_train_test_val("./ae_data/data_normal")
split_to_train_test_val("./ae_data/data_anomaly")
