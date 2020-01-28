import glob
import numpy as np
import os
from shutil import copyfile


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def split_to_train_test_val(in_path, train_path, val_path, test_path, train_frac=0.6, test_frac=0.2, val_frac=0.2):


    files = glob.glob(in_path + "/*.png")  # impurities
    random = np.random.permutation(files)
    num_files = len(files)
    train_files = random[0:int(num_files*train_frac)]
    val_files = random[-int(num_files*val_frac):]

    if test_path is not None:
        test_files = random[int(num_files * train_frac): int(num_files * train_frac + num_files * test_frac)]

    for train_file in train_files:
        base = os.path.basename(train_file)
        copyfile(train_file, train_path + "/" + base)
    for val_file in val_files:
        base = os.path.basename(val_file)
        copyfile(val_file, val_path + "/" + base)
    if test_path is not None:
        for test_file in test_files:
            base = os.path.basename(test_file)
            copyfile(test_file, test_path + "/" + base)


def split_to_classes(input_data_path, test_path=None, out_two_classes=None, out_one_class=None,
                            train_frac=0.6, test_frac=0.2, val_frac=0.2):
    if out_one_class is None and out_two_classes is None:
        return
    if out_one_class is not None and out_two_classes is not None:
        print("Nothing has changed: Please choose only one out method! One class / Two classes.")
        return

    if out_one_class is not None:
        create_dir(out_one_class)
        base_path = out_one_class
    else:
        create_dir(out_two_classes)
        base_path = out_two_classes

    train_path = base_path + "/train"
    train_path_normal = train_path + "/normal"
    create_dir(train_path)
    create_dir(train_path_normal)

    val_path = base_path + "/validation"
    val_path_normal = val_path + "/normal"
    create_dir(val_path)
    create_dir(val_path_normal)

    test_path_normal_class = None
    test_path_anomaly_class = None
    if test_path is not None:
        test_path_normal = test_path + "/normal"
        test_path_anomaly = test_path + "/anomaly"
        test_path_normal_class = test_path_normal + "/test"
        test_path_anomaly_class = test_path_anomaly + "/test"
        create_dir(test_path)
        create_dir(test_path_normal)
        create_dir(test_path_anomaly)
        create_dir(test_path_normal_class)
        create_dir(test_path_anomaly_class)

    if out_one_class is not None:
        split_to_train_test_val(in_path=input_data_path+"/normal", train_path=train_path_normal,
                                val_path=val_path_normal, test_path=test_path_normal_class)

    if out_two_classes is not None:
        train_path_anomaly = train_path + "/anomaly"
        create_dir(train_path_anomaly)

        val_path_anomaly = val_path + "/anomaly"
        create_dir(val_path_anomaly)

        split_to_train_test_val(in_path=input_data_path+"/normal", train_path=train_path_normal,
                                val_path=val_path_normal, test_path=test_path_normal_class)
        split_to_train_test_val(in_path=input_data_path+"/anomaly", train_path=train_path_anomaly,
                                val_path=val_path_anomaly, test_path=test_path_anomaly_class)

# split_to_classes(input_data_path="./data/rescaled_extended", test_path="./data/test_rescaled_extended",
#                  out_two_classes="./data/rescaled_extended_2_classes", out_one_class=None)
split_to_classes(input_data_path="./data/rescaled_extended", test_path=None,    # need to create test dir only once
                 out_two_classes=None, out_one_class="./data/rescaled_extended_1_class")
