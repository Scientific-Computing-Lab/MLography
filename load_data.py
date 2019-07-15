# two classes (class1, class2)
# only replace with a directory of yours
# finally .npy files saved in your directory with names train.npy, #test.npy, train_labels.npy, test_labels.npy
import cv2
import glob
import numpy as np


#Train data
def load_train(train_normal_path, train_anomaly_path):
    train = []
    train_labels = []
    files = glob.glob (train_normal_path +"/*.png")  # normal impurities
    for myFile in files:
        image = cv2.imread(myFile)
        train.append(image)
        train_labels.append(0)
    files = glob.glob(train_anomaly_path+"/*.png")  # anomaly impurities
    for myFile in files:
        image = cv2.imread(myFile)
        train.append(image)
        train_labels.append(1)
    train = np.array(train, dtype='float32')  # as mnist
    train_labels = np.array(train_labels, dtype='float64')  # as mnist
    # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
    # for example (120 * 40 * 40 * 3)-> (120 * 4800)
    train = np.reshape(train, [train.shape[0], train.shape[1]*train.shape[2]*train.shape[3]])

    # save numpy array as .npy formats
    np.save('train', train)
    np.save('train_labels', train_labels)


#Test data
def load_test(test_normal_path, test_anomaly_path):
    test = []
    test_labels = []
    files = glob.glob(test_normal_path+"/*.png")
    for myFile in files:
        image = cv2.imread(myFile)
        test.append(image)
        test_labels.append(0)  # normal
    files = glob.glob(test_anomaly_path+"/*.png")
    for myFile in files:
        image = cv2.imread(myFile)
        test.append(image)
        test_labels.append(1)  # anomaly

    test = np.array(test, dtype='float32')  # as mnist example
    test_labels = np.array(test_labels, dtype='float64')  # as mnist
    test = np.reshape(test, [test.shape[0], test.shape[1]*test.shape[2]*test.shape[3]])

    # save numpy array as .npy formats
    np.save('test', test)  # saves test.npy
    np.save('test_labels', test_labels)
