import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.models import load_model
    from keras.preprocessing import image
    import matplotlib.pyplot as plt
    import numpy as np
    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing.image import array_to_img
    from keras.preprocessing.image import save_img
    from sklearn.metrics import mean_squared_error
    from PIL import Image
    import os
    import re
    import cv2 as cv
    import ray
    from utils import num_threads


def load_image(img_path, height=100, width=100, show=True):

    img = image.load_img(img_path, target_size=(height, width), color_mode = "grayscale")
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor


def fixed_generator_none(generator):
    """
    """
    for batch in generator:
        yield (batch, batch)


def test_2_impurities(model_name='./model.h5', height=100, width=100):
    model = load_model(model_name)

    normal_imp = "./data/test_rescaled/normal/test/0.151907915704087scan3tag-48_impurity_242.png"

    anomaly_imp = "./data/test_rescaled/anomaly/test/0.624159885095173scan2tag-34_impurity_875.png"

    anomaly_imp2 = "./data/test_rescaled/anomaly/test/0.93742507727708scan1tag-47_impurity_1056.png"

    anomaly_imp717 = "./data/test_scan1tag-47/scan1tag-47/0.8692158152501969scan1tag-47_impurity_717.png"

    model.compile(loss="mean_squared_error",
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # load a single image
    new_image_anomaly = load_image(anomaly_imp, height, width)

    new_image_anomaly2 = load_image(anomaly_imp2, height, width)

    new_image_anomaly717 = load_image(anomaly_imp717, height, width)

    new_image_normal = load_image(normal_imp, height, width)
    print("loaded images")

    # check prediction
    pred_a = model.predict(new_image_anomaly)
    print("predicted anomaly")

    pred_a2 = model.predict(new_image_anomaly2)
    print("predicted line anomaly")

    pred_a717 = model.predict(new_image_anomaly717)
    print("predicted anomaly 717")

    pred_n = model.predict(new_image_normal)
    print("predicted normal")

    # print prediction
    # print('Predicted anomaly:', pred_a)
    # print('Predicted normal:', pred_n)

    save_img('reconstructed_anomaly.jpg', pred_a[0])
    save_img('reconstructed_anomaly_line.jpg', pred_a2[0])
    save_img('reconstructed_anomaly_717.jpg', pred_a717[0])

    save_img('reconstructed_normal.jpg', pred_n[0])
    print("saved images")

    fig, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(2, 4)

    ax11.imshow(cv.imread(normal_imp))
    ax12.imshow(cv.imread(anomaly_imp))
    ax13.imshow(cv.imread(anomaly_imp2))
    ax14.imshow(cv.imread(anomaly_imp717))

    ax21.imshow(np.expand_dims(pred_n[0], axis=0))
    ax22.imshow(np.expand_dims(pred_a[0], axis=0))
    ax23.imshow(np.expand_dims(pred_a2[0], axis=0))
    ax24.imshow(np.expand_dims(pred_a717[0], axis=0))

    plt.show()



    # print("len anomal: " + str(pred_a.shape))
    # print("len normal: " + str(pred_n.shape))
    #
    # # display reconstruction
    # plt.figure(figsize=(1, 2))
    #
    # # display anomaly reconstruction
    # ax = plt.subplot(1, 2, 1)
    # plt.imshow(pred_a[0], cmap='gray')
    #
    # # display normal reconstruction
    # ax = plt.subplot(1, 2, 2)
    # plt.imshow(pred_n[0], cmap='gray')
    # plt.show()
    # plt.savefig('reconstruct.png')


def get_score_from_prediction(input, prediction):
    loss = mean_squared_error(input, prediction)
    return loss ** 4

@ray.remote
def get_scores_single(files_chunk, path, pred_chunk):
    chunk_indices = []
    impurity_anomaly_shape_scores = np.full(len(files_chunk), np.infty)
    for i in range(len(files_chunk)):
        img_name = os.path.splitext(os.path.basename(files_chunk[i]))[0]
        img_name = img_name[img_name.find("_impurity_"):]
        imp_num = int(re.search(r'\d+', img_name).group())

        input_image = load_image(path + files_chunk[i])
        impurity_anomaly_shape_scores[i] = get_score_from_prediction(input_image[0,:,:,0], pred_chunk[i][:,:,0])
        chunk_indices.append(imp_num)
    return chunk_indices, impurity_anomaly_shape_scores


def predict(path, impurities_num, model_name='./model_ae_extended.h5', height=100, width=100, BATCH_SIZE=64):
    model = load_model(model_name)

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1. / 255)
    test_it = datagen.flow_from_directory(path, target_size=(height, width), class_mode=None,
                                          batch_size=BATCH_SIZE,  color_mode='grayscale')
    filenames = test_it.filenames
    samples_num = len(filenames)
    test_it.reset()
    pred = model.predict_generator(fixed_generator_none(test_it), verbose=1, steps=samples_num/BATCH_SIZE)

    pred_chunks = np.array_split(pred, num_threads)
    impurity_anomaly_shape_scores = np.full(impurities_num, np.infty)

    chunk_size = int(np.ceil(len(filenames) / num_threads))
    files_chunks = np.array_split(filenames, num_threads)

    tasks = list()
    for i in range(num_threads):
        tasks.append(get_scores_single.remote(files_chunks[i], path, pred_chunks[i]))
    for i in range(num_threads):
        chunk_indices, task_out = ray.get(tasks[i])
        impurity_anomaly_shape_scores[chunk_indices] = task_out[:]

    return impurity_anomaly_shape_scores


def predict_not_parallel(path, impurities_num, model_name='./model_ae_extended.h5', height=100, width=100, BATCH_SIZE=64):
    model = load_model(model_name)

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1. / 255)
    test_it = datagen.flow_from_directory(path, target_size=(height, width), class_mode=None,
                                          batch_size=BATCH_SIZE,  color_mode='grayscale')
    filenames = test_it.filenames
    samples_num = len(filenames)
    test_it.reset()
    pred = model.predict_generator(fixed_generator_none(test_it), verbose=1, steps=samples_num/BATCH_SIZE)
    # evaluated_loss = model.evaluate_generator(fixed_generator_none(test_it), verbose=1, steps=samples_num/BATCH_SIZE)
    # print("filenames: ", filenames)
    # print(pred)

    # labels = (train_it.class_indices)
    # labels = dict((v, k) for k, v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]

    # print(filenames)
    impurity_anomaly_shape_scores = np.full(impurities_num, np.infty)
    # impurity_anomaly_shape_reconstruct_loss = np.full(impurities_num, np.infty)

    for i in range(len(filenames)):
        img_name = os.path.splitext(os.path.basename(filenames[i]))[0]
        img_name = img_name[img_name.find("_impurity_"):]
        imp_num = int(re.search(r'\d+', img_name).group())

        # print("input path: " + path + filenames[i])
        input_image = load_image(path + filenames[i])
        # print("input dim:" + str(input_image.shape))
        # print("pred dim:" + str(pred[i].shape))
        impurity_anomaly_shape_scores[imp_num] = get_score_from_prediction(input_image[0,:,:,0], pred[i][:,:,0])
        # if imp_num == 717:
        #     print("loss of impurity 717: "+str(impurity_anomaly_shape_scores[imp_num]))

        # impurity_anomaly_shape_reconstruct_loss[imp_num] = evaluated_loss[i]
    # print(impurity_anomaly_shape_scores)

    # return impurity_anomaly_shape_reconstruct_loss
    return impurity_anomaly_shape_scores

if __name__ == "__main__":
    test_2_impurities('./model_ae.h5')
    # predict(path, impurities_num, model_name='./model.h5', height=600, width=600, BATCH_SIZE=64)


