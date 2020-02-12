import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import array_to_img, save_img, ImageDataGenerator
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from skimage import measure
    # from skimage.measure import structural_similarity as ssim
    import re
    import cv2 as cv
    import ray
    from utils import num_threads
    from glob import glob


def load_image(img_path, height=100, width=100):

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


def postprocess_prediction(prediction):
    image = np.array(prediction)
    image *= 255
    image = 255 - image
    ret, thresh = cv.threshold(image, 100, 255, cv.THRESH_BINARY_INV)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)

    # post_img = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=3)
    # post_img = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=3)

    post_img = cv.morphologyEx(thresh, cv.MORPH_DILATE, kernel, iterations=3)
    post_img = cv.morphologyEx(post_img, cv.MORPH_ERODE, kernel, iterations=3)

    return post_img
    # return image
    # return thresh


def test_prediction(model, img_path, height, width, img_name, out_path):
    img = cv.imread(img_path)
    save_img(out_path + img_name + '.jpg', img)

    img = load_image(img_path, height, width)
    pred = model.predict(img)
    pred_img = postprocess_prediction(pred[0, :, :, :])

    l11 = measure.compare_ssim(img[0, :, :, 0], pred[0, :, :, 0])
    l12 = measure.compare_ssim(img[0, :, :, 0], pred_img)
    l13 = mean_squared_error(img[0, :, :, 0], pred[0, :, :, 0])
    l14 = mean_squared_error(img[0, :, :, 0], pred_img)
    # print("predicted {}, ssim loss is={}".format(img_name, l11))
    # print("predicted {} post, ssim loss is={}".format(img_name, l12))
    print("predicted {}, mse loss is={}".format(img_name, l13))
    print("predicted {} post, mse loss is={}".format(img_name, l14))

    save_img(out_path + 'reconstructed_' + img_name + '.jpg', pred[0, :, :, :])

    pred_img = np.expand_dims(pred_img, axis=2)
    save_img(out_path + 'reconstructed_post_' + img_name + '.jpg', pred_img)


def test_impurities(model_name, height=100, width=100, out_path='./'):
    model = tf.keras.models.load_model(model_name)

    normal_imp = glob("./data/test_scan3tag-48/*/*impurity_242.png")[0]

    anomaly_imp = glob("./data/test_scan2tag-34/*/*impurity_875.png")[0]

    anomaly_line = glob("./data/test_scan1tag-47/*/*impurity_1056.png")[0]

    anomaly_imp717 = glob("./data/test_scan1tag-47/*/*impurity_717.png")[0]

    anomaly_imp699 = glob("./data/test_scan1tag-47/*/*impurity_699.png")[0]

    normal_imp2228 = glob("./data/test_scan1tag-47/*/*impurity_2228.png")[0]

    normal_imp2345 = glob("./data/test_scan1tag-47/*/*impurity_2345.png")[0]

    normal_imp2258 = glob("./data/test_scan1tag-47/*/*impurity_2258.png")[0]

    normal_imp2131 = glob("./data/test_scan1tag-47/*/*impurity_2131.png")[0]

    normal_imp2309 = glob("./data/test_scan1tag-47/*/*impurity_2309.png")[0]

    test_prediction(model, normal_imp, height, width, "normal", out_path)
    test_prediction(model, anomaly_imp, height, width, "anomaly", out_path)
    test_prediction(model, anomaly_line, height, width, "anomaly_line", out_path)
    test_prediction(model, anomaly_imp717, height, width, "anomaly_717", out_path)
    test_prediction(model, anomaly_imp699, height, width, "anomaly_699", out_path)
    test_prediction(model, normal_imp2228, height, width, "normal2228", out_path)
    test_prediction(model, normal_imp2345, height, width, "normal2345", out_path)
    test_prediction(model, normal_imp2258, height, width, "normal2258", out_path)
    test_prediction(model, normal_imp2131, height, width, "normal2131", out_path)
    test_prediction(model, normal_imp2309, height, width, "normal2309", out_path)


    # Write the net summary
    with open(out_path + 'summary.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print("saved images")




def get_score_from_prediction(input, prediction):
    loss = mean_squared_error(input, prediction)
    # loss = measure.compare_ssim(input, prediction)
    return loss

@ray.remote
def get_scores_single(files_chunk, path, pred_chunk):
    chunk_indices = []
    impurity_anomaly_shape_scores = np.full(len(files_chunk), np.infty)
    for i in range(len(files_chunk)):
        img_name = os.path.splitext(os.path.basename(files_chunk[i]))[0]
        img_name = img_name[img_name.find("_impurity_"):]
        imp_num = int(re.search(r'\d+', img_name).group())

        input_image = load_image(path + files_chunk[i])
        post_pred = postprocess_prediction(pred_chunk[i][:,:,0])
        impurity_anomaly_shape_scores[i] = get_score_from_prediction(input_image[0, :, :, 0], post_pred)

        # if impurity_anomaly_shape_scores[i] is np.infty:
        #     img = cv.imread(path + files_chunk[i])
        #     save_img("/home/matanr/MLography/logs/shape/under_thresh/" + "imp" + str(imp_num) + ".png", img)
        #     pred_img = np.expand_dims(post_pred, axis=2)
        #     save_img("/home/matanr/MLography/logs/shape/under_thresh/" + "post_recon" + str(imp_num) + ".png", pred_img)
        #     save_img("/home/matanr/MLography/logs/shape/under_thresh/" + "recon" + str(imp_num) + ".png", pred_chunk[i][:,:,:])

        chunk_indices.append(imp_num)
    return chunk_indices, impurity_anomaly_shape_scores


def predict(path, impurities_num, model=None, model_name='./model_ae_extended.h5',
            height=100, width=100, BATCH_SIZE=64):
    if model is None:
        model = tf.keras.models.load_model(model_name)

    datagen = ImageDataGenerator(rescale=1. / 255)
    test_it = datagen.flow_from_directory(path, target_size=(height, width), class_mode=None, shuffle=False,
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


def check_post_process(img_path, out_dir):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    post_img = postprocess_prediction(gray)


    out_img = np.expand_dims(post_img, axis=2)
    img_base_name = os.path.basename(img_path)
    save_img(out_dir + 'check_post_' + img_base_name, out_img)


if __name__ == "__main__":
    # check_post_process("/home/matanr/MLography/logs/shape/under_thresh/recon303.png",
    #                    "/home/matanr/MLography/logs/shape/under_thresh/")
    print("ae_model_same_label")
    test_impurities('./ae_model_same_label.h5',
                    out_path='/home/matanr/MLography/logs/shape/reconstructed_ae_model_same_label/')
    print("ae_model_blank_label")
    test_impurities('./ae_model_blank_label.h5',
                    out_path='/home/matanr/MLography/logs/shape/reconstructed_ae_model_blank_label/')
    print("older")
    test_impurities('./ae_blank_label.h5', out_path='/home/matanr/MLography/logs/shape/reconstructed_ae_blank_label/')


    # test_impurities('./model_ae.h5',
    #                   out_path='/home/matanr/MLography/logs/shape/reconstructed_old_model/')
    # test_impurities('./model_blank_label.h5',
    #                   out_path='/home/matanr/MLography/logs/shape/reconstructed_model_blank_label/')



    # test_impurities('./smaller_blank_label.h5',
    #                   out_path='/home/matanr/MLography/logs/shape/reconstructed_blank_label/')

    # predict(path, impurities_num, model_name='./model.h5', height=600, width=600, BATCH_SIZE=64)


