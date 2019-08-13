from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def load_image(img_path, show=True):

    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


def test_2_impurities():
    model = load_model('./model.h5')

    anomaly_imp = "./data/test_with_2_classes/anomaly/0.603991446964322scan3tag-16_impurity_1028.png"
    normal_imp = "./data/test_with_2_classes/normal/scan1tag-1_impurity_1936.png"

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # load a single image
    new_image_anomaly = load_image(anomaly_imp)
    new_image_normal = load_image(normal_imp)

    # check prediction
    pred_a = model.predict(new_image_anomaly)
    pred_n = model.predict(new_image_normal)

    # print prediction
    print('Predicted anomaly:', pred_a)
    print('Predicted normal:', pred_n)



def test_scan(HEIGHT=100, WIDTH=100, BATCH_SIZE=64, path="./data/test_scan1tag-47/"):
    model = load_model('./model.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1. / 255)
    test_it = datagen.flow_from_directory(path, target_size=(HEIGHT, WIDTH), class_mode=None, batch_size=BATCH_SIZE)
    test_it.reset()
    pred = model.predict_generator(test_it, verbose=1, steps=1253/BATCH_SIZE)
    print(pred)
    # labels = (train_it.class_indices)
    # labels = dict((v, k) for k, v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]


test_scan()



