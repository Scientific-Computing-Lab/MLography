from load_data import *


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# base_model = ResNet50(weights='imagenet',
#                       include_top=False,
#                       input_shape=(HEIGHT, WIDTH, 3))
# base_model = ResNet50(input_shape=(HEIGHT, WIDTH, 3))

TRAIN_DIR = "./ae_data/data/"
HEIGHT = 600
WIDTH = 600
BATCH_SIZE = 64

from keras.preprocessing.image import ImageDataGenerator
# create generator
datagen = ImageDataGenerator(rescale=1./255)
# datagen = ImageDataGenerator()
# prepare an iterators for each dataset
# train_it = datagen.flow_from_directory('ae_data/data_two_classes/train/', target_size=(HEIGHT, WIDTH),
#                                        class_mode="binary", batch_size=BATCH_SIZE)
# val_it = datagen.flow_from_directory('ae_data/data_two_classes/validation/', target_size=(HEIGHT, WIDTH),
#                                      class_mode="binary", batch_size=BATCH_SIZE)

train_it = datagen.flow_from_directory('ae_data/data/train/', target_size=(HEIGHT, WIDTH),
                                       class_mode=None, batch_size=BATCH_SIZE)
val_it = datagen.flow_from_directory('ae_data/data/validation/', target_size=(HEIGHT, WIDTH),
                                     class_mode=None, batch_size=BATCH_SIZE)
# confirm the iterator works

# batchX, batchy = train_it.next()
# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from keras.applications import VGG19
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

if K.image_data_format() == 'channels_first':
    input_shape = (3, WIDTH, HEIGHT)
else:
    input_shape = (WIDTH, HEIGHT, 3)

def autoencoder():
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
    x = Conv2D(3*16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(3*8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(3*8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(3*8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3*8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3*16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3*1, (3, 3), activation='sigmoid', padding='same')(x)

    # output = Reshape((WIDTH, HEIGHT, 3))(decoded)
    ae = Model(input_img, decoded)
    optimizer = Adam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ae.compile(optimizer=optimizer, loss='binary_crossentropy')
    # ae.compile(optimizer='adadelta', loss='binary_crossentropy')
    return ae


def get_model_autoencoder():
    input_img = Input(shape=input_shape, name="input_img")  # adapt this if using `channels_first` image data format
    bn_model = 0
    x = Conv2D(16, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(
        (BatchNormalization(momentum=bn_model))(input_img))
    x = Conv2D(16, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    #    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') ((BatchNormalization(momentum=bn_model))(input_img))
    #    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #    #img_1 = BatchNormalization()(img_1)
    #    img_1 = MaxPooling2D((2,2)) (img_1)
    #    img_1 = Dropout(0.3)(img_1)
    #    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #    #img_1 = BatchNormalization()(img_1)
    #    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #    #img_1 = BatchNormalization()(img_1)
    #
    #    img_1 = MaxPooling2D((2,2)) (img_1)
    #    img_1 = Dropout(0.3)(img_1)
    #    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)
    #    #img_1 = BatchNormalization()(img_1)
    #    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation,kernel_initializer='random_uniform') (img_1)

    encoded = MaxPooling2D((2, 2), padding='same', name="encoded")(x)

    x = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(encoded)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(16, (3, 3), activation='elu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='valid', name="decoded")(x)

    autoencoder = Model(input_img, decoded)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    return autoencoder


def get_model_regular_net():
    input_img = Input(shape=input_shape, name="input_img")  # adapt this if using `channels_first` image data format
    x = Conv2D(20, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(input_img)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    result = Dense(2, activation='softmax')(x)

    model = Model(input_img, result)
    optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
    return model


from keras.callbacks import TensorBoard

# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# model = get_model_regular_net()

# model = VGG19(weights=None, include_top=False, input_shape=(WIDTH, HEIGHT, 3), classes=2)
# last = model.output
#
# x = Flatten()(last)
# preds = Dense(2, activation='softmax')(x)
#
# model = Model(model.input, preds)
# optimizer = Adam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

model = autoencoder()


def fixed_generator(generator):
    for batch in generator:
        # y_hot = to_categorical(batch[1])
        yield (batch, batch)


history = model.fit_generator(fixed_generator(train_it), epochs=1000, validation_data=fixed_generator(val_it),
                              validation_steps=8,
                              steps_per_epoch=16, workers=8, use_multiprocessing=True)

# history = model.fit_generator(train_it, epochs=100, validation_data=val_it,
#                               validation_steps=8,
#                               steps_per_epoch=16, workers=8, use_multiprocessing=True)

# test_it_normal = datagen.flow_from_directory('ae_data/data/test_normal/', target_size=(HEIGHT, WIDTH),
#                                       class_mode=None, batch_size=BATCH_SIZE)
#
# test_it_anomaly = datagen.flow_from_directory('ae_data/data/test_anomaly/', target_size=(HEIGHT, WIDTH),
#                                       class_mode=None, batch_size=BATCH_SIZE)

test_it_combined = datagen.flow_from_directory('ae_data/test_with_2_classes/', target_size=(HEIGHT, WIDTH),
                                      class_mode="binary", batch_size=BATCH_SIZE)


# loss_normal = model.evaluate_generator(test_it_normal, steps=24)
loss_combined = model.evaluate_generator(test_it_combined, steps=24)

# Plot the training and validation loss + accuracy
def plot_training(history):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    #val_acc = history.history['val_acc']
    #loss = history.history['loss']
    #val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    #plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')

plot_training(history)







# def main():
    # load_train("./ae_data/all_regularized_impurities_train_normal/",
    #            "./ae_data/all_regularized_impurities_train_anomaly/")
    # load_test("./ae_data/all_regularized_impurities_test_normal/",
    #           "./ae_data/all_regularized_impurities_test_anomaly/")




# if __name__ == "__main__":
    # main()
