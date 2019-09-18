from time import time
import numpy as np

HEIGHT = 100
WIDTH = 100
BATCH_SIZE = 64
EPOCHS_NUM = 500

anomaly_blank_label = True

from keras.preprocessing.image import ImageDataGenerator
# create generator
datagen = ImageDataGenerator(rescale=1./255)
# datagen = ImageDataGenerator()
# prepare an iterators for each dataset
if anomaly_blank_label:
    # train_it = datagen.flow_from_directory('data/data_two_classes/train/', target_size=(HEIGHT, WIDTH),
    #                                        class_mode="binary", batch_size=BATCH_SIZE, color_mode='grayscale')
    # val_it = datagen.flow_from_directory('data/data_two_classes/validation/', target_size=(HEIGHT, WIDTH),
    #                                      class_mode="binary", batch_size=BATCH_SIZE, color_mode='grayscale')
    train_it = datagen.flow_from_directory('data/rescaled_extended_2_classes/train/', target_size=(HEIGHT, WIDTH),
                                           class_mode="binary", batch_size=BATCH_SIZE, color_mode='grayscale')
    val_it = datagen.flow_from_directory('data/rescaled_extended_2_classes/validation/', target_size=(HEIGHT, WIDTH),
                                         class_mode="binary", batch_size=BATCH_SIZE, color_mode='grayscale')

# train_it = datagen.flow_from_directory('data/data_two_classes/train/', target_size=(HEIGHT, WIDTH),
#                                        class_mode=None, batch_size=BATCH_SIZE)
# val_it = datagen.flow_from_directory('data/data_two_classes/validation/', target_size=(HEIGHT, WIDTH),
#                                      class_mode=None, batch_size=BATCH_SIZE)
else:
    # train_it = datagen.flow_from_directory('data/data/train/', target_size=(HEIGHT, WIDTH),
    #                                        class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')
    # val_it = datagen.flow_from_directory('data/data/validation/', target_size=(HEIGHT, WIDTH),
    #                                      class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')
    train_it = datagen.flow_from_directory('data/rescaled_extended_1_class/train/', target_size=(HEIGHT, WIDTH),
                                           class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')
    val_it = datagen.flow_from_directory('data/rescaled_extended_1_class/validation/', target_size=(HEIGHT, WIDTH),
                                         class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

if K.image_data_format() == 'channels_first':
    input_shape = (1, WIDTH, HEIGHT)
else:
    input_shape = (WIDTH, HEIGHT, 1)


def conv_autoencoder():
    input_img = Input(shape=input_shape)

    # encode with a deep net
    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.3)(encoded)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), activation='relu', kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), activation='relu', kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (5, 5), activation='relu', kernel_initializer='random_uniform')(x)
    decoded = UpSampling2D((2, 2))(x)

    x = Flatten()(decoded)

    x = Dense(500, activation='relu')(x)
    x = Dense(1*WIDTH*HEIGHT, activation='sigmoid')(x)
    result = Reshape(input_shape)(x)

    ae = Model(input_img, result)
    optimizer = Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ae.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
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
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
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
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(16, (3, 3), activation='elu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (2, 2), activation='relu', padding='valid', name="decoded")(x)

    x = Flatten()(decoded)
    x = Dense(128, activation='relu')(x)
    x = Dense(3 * WIDTH * HEIGHT, activation='sigmoid')(x)
    result = Reshape(input_shape)(x)

    autoencoder = Model(input_img, result)
    optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder


def get_model_autoencoder_fully_connected():
    # min=32, on test:
    # Normal:  Loss:  0.000199182173673762 Accuracy:  0.3337676227092743
    # Anoamly: Loss:  0.001865490889770398 Accuracy:  0.3337482490229258
    input_img = Input(shape=input_shape, name="input_img")  # adapt this if using `channels_first` image data format
    x = Flatten()(input_img)

    x = Dense(550, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(540, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(520, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(0.3)(x)
    # x = Dense(400, activation='relu')(x)
    # x = Dense(350, kernel_regularizer='l2', activation='tanh')(x)
    # x = Dense(400, kernel_regularizer='l2', activation='relu')(x)
    # x = Dense(600, kernel_regularizer='l2', activation='relu')(x)
    # # x = Dense(700, kernel_regularizer='l2', activation='relu')(x)
    # # x = Dropout(0.3)(x)
    # x = Dense(600, kernel_regularizer='l2', activation='relu')(x)
    # x = Dense(600, kernel_regularizer='l2', activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(520, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(540, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(550, activation='relu')(x)

    # x = Dense(100, kernel_regularizer='l2', activation='relu')(x)
    # x = Dropout(0.3)(x)

    x = Dense(1 * WIDTH * HEIGHT, activation='sigmoid')(x)
    result = Reshape(input_shape)(x)

    autoencoder = Model(input_img, result)
    optimizer = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['accuracy'])
    return autoencoder


def get_model_regular_net():
    input_img = Input(shape=input_shape, name="input_img")  # adapt this if using `channels_first` image data format
    x = Conv2D(16, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(input_img)
    x = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    result = Dense(2, activation='softmax')(x)

    model = Model(input_img, result)
    optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# model = get_model_regular_net()
model = conv_autoencoder()
# model = get_model_autoencoder()
# model = get_model_autoencoder_fully_connected()

def fixed_generator(generator):
    """
    Modifies the data generator, such that normal data will get itself as label, and anomal data will get blank image
    as label. By that, the network learns to reconstruct successfully normal impurities, and will fail to reconstruct
    anomal impurities.
    """
    for batch in generator:
        fixed_x = np.empty(shape=batch[0].shape, dtype="float32")
        fixed_y = np.empty(shape=batch[0].shape, dtype="float32")
        data_pice_counter = 0
        for (x, y) in zip(batch[0], batch[1]):
            fixed_x[data_pice_counter] = x
            if y == 0:  # anomaly, give blank image as label. Thus, the auto encoder won't be able to reconstruct.
                blank = np.full(fill_value=1, shape=x.shape, dtype="float32")
                fixed_y[data_pice_counter] = blank
            else: # normal, give the input image as label so the auto encoder will be able to reconstruct.
                fixed_y[data_pice_counter] = x
            data_pice_counter += 1
        yield (fixed_x, fixed_y)


def fixed_generator_none(generator):
    """
    """
    for batch in generator:
        yield (batch, batch)

tbCallBack = TensorBoard(log_dir='./Graph/{}'.format(time()), histogram_freq=0, write_graph=True, write_images=True)

if anomaly_blank_label:
    history = model.fit_generator(fixed_generator(train_it), epochs=EPOCHS_NUM,
                                  validation_data=fixed_generator(val_it),
                                  validation_steps=8,
                                  steps_per_epoch=16, workers=8, use_multiprocessing=True, callbacks=[tbCallBack])
else:
    history = model.fit_generator(fixed_generator_none(train_it), epochs=EPOCHS_NUM,
                                  validation_data=fixed_generator_none(val_it),
                                  validation_steps=8,
                                  steps_per_epoch=16, workers=8, use_multiprocessing=True, callbacks=[tbCallBack])
# for regular net

# history = model.fit_generator(train_it, epochs=EPOCHS_NUM, validation_data=val_it,
#                               validation_steps=8,
#                               steps_per_epoch=16, workers=8, use_multiprocessing=True)


# not rescaled dataset

# test_it_normal = datagen.flow_from_directory('data/data/test_normal/', target_size=(HEIGHT, WIDTH),
#                                       class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')
#
# test_it_anomaly = datagen.flow_from_directory('data/data/test_anomaly/', target_size=(HEIGHT, WIDTH),
#                                       class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')

# rescaled dataset

test_it_normal = datagen.flow_from_directory('data/test_rescaled_extended/normal/', target_size=(HEIGHT, WIDTH),
                                             class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')

test_it_anomaly = datagen.flow_from_directory('data/test_rescaled_extended/anomaly/', target_size=(HEIGHT, WIDTH),
                                              class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')

# test_it_combined = datagen.flow_from_directory('data/test_with_2_classes/', target_size=(HEIGHT, WIDTH),
#                                       class_mode="binary", batch_size=BATCH_SIZE)


score = model.evaluate_generator(fixed_generator_none(test_it_normal), steps=24)
print("Normal:  Loss: ", score[0], "Accuracy: ", score[1])

score = model.evaluate_generator(fixed_generator_none(test_it_anomaly), steps=24)
print("Anoamly: Loss: ", score[0], "Accuracy: ", score[1])

model.save("model_ae_extended.h5")

# Write the net summary
with open('model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

print("Saved model to disk")


# Plot the training and validation loss + accuracy
def plot_training(history):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.show()
    plt.savefig('acc_vs_epochs.png')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()

    plt.savefig('loss_vs_epochs.png')

plot_training(history)


# def main():

# if __name__ == "__main__":
    # main()
