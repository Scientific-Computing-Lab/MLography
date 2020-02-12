from time import time
import numpy as np
from absl import flags, app


FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", None, "Path for Autoencoder model without extension")
flags.DEFINE_boolean("anomaly_blank_label", True, "True if the use of blank labels for anomalous impurity is desired")



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


def conv_autoencoder(input_shape, WIDTH, HEIGHT):
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

def conv_autoencoder_no_drop(input_shape, WIDTH, HEIGHT):
    input_img = Input(shape=input_shape)

    # encode with a deep net
    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(encoded)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(256, (5, 5), activation='relu', kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(256, (5, 5), activation='relu', kernel_initializer='random_uniform')(x)
    x = UpSampling2D((2, 2))(x)

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


def smaller_conv_autoencoder(input_shape, WIDTH, HEIGHT):
    input_img = Input(shape=input_shape)
    x = input_img

    depth = 4

    for encoding_layer in range(depth):
        x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        # x = Dropout(0.3)(x)

    for decoding_layer in range(depth-1):
        x = Conv2D(256, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
        x = UpSampling2D((2, 2))(x)
        # x = Dropout(0.3)(x)

    # x = Conv2D(128, (5, 5), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    # decoded = UpSampling2D((2, 2))(x)
    x = Flatten()(x)

    x = Dense(500, activation='relu')(x)
    x = Dense(1*WIDTH*HEIGHT, activation='sigmoid')(x)
    result = Reshape(input_shape)(x)

    ae = Model(input_img, result)
    optimizer = Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ae.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return ae



def fixed_generator(generator):
    """
    Modifies the data generator, such that normal data will get itself as label, and anomal data will get blank image
    as label. By that, the network learns to reconstruct successfully normal impurities, and will fail to reconstruct
    anomalous impurities.
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


def main(_):
    if FLAGS.model_name is None:
        print("Please provide a name for the model by providing --model_name=NAME without extension")
        return
    HEIGHT = 100
    WIDTH = 100
    BATCH_SIZE = 32
    EPOCHS_NUM = 500


    # create generator
    datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True, rotation_range=360)
    # datagen = ImageDataGenerator()
    # prepare an iterators for each dataset
    if FLAGS.anomaly_blank_label:
        train_it = datagen.flow_from_directory('data/rescaled_extended_2_classes/train/', target_size=(HEIGHT, WIDTH),
                                               class_mode="binary", batch_size=BATCH_SIZE, color_mode='grayscale')
        val_it = datagen.flow_from_directory('data/rescaled_extended_2_classes/validation/',
                                             target_size=(HEIGHT, WIDTH),
                                             class_mode="binary", batch_size=BATCH_SIZE, color_mode='grayscale')
    else:
        train_it = datagen.flow_from_directory('data/rescaled_extended_1_class/train/', target_size=(HEIGHT, WIDTH),
                                               class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')
        val_it = datagen.flow_from_directory('data/rescaled_extended_1_class/validation/', target_size=(HEIGHT, WIDTH),
                                             class_mode=None, batch_size=BATCH_SIZE, color_mode='grayscale')

    if K.image_data_format() == 'channels_first':
        input_shape = (1, WIDTH, HEIGHT)
    else:
        input_shape = (WIDTH, HEIGHT, 1)

    # model = conv_autoencoder(input_shape, WIDTH, HEIGHT)
    model = conv_autoencoder_no_drop(input_shape, WIDTH, HEIGHT)

    tbCallBack = TensorBoard(log_dir='./Graph/{}'.format(time()), histogram_freq=0, write_graph=True, write_images=True)

    if FLAGS.anomaly_blank_label:
        history = model.fit_generator(fixed_generator(train_it), epochs=EPOCHS_NUM,
                                      validation_data=fixed_generator(val_it),
                                      validation_steps=8, steps_per_epoch=16, callbacks=[tbCallBack])
        # history = model.fit_generator(fixed_generator(train_it), epochs=EPOCHS_NUM,
        #                               validation_data=fixed_generator(val_it),
        #                               validation_steps=8,
        #                               steps_per_epoch=16, workers=8, use_multiprocessing=True, callbacks=[tbCallBack])
    else:
        history = model.fit_generator(fixed_generator_none(train_it), epochs=EPOCHS_NUM,
                                      validation_data=fixed_generator_none(val_it),
                                      validation_steps=8, steps_per_epoch=16, callbacks=[tbCallBack])
        # history = model.fit_generator(fixed_generator_none(train_it), epochs=EPOCHS_NUM,
        #                               validation_data=fixed_generator_none(val_it),
        #                               validation_steps=8,
        #                               steps_per_epoch=16, workers=8, use_multiprocessing=True, callbacks=[tbCallBack])


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

    model.save(FLAGS.model_name + ".h5")

    # Write the net summary
    with open(FLAGS.model_name + '_summary.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print("Saved model to disk")

    plot_training(history)

if __name__ == "__main__":
   app.run(main)
