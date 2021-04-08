import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.utils import multi_gpu_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import keras.backend.tensorflow_backend as tfback

# print("tf.__version__ is", tf.__version__)
# print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


def binary_focal_loss(gamma=2, alpha=0.99):
    """
    Binary form of focal loss.
         Focal loss for binary classification problems

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def unet(pretrained_weights=None, input_size=(512, 512, 1), loss_func='binary_crossentropy'):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-6), loss=loss_func, metrics=['accuracy'])
    
    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



def unet16(input_size=(512, 512, 3), loss_func='binary_crossentropy'):

    # Get back the convolutional part of a VGG network trained on ImageNet
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_size)
    # vgg16 = VGG16(weights=None, include_top=False, input_shape=input_size)
    # vgg16.summary()

    # Create your own input format
    input = Input(input_size, name='image_input')

    # Use the generated model
    block1_conv1 = vgg16.get_layer("block1_conv1")(input)
    block1_conv2 = vgg16.get_layer("block1_conv2")(block1_conv1)
    block1_pool = vgg16.get_layer("block1_pool")(block1_conv2)

    block2_conv1 = vgg16.get_layer("block2_conv1")(block1_pool)
    block2_conv2 = vgg16.get_layer("block2_conv2")(block2_conv1)
    block2_pool = vgg16.get_layer("block2_pool")(block2_conv2)

    block3_conv1 = vgg16.get_layer("block3_conv1")(block2_pool)
    block3_conv2 = vgg16.get_layer("block3_conv2")(block3_conv1)
    block3_conv3 = vgg16.get_layer("block3_conv3")(block3_conv2)
    block3_pool = vgg16.get_layer("block3_pool")(block3_conv3)

    block4_conv1 = vgg16.get_layer("block4_conv1")(block3_pool)
    block4_conv2 = vgg16.get_layer("block4_conv2")(block4_conv1)
    block4_conv3 = vgg16.get_layer("block4_conv3")(block4_conv2)
    block4_pool = vgg16.get_layer("block4_pool")(block4_conv3)

    block5_conv1 = vgg16.get_layer("block5_conv1")(block4_pool)
    block5_conv2 = vgg16.get_layer("block5_conv2")(block5_conv1)
    block5_conv3 = vgg16.get_layer("block5_conv3")(block5_conv2)
    # block5_pool = vgg16.get_layer("block5_pool")(block5_conv3)

    drop5 = Dropout(0.5)(block5_conv3)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([block4_conv3, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([block3_conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([block2_conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([block1_conv2, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)
    # model.summary()

    model.compile(optimizer=Adam(lr=1e-6), loss=loss_func, metrics=['accuracy'])
    return model

