import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Dropout
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import RandomFlip, RandomRotation


def inputBlock(inputs, filters, kernel_size=3, padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inputs)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x)

    return x


def contractiveBlock(inputs, filters, kernel_size=3, padding='same', dropout_rate=0.2):
    x = MaxPooling2D((2,2), strides=2)(inputs)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu')(x)

    return x


def expansiveBlock(inputs, skip, filters, kernel_size=3, trans_kernel_size=2, trans_strides=2, padding='same'):
    x = Conv2DTranspose(filters=filters, kernel_size=trans_kernel_size, strides=trans_strides, padding=padding)(inputs)
    concat = Concatenate(axis=3)([skip, x])
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu')(concat)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu')(x)

    return x


def get_unet(inputs, num_classes, activation='softmax'):
    input_block = inputBlock(inputs, 64)
    skip_input = input_block
    # x = RandomFlip("horizontal_and_vertical")(input_block)
    # x = RandomRotation(0.3)(x)

    # contractive path
    c1 = contractiveBlock(input_block, 128)
    skip_c1 = c1
    c2 = contractiveBlock(c1, 256)
    skip_c2 = c2
    c3 = contractiveBlock(c2, 512)
    skip_c3 = c3
    c4 = contractiveBlock(c3, 1024)

    # expansive path
    e1 = expansiveBlock(c4, skip_c3, 512)
    e2 = expansiveBlock(e1, skip_c2, 256)
    e3 = expansiveBlock(e2, skip_c1, 128)
    e4 = expansiveBlock(e3, skip_input, 64)

    # outputs
    if num_classes == 1:
        activation = 'sigmoid'

    outputs = Conv2D(num_classes, 1, activation=activation)(e4)

    # build model
    model = Model(inputs=inputs, outputs=outputs)

    return model

