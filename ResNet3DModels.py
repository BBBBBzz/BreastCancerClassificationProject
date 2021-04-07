from __future__ import division, absolute_import, print_function, unicode_literals
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv3D, AveragePooling3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.regularizers import l2
from keras import backend as K
import numpy as np
import six


def _bn_relu(input):
    """
    A helper function to build Batch Normalization(BN) followed by ReLU.
    :param input: 5-dimensional input for BN
    :returns: Output of ReLU activation
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation('relu')(norm)


def _conv3D_bn_relu(**conv_params):
    """
    A helper function to build a [3D convolutional layer -> Batch Normalization -> ReLU] block.
    :param **conv_params: **conv_params includes the following keywords:
        - filters: Number of filters used in convolutional layer
        - kernel_size: Size of kernel used in convolutional layer, e.g. (3, 3, 3)
        - strides: Size of stride used in convolutional layer, e.g. (1, 1, 1)
        - kernel_initializer: Kernel initialization used in convolutional layer, e.g. 'he_normal'
        - padding: Padding mode used in covolutional layer, e.g. 'same'
        - kernel_regularizer: Kernel regularization used in convolutional layer, e.g. l2(1e-4)
    :param input: 5-dimensional input for block 'conv3D_bn_relu'
    :returns: A call to an inner function consisting of block 'conv3D_bn_relu'
    """
    filters = conv_params['filters']
    kernel_size = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1, 1, 1))
    kernel_initializer = conv_params.setdefault('kernel_initializer', 'he_normal')
    padding = conv_params.setdefault('padding', 'same')
    kernel_regularizer = conv_params.setdefault('kernel_regularizer', l2(1e-4))

    def inner_func(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kenel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return inner_func


def _bn_relu_conv3D(**conv_params):
    """
    A helper function to build a [Batch Normalization -> ReLU -> 3D convolutional layer] block .
    :param **conv_params: **conv_params includes the following keywords:
        - filters: Number of filters used in convolutional layer
        - kernel_size: Size of kernel used in convolutional layer, e.g. (3, 3, 3)
        - strides: Size of stride used in convolutional layer, e.g. (1, 1, 1)
        - kernel_initializer: Kernel initialization used in convolutional layer, e.g. 'he_normal'
        - padding: Padding mode used in covolutional layer, e.g. 'same'
        - kernel_regularizer: Kernel regularization used in convolutional layer, e.g. l2(1e-4)
    :param input: 5-dimensional input for block 'bn_relu_conv3D'
    :returns: A call to an inner function consisting of block 'bn_relu_conv3D'
    """
    filters = conv_params['filters']
    kernel_size = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1, 1, 1))
    kernel_initializer = conv_params.setdefault('kernel_initializer', 'he_normal')
    padding = conv_params.setdefault('padding', 'same')
    kernel_regularizer = conv_params.setdefault('kernel_regularizer', l2(1e-4))

    def inner_func(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)

    return inner_func


def _shortcut3D(input, residual):
    """
    A 3D shortcut to match input and residual and merges them together with "add" operation.
    :param input: 5-dimensional input and also the dentity map of it
    :param residual: Output of the 5-dimensional input which passed through
                     a series of operations (convolution, batch normalization, etc.)
    :returns: the merged version of the input and the processed residual
    """
    # compare the shape of the input and the residual
    # to determine if we need to use 1 x 1 convolution
    # to help match the shape
    stride_dim1 = input._keras_shape[DIM1_AXIS] \
        // residual._keras_shape[DIM1_AXIS]
    stride_dim2 = input._keras_shape[DIM2_AXIS] \
        // residual._keras_shape[DIM2_AXIS]
    stride_dim3 = input._keras_shape[DIM3_AXIS] \
        // residual._keras_shape[DIM3_AXIS]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] \
        == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
        or not equal_channels:
        shortcut = Conv3D(filters=residual._keras_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1, 1),
                          strides=(stride_dim1, stride_dim2, stride_dim3),
                          kernel_initializer='he_normal', padding='valid',
                          kernel_regularizer=l2(1e-4))(input)

    return add([shortcut, residual])


def _residual_block3D(block_func, filters, kernel_regularizer, repetitions,
                      is_first_layer=False):
    """
    A 3D residual block consisting of residual connection and basic connection.
    :param block_func: Can either be 'bottleneck_block' or 'basic_block'
    :param kernel_regularizer: Kernel regularization used in convolutional layer, e.g. l2(1e-4)
    :param repetitions: A tuple indicating how many times a certain residual block needs to be repeated
    :param is_first_layer: A boolean value indicating whether it is the first layer of a series repeated blocks
    :param input: 5-dimensional input for block 'residual_block3D'
    :returns: A call to an inner function consisting of block 'residual_block3D'
    """
    def inner_func(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filster, kernel_size=(3, 3, 3),
                           strides=strides, padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=kernel_regularizer)(input)
        else:
            conv1 = _bn_relu_conv3D(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer)(input)
        residual = _bn_relu_conv3D(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer)(conv1)
        return _shortcut3D(input, residual)

    return inner_func


def bottleneck_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                     is_first_block_of_first_layer=False):
    """
    A 3D bottelneck block consisting of 1 x 1 convolution to downsample the input dimension.
    :param filters: Number of filters used in convolutional layer
    :param strides: Size of stride used in convolutional layer, e.g. (1, 1, 1)
    :param kernel_regularizer: Kernel regularization used in convolutional layer, e.g. l2(1e-4)
    :param is_first_block_of_first_layer: A boolean value indicating whether it is the first block of first layer
    :param input: 5-dimensional input for block 'residual_block3D'
    :returns: A call to an inner function consisting of block 'bottleneck_block'
    """
    def inner_func(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                              strides=strides, padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=kernel_regularizer
                              )(input)
        else:
            conv_1_1 = _bn_relu_conv3D(filters=filters, kernel_size=(1, 1, 1),
                                       strides=strides,
                                       kernel_regularizer=kernel_regularizer
                                       )(input)

        conv_3_3 = _bn_relu_conv3D(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_1_1)
        residual = _bn_relu_conv3D(filters=filters * 4, kernel_size=(1, 1, 1),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_3_3)

        return _shortcut3D(input, residual)

    return inner_func


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """
    A 3D basic block consisting 3 x 3 convolutions and indentity function.
    :param filters: Number of filters used in convolutional layer
    :param strides: Size of stride used in convolutional layer, e.g. (1, 1, 1)
    :param kernel_regularizer: Kernel regularization used in convolutional layer, e.g. l2(1e-4)
    :param is_first_block_of_first_layer: A boolean value indicating whether it is the first block of first layer
    :param input: 5-dimensional input for block 'residual_block3D'
    :returns: A call to an inner function consisting of block 'basic_block'
    """

    def inner_func(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer
                           )(input)
        else:
            conv1 = _bn_relu_conv3D(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer
                                    )(input)
        residual = _bn_relu_conv3D(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv1)
        return _shortcut3D(input, residual)

    return inner_func


def _handle_data_format():
    """
    A helper function convert the data format of input into number representation
    accroding to the used backend.
    """
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    """
    A helper function to recognize the indentifier and
    return the call to the corresponding function.
    """
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResNet3DBuilder(object):
    """ResNet3D."""

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor):
        """
        Instantiate a vanilla ResNet3D keras model.
        :param input_shape: Tuple of input shape in the format
                            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
                            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
        :param num_outputs: The number of outputs at the final softmax layer
        :param block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
        :param repetitions: Repetitions of unit blocks
        :Returns: model: A 3D ResNet model that takes a 5D tensor (volumetric images
                         in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        block_fn = _get_block(block_fn)
        input = Input(shape=input_shape)
        # first conv
        conv1 = _conv3D_bn_relu(filters=64, kernel_size=(7, 7, 7),
                                strides=(2, 2, 2),
                                kernel_regularizer=l2(reg_factor)
                                )(input)
        pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2),
                             padding="same")(conv1)

        # repeat blocks
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block3D(block_fn, filters=filters,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average poll and classification
        pool2 = AveragePooling3D(pool_size=(block._keras_shape[DIM1_AXIS],
                                            block._keras_shape[DIM2_AXIS],
                                            block._keras_shape[DIM3_AXIS]),
                                 strides=(1, 1, 1))(block_output)
        # additional drop out
        pool2 = Dropout(0.5)(pool2)
        flatten1 = Flatten()(pool2)

        # if num_outputs > 1:
        #     dense = Dense(units=num_outputs,
        #                   kernel_initializer="he_normal",
        #                   activation="softmax",
        #                   kernel_regularizer=l2(reg_factor))(flatten1)
        # else:
        #     dense = Dense(units=num_outputs,
        #                   kernel_initializer="he_normal",
        #                   activation="sigmoid",
        #                   kernel_regularizer=l2(reg_factor))(flatten1)

        dense = flatten1
        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 18."""
        return ResNet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [2, 2, 2, 2], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 34."""
        return ResNet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 50."""
        return ResNet3DBuilder.build(input_shape, num_outputs, bottleneck_block,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 101."""
        return ResNet3DBuilder.build(input_shape, num_outputs, bottleneck_block,
                                     [3, 4, 23, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 152."""
        return ResNet3DBuilder.build(input_shape, num_outputs, bottleneck_block,
                                     [3, 8, 36, 3], reg_factor=reg_factor)
