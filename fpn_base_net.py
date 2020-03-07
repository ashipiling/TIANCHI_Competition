from keras import layers, backend, regularizers
# from faster_rcnn.layers.batch_norm import BatchNorm
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import TimeDistributed


def resnet50(inputs, l2_reg=5e-4):
    bn_axis = 3
    #
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      bias_regularizer=regularizers.l2(l2_reg),
                      name='conv1')(x)
    x = BatchNorm(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    c3 = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(c3, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    c4 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # model = Model(input, x, name='resnet50')

    #输出8倍和16倍下采样feature map

    base4 = layers.Conv2D(512, (1, 1), name='fpn_b4')(c4)
    base3 = layers.Add(name="fpn_b3_add_b4ups")([
        layers.UpSampling2D(size=(2, 2), name="fpn_b4_upsampled")(base4),
        layers.Conv2D(512, (1, 1), name='fpn_b3')(c3)])
    return [base3, base4]


def resnet50_head(features, l2_reg=5e-4):
    filters = 512
    x = features
    x = conv_block_5d(x, 3, [filters, filters, filters * 4],
                      stage=5, block='a', strides=(2, 2), l2_reg=l2_reg)
    x = identity_block_5d(x, 3, [filters, filters, filters * 4], stage=5, block='b', l2_reg=l2_reg)
    x = identity_block_5d(x, 3, [filters, filters, filters * 4], stage=5, block='c', l2_reg=l2_reg)
    # 全局平均池化(batch_size,roi_num,channels)
    x = layers.TimeDistributed(layers.GlobalAvgPool2D())(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, l2_reg=5e-4):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      bias_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      bias_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2b')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      bias_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2c')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               l2_reg=5e-4):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      bias_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      bias_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2b')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      bias_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2c')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(l2_reg),
                             bias_regularizer=regularizers.l2(l2_reg),
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNorm(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def identity_block_5d(input_tensor, kernel_size, filters, stage, block, l2_reg):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.TimeDistributed(layers.Conv2D(filters1, (1, 1),
                                             kernel_regularizer=regularizers.l2(l2_reg),
                                             bias_regularizer=regularizers.l2(l2_reg),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2a')(input_tensor)
    x = layers.TimeDistributed(BatchNorm(axis=-1), name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(filters2, kernel_size,
                                             padding='same',
                                             kernel_regularizer=regularizers.l2(l2_reg),
                                             bias_regularizer=regularizers.l2(l2_reg),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2b')(x)
    x = layers.TimeDistributed(BatchNorm(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(filters3, (1, 1),
                                             kernel_regularizer=regularizers.l2(l2_reg),
                                             bias_regularizer=regularizers.l2(l2_reg),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2c')(x)
    x = layers.TimeDistributed(BatchNorm(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block_5d(input_tensor,
                  kernel_size,
                  filters,
                  stage,
                  block,
                  strides=(2, 2),
                  l2_reg=5e-4):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    print(input_tensor.shape)
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.TimeDistributed(layers.Conv2D(filters1, (1, 1), strides=strides,
                                             kernel_regularizer=regularizers.l2(l2_reg),
                                             bias_regularizer=regularizers.l2(l2_reg),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2a')(input_tensor)
    x = layers.TimeDistributed(BatchNorm(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(filters2, kernel_size, padding='same',
                                             kernel_regularizer=regularizers.l2(l2_reg),
                                             bias_regularizer=regularizers.l2(l2_reg),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2b')(x)
    x = layers.TimeDistributed(BatchNorm(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(filters3, (1, 1),
                                             kernel_regularizer=regularizers.l2(l2_reg),
                                             bias_regularizer=regularizers.l2(l2_reg),
                                             kernel_initializer='he_normal'),
                               name=conv_name_base + '2c')(x)
    x = layers.TimeDistributed(BatchNorm(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = layers.TimeDistributed(layers.Conv2D(filters3, (1, 1), strides=strides,
                                                    kernel_regularizer=regularizers.l2(l2_reg),
                                                    bias_regularizer=regularizers.l2(l2_reg),
                                                    kernel_initializer='he_normal'),
                                      name=conv_name_base + '1')(input_tensor)
    shortcut = layers.TimeDistributed(BatchNorm(
        axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x
