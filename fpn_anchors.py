import tensorflow as tf
import keras
import numpy as np

def generate_anchors(heights, widths, base_size, ratios, scales):
    """
    :param heights: anchor高度列表
    :param widths: anchor宽度列表
    根据基准尺寸、长宽比、缩放比生成边框
    :param base_size: anchor的base_size,如：64
    :param ratios: 长宽比 shape:(M,)
    :param scales: 缩放比 shape:(N,)
    :return: （N*M,(y1,x1,y2,x2))
    """
    if heights is not None:
        h = np.array(heights, np.float32)
        w = np.array(widths, np.float32)
    else:
        ratios = np.expand_dims(np.array(ratios), axis=1)  # (N,1)
        scales = np.expand_dims(np.array(scales), axis=0)  # (1,M)
        # 计算高度和宽度，形状为(N,M)
        h = np.sqrt(ratios) * scales * base_size
        w = 1.0 / np.sqrt(ratios) * scales * base_size

    # reshape为（N*M,1)
    h = np.reshape(h, (-1, 1))
    w = np.reshape(w, (-1, 1))

    return np.hstack([-0.5 * h, -0.5 * w, 0.5 * h, 0.5 * w])

def generate_anchors2(heights, widths, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios

    # Enumerate heights and widths from scales and ratios
    heights = np.array(heights)
    widths = np.array(widths)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes

def shift(feature, strides, base_anchors):
    """
    根据feature map的长宽，生成所有的anchors
    :param feature: feature map
    :param strides: 步长
    :param base_anchors:所有的基准anchors，(anchor_num,4)
    :return:
    """
    shape = tf.shape(feature) #[-1, h, w, 3]
    H, W = shape[1], shape[2]
    print("shape:{}".format(shape))
    ctr_x = (tf.cast(tf.range(W), tf.float32) + tf.constant(0.5, dtype=tf.float32)) * strides
    ctr_y = (tf.cast(tf.range(H), tf.float32) + tf.constant(0.5, dtype=tf.float32)) * strides

    ctr_x, ctr_y = tf.meshgrid(ctr_x, ctr_y)

    # 打平为1维,得到所有锚点的坐标
    ctr_x = tf.reshape(ctr_x, [-1])
    ctr_y = tf.reshape(ctr_y, [-1])
    #  (H*W,1,4)
    shifts = tf.expand_dims(tf.stack([ctr_y, ctr_x, ctr_y, ctr_x], axis=1), axis=1)
    # (1,anchor_num,4)
    base_anchors = tf.expand_dims(tf.constant(base_anchors, dtype=tf.float32), axis=0)

    # (H*W,anchor_num,4)
    anchors = shifts + base_anchors
    # 转为(H*W*anchor_num,4)
    anchors = tf.reshape(anchors, [-1, 4])
    # 丢弃越界的anchors;   步长*feature map的高度就是图像高度
    is_valid_anchors = tf.logical_and(tf.less_equal(anchors[:, 2], tf.cast(strides * H, tf.float32)),
                                      tf.logical_and(tf.less_equal(anchors[:, 3], tf.cast(strides * W, tf.float32)),
                                                     tf.logical_and(tf.greater_equal(anchors[:, 0], 0),
                                                                    tf.greater_equal(anchors[:, 1], 0))))
    return tf.reshape(anchors, [-1, 4]), is_valid_anchors


class Anchor(keras.layers.Layer):
    def __init__(self, heights_list=None, widths_list=None, feature_strides = None,
                 base_size=None, ratios=None, scales=None, **kwargs):
        """
        :param heights_list: [[20,34,54,...], [120, 140, 200, ...], [],...]
        :param widths_list: [[20,34,54,...], [120, 140, 200, ...], [],...]\
        :param feature_strides : [8, 16, ...]
        :param base_size: anchor的base_size,如：64
        :param ratios: 长宽比; 如 [1,1/2,2]
        :param scales: 缩放比: 如 [1,2,4]
        :param strides: 步长,一般为base_size的四分之一
        """
        self.heights_list = heights_list
        self.widths_list = widths_list
        self.feature_strides = feature_strides
        self.base_size = base_size
        self.ratios = ratios
        self.scales = scales
        # base anchors数量
        if heights_list:
            num_anchors = []
            for i in self.heights_list:
                num_anchors.append( len(i))
            self.num_anchors = num_anchors

            #TODO
        super(Anchor, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs：输入
        input[0]: 卷积层特征(锚点所在层)，shape：[batch_size,H,W,C]
        input[1]: 图像的元数据信息, shape: [batch_size, 12 ];
        inputs: [feature_map_1, feature_map_2, ...]basenet卷积后各特征图
        :param kwargs:
        :return:
        """
        features = inputs
        features_shape = tf.shape(features[0])
        anchors = []
        tags = []
        for i in range(len(features)):
            base_anchors = generate_anchors(self.heights_list[i], self.widths_list[i],
                                            self.base_size, self.ratios, self.scales)
            out = shift(features[i], self.feature_strides[i], base_anchors)
            anchors.append(out[0])
            tags.append(out[1])

        #连接
        anchors = tf.concat(anchors, axis = 0)
        tags = tf.concat(tags, axis =0)

        anchors = tf.tile(tf.expand_dims(anchors, axis=0), [features_shape[0], 1, 1])
        tags = tf.tile(tf.expand_dims(tags, axis=0), [features_shape[0], 1])

        '''
        base_anchors = generate_anchors(self.heights, self.widths, self.base_size, self.ratios, self.scales)
        anchors, anchors_tag = shift(features_shape[1:3], self.strides, base_anchors)
        # 扩展第一维，batch_size;每个样本都有相同的anchors
        anchors = tf.tile(tf.expand_dims(anchors, axis=0), [features_shape[0], 1, 1])
        anchors_tag = tf.tile(tf.expand_dims(anchors_tag, axis=0), [features_shape[0], 1])
        '''
        return [anchors, tags]

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: [batch_size,H,W,C]
        :return:
        """
        # 计算所有的anchors数量
        total = np.prod(input_shape[0][1:3]) * self.num_anchors[0] + np.prod(input_shape[1][1:3]) * self.num_anchors[1]
        # total = 49 * self.num_anchors
        return [(input_shape[0][0], total, 4),
                (input_shape[0][0], total)]


if __name__ == '__main__':
    sess = tf.Session()
    achrs = generate_anchors(64, [1], [1, 2, 4])
    print(achrs)
    all_achrs = shift([3, 3], 32, achrs)
    print(sess.run(tf.shape(all_achrs)))
    print(sess.run(all_achrs))