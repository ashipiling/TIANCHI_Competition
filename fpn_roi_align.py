from keras import layers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class RoiAlign(layers.Layer):
    """
    将proposal边框投影到最后一层feature map上，并池化为7*7
    """

    def __init__(self, image_max_dim, pool_size=(7, 7), **kwargs):
        self.pool_size = pool_size
        self.image_max_dim = image_max_dim
        super(RoiAlign, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        inputs[0]: feature maps  [batch_num,H,W,feature_channel_num]
        inputs[1]: rois   [batch_num,roi_num,(y1,x1,y2,x2,tag)] , 训练和测试时，roi_num不同
        :param kwargs:
        :return:
        """
        features_list = inputs[:2]
        rois = inputs[2][..., :-1]  # 去除tag列
        batch_size, roi_num = rois.shape.as_list()[:2]
        y1, x1, y2, x2 = tf.split(rois, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        index = tf.where(tf.logical_or(h > 48., w > 48.))
        index = tf.cast( index[..., :-1], dtype= tf.int32)
        level = tf.zeros(shape=(tf.shape(h)[0], tf.shape(h)[1]))
        val = tf.ones(shape=(tf.shape(index)[0]))
        level = tf.scatter_nd( index, val, tf.shape(level))

        box_to_level = []
        pooled = []
        for i in range(len(features_list)):
            ix = tf.where(tf.equal(level, i))
            level_boxes = tf.gather_nd(rois, ix)/ tf.constant(self.image_max_dim, dtype=tf.float32)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            pooled.append(tf.image.crop_and_resize(
                features_list[i], level_boxes, box_indices, self.pool_size,
                method="bilinear"))
        pooled = tf.concat(pooled, axis = 0)
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)
        shape = pooled.shape.as_list()
        output = tf.reshape(pooled, [batch_size, roi_num] + shape[1:4], name='roi_align_output')
        print(self.pool_size)
        return output



    def compute_output_shape(self, input_shape):
        channel_num = input_shape[0][-1]  # feature通道数
        return input_shape[2][:2] + self.pool_size + (channel_num,)  # (batch_size,roi_num,h,w,channels)


def main():
    x = tf.expand_dims(tf.range(2), axis=1)
    y = tf.tile(x, [1, 3])
    sess = tf.Session()
    print(sess.run(tf.reshape(y, [-1])))
    print(sess.run(x))
    print(sess.run(x[:100]))
    m = tf.keras.models.Sequential()
    m.add(RoiAlign(608, (14, 14)), input_shape=(4, 300, 24, 24))
    m.summary()


if __name__ == '__main__':
    main()
