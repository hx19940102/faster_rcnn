import tensorflow as tf
import tensorflow.contrib.slim as slim


class vgg_16(object):
    def __init__(self, is_training, dropout_keep_prob=0.5):
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob

    def image_to_feature(self, images):
        with tf.variable_scope("vgg_16"):
            with slim.arg_scope([slim.conv2d],
                                activation_fn = tf.nn.relu,
                                weights_initializer = tf.truncated_normal_initializer(0, 0.01),
                                biases_initializer = tf.zeros_initializer,
                                weights_regularizer = slim.l2_regularizer(0.0005),
                                padding = 'SAME',
                                trainable = self.is_training):
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                feature_map = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                # max_pooling 5 is converted to ROI_pooling layer
                # net = slim.max_pool2d(net, [2, 2], scope='pool5')
                return feature_map

    def feature_to_tail(self, roi):
        with tf.variable_scope("vgg_16_tail"):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                                biases_initializer=tf.zeros_initializer,
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding = 'VALID',
                                trainable = self.is_training):
                net = slim.conv2d(roi, 4096, [7, 7], scope='fc6')
                net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training, scope='dropout7')
                # logits layer is replaced by fast_rcnn cls layer
                # net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
                return net

    def extract_variables(self):
        variables = slim.get_variables('vgg_16')
        vgg_variables = {}
        for variable in variables:
            if 'tail' not in variable.name:
                vgg_variables[variable.name[0:-2]] = variable
        return vgg_variables
