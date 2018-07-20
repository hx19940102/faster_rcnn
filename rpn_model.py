import tensorflow as tf
import tensorflow.contrib.slim as slim
import anchor_generator
import anchor_target_layer
import vgg_16
import utils


class regional_proposal_network(object):
    def __init__(self, is_training):
        self.vgg_16 = vgg_16.vgg_16(is_training=True)
        self.is_training = is_training

    def anchor_generator_layer(self, feature_map_shape, ratios, scales):
        with tf.variable_scope("anchor_generator_layer"):
            all_anchors = tf.py_func(anchor_generator.generator_anchor,
                                     [feature_map_shape, ratios, scales],
                                     tf.float32
                                     )
            all_anchors = tf.reshape(all_anchors, [-1, 4])
            return all_anchors

    def anchor_target_layer(self, gt_boxes, all_anchors, image_shape, feature_map_shape, k):
        with tf.variable_scope("anchor_target_layer"):
            labels, anchor_targets = tf.py_func(anchor_target_layer.anchor_target_layer,
                                                [gt_boxes, all_anchors, image_shape, feature_map_shape, k],
                                                [tf.int32, tf.float32])
            return labels, anchor_targets

    def image_2_feature_layer(self, image):
        feature_map = self.vgg_16.image_to_feature(image)
        return feature_map

    def smooth_l1_loss(self, rpn_proposals, anchor_targets):
        """  """
        with tf.variable_scope("smooth_l1_loss"):
            loss = tf.py_func(utils.smooth_l1_loss,
                              [rpn_proposals, anchor_targets],
                              tf.float32)
        return loss


    def build_model(self, image, gt_boxes):
        feature_map = self.image_2_feature_layer(image)
        feature_map_shape = tf.shape(feature_map)[1:3]
        image_shape = tf.shape(image)[1:3]

        anchors = self.anchor_generator_layer(feature_map_shape,
                                              tf.constant([0.5, 1, 2], tf.float32),
                                              tf.constant([8, 16, 32], tf.float32))
        if self.is_training is True:
            labels, anchor_targets = self.anchor_target_layer(gt_boxes, anchors, image_shape, feature_map_shape, tf.constant(9))

        with tf.variable_scope("regional_proposal_layer"):
            net = slim.conv2d(feature_map, 512, [3, 3], trainable=self.is_training,
                              padding="SAME", activation_fn=tf.nn.relu,
                              weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                              weights_regularizer=slim.l2_regularizer(0.0005),
                              biases_initializer=tf.zeros_initializer,
                              scope="rpn_conv")
            rpn_cls_score = slim.conv2d(net, 9 * 2, [1, 1], trainable=self.is_training,
                                        padding="SAME", activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005),
                                        biases_initializer=tf.zeros_initializer,
                                        scope="rpn_cls_score")
            rpn_bbox_pred = slim.conv2d(net, 9 * 4, [1, 1], trainable=self.is_training,
                                        padding="SAME", activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005),
                                        biases_initializer=tf.zeros_initializer,
                                        scope="rpn_bbox_pred")
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

        if self.is_training is True:
            vgg_variables = self.vgg_16.extract_variables()
            return labels, anchor_targets, rpn_cls_score, rpn_bbox_pred, vgg_variables
        else:
            return anchors, rpn_cls_score, rpn_bbox_pred, feature_map

    def loss_func(self, labels, anchor_targets, rpn_cls_score, rpn_bbox_pred):
        cls_valid_idx = tf.where(tf.not_equal(labels, -1))
        rpn_cls_score = tf.gather_nd(rpn_cls_score, cls_valid_idx)
        labels = tf.gather_nd(labels, cls_valid_idx)
        rpn_cls_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=labels),
            axis=0
        )
        reg_valid_idx = tf.where(tf.equal(labels, 1))
        rpn_bbox_pred = tf.gather_nd(rpn_bbox_pred, reg_valid_idx)
        anchor_targets = tf.gather_nd(anchor_targets, reg_valid_idx)
        rpn_bbox_loss = self.smooth_l1_loss(rpn_bbox_pred, anchor_targets)
        rpn_bbox_loss = tf.reduce_mean(rpn_bbox_loss)
        return rpn_cls_loss, rpn_bbox_loss