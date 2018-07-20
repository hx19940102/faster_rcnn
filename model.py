import tensorflow as tf
import tensorflow.contrib.slim as slim
import resnet_v1
import resnet_utils
import vgg_16
import anchor_generator
import anchor_target_layer
import proposal_top_layer
import proposal_target_layer


class rpn_fast_rcnn(object):
    def __init__(self, is_training):
        self.is_training = is_training
        self.num_anchors = 9
        self.all_anchors = None
        self.image_shape = None
        self.ratios = tf.constant([0.5, 1.0, 2.0], tf.float32)
        self.scales = tf.constant([8., 16. ,32.], tf.float32)
        self.gt_boxes = None
        self.feature_map_shape = None
        self.feature_map = None
        self.rpn_cls_prob = None
        self.rpn_bbox_pred = None
        self.rois = None
        self.N = None

    def anchor_generator_layer(self):
        with tf.variable_scope("anchor_generator_layer"):
            all_anchors = tf.py_func(anchor_generator.generator_anchor,
                                          [self.feature_map_shape, self.ratios, self.scales],
                                          tf.float32)
            return all_anchors

    def anchor_target_layer(self):
        with tf.varaible_scope("anchor_target_layer"):
            labels, targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(anchor_target_layer.anchor_target_layer,
                                                                                    [self.gt_boxes, self.all_anchors,
                                                                                     self.image_shape, self.feature_map_shape,
                                                                                     self.num_anchors],
                                                                                    [tf.float32, tf.float32, tf.float32, tf.float32])
            return labels, targets, bbox_inside_weights, bbox_outside_weights

    def proposal_layer(self):
        # TODO: use non-maxmimal suppression to reduce number of region proposals
        return

    def proposal_top_layer(self):
        with tf.variable_scope("proposal_top_layer"):
            proposals, scores = tf.py_func(proposal_top_layer.proposal_top_layer,
                                           [self.rpn_cls_prob, self.rpn_bbox_pred,
                                            self.image_shape, self.all_anchors, self.num_anchors,
                                            self.N],
                                           [tf.float32, tf.float32])
            return proposals, scores

    def proposal_target_layer(self):
        with tf.variable_scope("proposal_target_layer"):
            rois, roi_scores, labels, bbox_targets, \
            bbox_inside_weights, bbox_outside_weights = tf.py_func(proposal_target_layer.proposal_target_layer,
                                                                   [self.rpn_proposals, self.rpn_proposal_scores,
                                                                    self.gt_boxes, self.num_classes])
            return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def softmax_layer(self):
        with tf.variable_scope("softmax_layer"):
            rpn_cls_prob, rpn_cls_pred = tf.py_func(proposal_top_layer.softmax_layer,
                                      [self.rpn_cls_score, self.num_anchors],
                                      [tf.float32, tf.float32])
            return rpn_cls_prob, rpn_cls_pred

    # TODO: build residual network here
    def feature_map_layer_res(self, image):
        return

    def vgg_network(self, image):
        """ VGG-16 network """
        self.vgg_16 = vgg_16.vgg_16(is_training=self.is_training)


    def region_proposal_layer(self):
        """ Regional Proposal Network """
        self.feature_map = self.vgg_16.image_to_feature(self.image)
        with tf.variable_scope("regional_proposal_layer"):
            net = slim.conv2d(self.feature_map, 512, [3, 3], trainable=self.is_training,
                              padding="SAME", activation_fn=tf.nn.relu,
                              weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                              weights_regularizer=slim.l2_regularizer(0.0005),
                              biases_initializer=tf.zeros_initializer,
                              scope="rpn_conv")
            self.rpn_cls_score = slim.conv2d(net, self.num_anchors * 2, [1, 1], trainable=self.is_training,
                                            padding="SAME", activation_fn=None,
                                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                            weights_regularizer=slim.l2_regularizer(0.0005),
                                            biases_initializer=tf.zeros_initializer,
                                            scope="rpn_cls_score")
            self.rpn_bbox_pred = slim.conv2d(net, self.num_anchors * 4, [1, 1], trainable=self.is_training,
                                             padding="SAME", activation_fn=None,
                                             weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                             weights_regularizer=slim.l2_regularizer(0.0005),
                                             biases_initializer=tf.zeros_initializer,
                                             scope="rpn_bbox_pred")
            # Convert classification sores to probability using softmax layer
            self.rpn_cls_prob, self.rpn_cls_pred = self.softmax_layer()


            if self.is_training is True:
                # For training, sample both fg and bg region proposals
                self.rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = self.proposal_target_layer()
            else:
                # Select top region proposals with highest foreground scores
                self.rpn_top_proposals, self.rpn_top_proposal_scores = self.proposal_top_layer()

    def roi_pooling_layer(self, rois, crop_size):
        """ Crop and resize layer to replace orginal ROI-pooling layer """
        with tf.variable_scope("roi_pooling_layer"):
            batch_idxes = tf.slice(rois, [0, 0], [-1, 1])
            batch_idxes = tf.cast(tf.squeeze(batch_idxes, axis=-1), tf.int32)
            x0s = tf.slice(rois, [0, 1], [-1, 1]) / tf.cast(self.image_shape[1], tf.float32)
            y0s = tf.slice(rois, [0, 2], [-1, 1]) / tf.cast(self.image_shape[0], tf.float32)
            x1s = tf.slice(rois, [0, 3], [-1, 1]) / tf.cast(self.image_shape[1], tf.float32)
            y1s = tf.slice(rois, [0, 4], [-1, 1]) / tf.cast(self.image_shape[0], tf.float32)
            rois = tf.concat([y0s, x0s, y1s, x1s], axis=1)
            rois = tf.image.crop_and_resize(self.feature_map, rois, batch_idxes, crop_size * 2)

            return slim.max_pool2d(rois, [2, 2], padding='SAME', scope='pool5')


    def fast_rcnn(self):
        """ Fast RCNN network """
        if self.is_training is True:
            rois = self.rois
        else:
            rois = self.rpn_proposals
        rois = self.roi_pooling_layer(rois, self.crop_size)
        feature_vectors = self.vgg_16.feature_to_tail(rois)
        with tf.variable_scope("fast_rcnn"):
            roi_cls_score = slim.conv2d(feature_vectors, self.num_classes, [1, 1], padding='SAME', activation_fn=None,
                                  weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                  biases_initializer=tf.zeros_initializer,
                                  scope='rcnn_cls_score')
            roi_reg_pred = slim.conv2d(feature_vectors, self.num_classes * 4, [1, 1], padding='SAME', activation_fn=None,
                                  weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                  biases_initializer=tf.zeros_initializer,
                                  scope='rcnn_reg_pred')
            roi_cls_prob = slim.nn.softmax(roi_cls_score, -1, name='softmax')

            return roi_cls_prob, roi_reg_pred


    def rpn_loss(self):
        """ Combined loss for RPN """
        valid_idx = tf.where(tf.not_equal(self.anchor_labels, -1))
        anchor_labels = tf.gather_nd(self.anchor_labels, valid_idx)
        anchor_targets = tf.gather_nd(self.anchor_targets, valid_idx)
        rpn_fg_prob = tf.reshape(self.rpn_cls_prob[:, :, :, self.num_anchors:], [-1, 1])
        rpn_bg_prob = tf.reshape(self.rpn_cls_prob[:, :, :, 0:self.num_anchors], [-1, 1])
        rpn_logits = tf.concat([rpn_bg_prob, rpn_fg_prob], axis=-1)
        rpn_logits = tf.gather_nd(rpn_logits, valid_idx)
        rpn_proposals = tf.gather_nd(tf.reshape(self.rpn_bbox_pred, [-1, 4]), valid_idx)
        rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_logits, anchor_labels))
        rpn_reg_loss = tf.reduce_sum(self.smooth_l1_loss(rpn_proposals, anchor_targets) * anchor_labels) / tf.reduce_sum(anchor_labels)
        rpn_combined_loss = rpn_cls_loss + rpn_reg_loss
        return rpn_combined_loss

    def rcnn_loss(self):
        """ Combined loss for R-CNN """


    def smooth_l1_loss(self, rpn_proposals, anchor_targets):
        """  """
        diff = rpn_proposals - anchor_targets
        diff = tf.abs(diff)
        idx_pos = tf.where(tf.greater_equal(diff, 1))
        idx_neg = tf.where(tf.less(diff, 1))
        diff[idx_pos] = diff[idx_pos] - 0.5
        diff[idx_neg] = tf.pow(diff[idx_neg], 2) * 0.5
        loss = tf.reduce_sum(diff, axis=-1)
        return loss
