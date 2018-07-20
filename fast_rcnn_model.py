import tensorflow as tf
import tensorflow.contrib.slim as slim
import vgg_16
import proposal_target_layer
import nms
import bbox


NMS_THRES = 0.5


class fast_rcnn(object):
    def __init__(self, is_training, num_of_classes):
        self.is_training = is_training
        self.vgg_16 = vgg_16.vgg_16(is_training=True)
        self.num_of_classes = num_of_classes

    def proposal_target_layer(self, rpn_bbox, rpn_cls_prob, gt_boxes):
        with tf.variable_scope('proposal_target_layer'):
            labels, bbox_reg, rpn_bbox = tf.py_func(proposal_target_layer.proposal_target_layer,
                                                    [rpn_bbox, rpn_cls_prob, gt_boxes, self.num_of_classes],
                                                    [tf.int32, tf.float32, tf.float32])
        return labels, bbox_reg, rpn_bbox

    def non_maximal_suppression(self, rpn_bbox, overlap_thres):
        with tf.variable_scope('non_maximal_suppresion'):
            nms_idxs = tf.py_func(nms.non_maximal_suppression,
                                 [rpn_bbox, overlap_thres],
                                 [tf.int64])
        return nms_idxs

    def roi_pooling_layer(self, feature_map, image_shape, rois):
        with tf.variable_scope('roi_pooling_layer'):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            height = tf.cast(image_shape[0], tf.float32)
            width = tf.cast(image_shape[1], tf.float32)
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            crops = tf.image.crop_and_resize(feature_map, bboxes, tf.to_int32(batch_ids), [7, 7],
                                             name="crops")

        return crops

    def bbox_transform_inv(self, anchors, rpn_bbox_reg):
        with tf.variable_scope('bbox_transform_inv'):
            rpn_bbox = tf.py_func(bbox.bbox_transform_inv,
                                  [anchors, rpn_bbox_reg],
                                  [tf.float32])
        return rpn_bbox

    def remove_out_bound(self, rpn_bbox, image_shape):
        with tf.variable_scope('remove_out_bound'):
            valid_idxs = tf.py_func(bbox.remove_out_bound,
                                    [rpn_bbox, image_shape],
                                    tf.int64)
        return valid_idxs

    def smooth_l1_loss(self, labels, rcnn_bbox_reg, bbox_reg):
        with tf.variable_scope("rcnn_l1_loss"):
            loss = tf.py_func(proposal_target_layer.smooth_l1_loss,
                              [labels, rcnn_bbox_reg, bbox_reg],
                              [tf.float32])
            return loss

    def loss_func(self, rcnn_cls_score, rcnn_bbox_reg, labels, bbox_reg):
        rcnn_cls_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=rcnn_cls_score)
            )
        fg_idxs = tf.where(tf.not_equal(labels, 0))
        rcnn_bbox_reg = tf.gather_nd(rcnn_bbox_reg, fg_idxs)
        bbox_reg = tf.gather_nd(bbox_reg, fg_idxs)
        fg_labels = tf.gather_nd(labels, fg_idxs)
        rcnn_reg_loss = self.smooth_l1_loss(fg_labels, rcnn_bbox_reg, bbox_reg)
        rcnn_bbox_reg = tf.reduce_mean(rcnn_reg_loss)
        return rcnn_cls_loss, rcnn_bbox_reg

    def extract_vgg_variables(self):
        variables = slim.get_variables('vgg_16_tail')
        vgg_variables = {}
        for variable in variables:
            vgg_variables['vgg_16' + variable.name[11:-2]] = variable
        return vgg_variables

    def build_model(self, image_shape, feature_map, rpn_cls_score, rpn_bbox_reg, anchors, gt_boxes):
        rpn_bbox = self.bbox_transform_inv(anchors, rpn_bbox_reg)

        # Remove out-bounders
        #valid_idxs = self.remove_out_bound(rpn_bbox, image_shape)
        #rpn_bbox = tf.gather_nd(rpn_bbox, valid_idxs)
        #rpn_cls_score = tf.gather_nd(rpn_cls_score, valid_idxs)

        rpn_cls_prob = tf.divide(tf.exp(rpn_cls_score),
                                 tf.expand_dims(tf.reduce_sum(tf.exp(rpn_cls_score), axis=-1), axis=-1))
        rpn_cls = tf.argmax(rpn_cls_prob, 1)
        pos_idxs = tf.where(tf.equal(rpn_cls, 1))
        neg_idxs = tf.where(tf.equal(rpn_cls, 0))
        fg_bbox = tf.gather_nd(rpn_bbox, pos_idxs)
        fg_cls_prob = tf.gather_nd(rpn_cls_prob, pos_idxs)
        bg_bbox = tf.gather_nd(rpn_bbox, neg_idxs)
        bg_cls_prob = tf.gather_nd(rpn_cls_prob, neg_idxs)

        nms_idxs = self.non_maximal_suppression(fg_bbox, tf.constant(NMS_THRES, tf.float32))
        fg_bbox = tf.gather_nd(fg_bbox, nms_idxs)
        fg_cls_prob = tf.gather_nd(fg_cls_prob, nms_idxs)

        if self.is_training is True:
            all_bbox = tf.stack((fg_bbox, bg_bbox))
            all_cls_prob = tf.stack((fg_cls_prob, bg_cls_prob))
            labels, bbox_reg, rpn_bbox = self.proposal_target_layer(all_bbox, all_cls_prob, gt_boxes)
            rpn_bbox_with_batch_id = tf.concat((tf.zeros((tf.shape(rpn_bbox)[0], 1), tf.float32), rpn_bbox), axis=-1)
            rois = self.roi_pooling_layer(feature_map, image_shape, rpn_bbox_with_batch_id)
            feature_vector = self.vgg_16.feature_to_tail(rois)

            with tf.variable_scope('fast_rcnn'):
                rcnn_cls_score = slim.conv2d(feature_vector, self.num_of_classes + 1, [1, 1],
                                             activation_fn=None,
                                             weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                                             biases_initializer=tf.zeros_initializer,
                                             weights_regularizer=slim.l2_regularizer(0.0005),
                                             padding='VALID',
                                             trainable=self.is_training,
                                             scope='rcnn_cls_layer'
                                             )

                rcnn_bbox_reg = slim.conv2d(feature_vector, 4 * (self.num_of_classes + 1), [1, 1],
                                             activation_fn=None,
                                             weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                                             biases_initializer=tf.zeros_initializer,
                                             weights_regularizer=slim.l2_regularizer(0.0005),
                                             padding='VALID',
                                             trainable=self.is_training,
                                             scope='rcnn_reg_layer'
                                             )
            vgg_16_variables = self.extract_vgg_variables()
            return vgg_16_variables, rcnn_cls_score, rcnn_bbox_reg, labels, bbox_reg, rpn_bbox


