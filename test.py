import tensorflow as tf
import numpy as np
import rpn_model
import cv2
import bbox
import nms

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
para_dir = 'rpn.ckpt'
img_dir = '5.jpg'
k = 5

image = tf.placeholder(tf.uint8, [None, None, None, 3])
image = tf.cast(image, tf.float32)
image = image - [_R_MEAN, _G_MEAN, _B_MEAN]


rpn = rpn_model.regional_proposal_network(False)
anchors, rpn_cls_score, rpn_bbox_pred, _ = rpn.build_model(image, None)


# Initialization
local_vars_init_op = tf.local_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(local_vars_init_op)
    saver.restore(sess=sess, save_path=para_dir)
    img = cv2.imread(img_dir)
    anchors_, rpn_cls_score_, rpn_bbox_pred_ = sess.run([anchors, rpn_cls_score, rpn_bbox_pred], feed_dict={image:[img]})
    bbox = bbox.bbox_transform_inv(anchors_, rpn_bbox_pred_)

    # Remove boxes that are out of bound
    valid_idx = np.where((bbox[:, 0] >= 0) &
                         (bbox[:, 1] >= 0) &
                         (bbox[:, 2] < img.shape[1]) &
                         (bbox[:, 3] < img.shape[0]))[0]
    bbox = bbox[valid_idx]
    rpn_cls_score_ = rpn_cls_score_[valid_idx]

    # Remove background boxes
    valid_idx = np.where(rpn_cls_score_[:, 1] > rpn_cls_score_[:, 0])[0]
    bbox = bbox[valid_idx, :]
    rpn_cls_score_ = rpn_cls_score_[valid_idx]

    nms_idxs = nms.non_maximal_suppression(bbox, 0.5)
    bbox = bbox[nms_idxs, :]
    rpn_cls_score_ = rpn_cls_score_[nms_idxs]

    top_inds = rpn_cls_score_[:, 1].argsort(0)[::-1]
    top_inds = top_inds[0: k]
    top_bbox = bbox[top_inds, :]

    for bbox in top_bbox:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 0, 255))
    print top_bbox
    cv2.imshow("image", img)
    cv2.waitKey(0)