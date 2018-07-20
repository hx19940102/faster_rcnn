import numpy as np
import tensorflow as tf
import bbox


def proposal_top_layer(rpn_cls_score, rpn_bbox_pred, image_shape, all_anchors, num_anchors, N):
    rpn_cls_prob = np.exp(rpn_cls_score) / np.sum(np.exp(rpn_cls_score), axis=-1)
    confidence_scores = rpn_cls_prob[:, 1]
    if confidence_scores.shape[0] < N:
    # Not enough region proposals
    # TODO: deal with this case
        top_inds = []
    else:
        top_inds = confidence_scores.argsort(0)[::-1]
        top_inds = top_inds[0: N]
        top_inds = np.squeeze(top_inds, axis=-1)

    anchors = all_anchors[top_inds, :]
    scores = confidence_scores[top_inds, :]
    rpn_bbox_pred = rpn_bbox_pred[top_inds, :]

    proposals = bbox.bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = bbox.clip_boxes(proposals, image_shape)

    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob, scores

def softmax_layer(rpn_cls_score, num_anchors):
    """
    Convert scores to probabilities using softmax function
    :param rpn_cls_score: (1, height, width, num_anchors * 2)
    :param num_anchors: 
    :return: rpn_cls_prob: (1, height, width, num_anchors * 2)
             rpn_cls_pred: (1, height, width, num_anchors
    """
    rpn_cls_prob = np.zeros(rpn_cls_score.shape, np.float32)
    rpn_cls_pred = np.zeros(np.concatenate((rpn_cls_score.shape[:-1], [num_anchors]), axis=0), tf.float32)
    for i in range(num_anchors):
        bg_scores = rpn_cls_score[:, :, :, i]
        fg_scores = rpn_cls_score[:, :, :, i + num_anchors]
        bg_probs = np.divide(np.exp(bg_scores), np.exp(bg_scores) + np.exp(fg_scores))
        fg_probs = np.divide(np.exp(fg_scores), np.exp(bg_scores) + np.exp(fg_scores))
        rpn_cls_prob[:, :, :, i] = bg_probs
        rpn_cls_prob[:, :, :, i + num_anchors] = fg_probs
        rpn_cls_pred[:, :, :, i] = fg_probs > bg_probs
    return rpn_cls_prob, rpn_cls_pred