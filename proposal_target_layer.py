import numpy as np
import tensorflow as tf
import bbox

BATCH_SIZE = 128
FG_RATIO = 0.25

def proposal_target_layer(rpn_bbox, rpn_cls_prob, gt_boxes, num_classes):
    confidence_scores = rpn_cls_prob[:, 1]

    # Add ground truth boxes as part of proposals
    rpn_bbox = np.vstack([rpn_bbox, gt_boxes[:, 0:-1]])
    confidence_scores = np.concatenate(confidence_scores, np.ones(gt_boxes.shape[0], np.float32))

    # Sample objects and backgrounds
    fg_cnt = int(BATCH_SIZE * FG_RATIO)
    fg_idxs = np.where(confidence_scores >= 0.5)[0]
    if len(fg_idxs) > fg_cnt:
        pos_inds = np.random.choice(
            fg_idxs, size=fg_cnt, replace=False)
    bg_cnt = BATCH_SIZE - len(pos_inds)
    bg_idxs = np.where((confidence_scores >= 0.1) & (confidence_scores < 0.5))[0]
    if len(bg_idxs) > bg_cnt:
        neg_inds = np.random.choice(bg_idxs, size=bg_cnt, replace=False)

    pos_bbox = rpn_bbox[pos_inds]
    overlaps = bbox.bbox_overlaps(pos_bbox, gt_boxes[:,0:-1])
    argmax_overlaps = np.argmax(overlaps, axis=-1)
    pos_labels = gt_boxes[:,-1][argmax_overlaps] + 1
    neg_labels = np.zeros(len(neg_inds), np.int32)
    labels = np.concatenate(pos_labels, neg_labels)
    bbox_reg = np.zeros([len(labels), (num_classes + 1) * 4], np.float32)
    bbox_reg_ = bbox.bbox_transform(rpn_bbox, gt_boxes[argmax_overlaps][:,:-1])
    for i in range(len(pos_labels)):
        bbox_reg[i, pos_labels[i] * 4 : (pos_labels[i] + 1) * 4] = bbox_reg_[i]

    neg_bbox = rpn_bbox[neg_inds]
    rpn_bbox = np.vstack([pos_bbox, neg_bbox])

    return labels, bbox_reg, rpn_bbox


def smooth_l1_loss(labels, rcnn_bbox_reg, bbox_reg):
    if len(bbox_reg) == 0: return 0

    proposals = np.zeros((len(labels), 4), np.float32)
    targets = np.zeros((len(labels), 4), np.float32)
    for i in range(len(labels)):
        proposals[i, :] = rcnn_bbox_reg[i, labels[i] * 4 : (labels[i] + 1) * 4]
        targets[i, :] = bbox_reg[i, labels[i] * 4 : (labels[i] + 1) * 4]

    diff = proposals - targets
    diff = np.abs(diff)
    pos_idx = np.where(np.greater_equal(diff, 1))[0]
    neg_idx = np.where(np.less(diff, 1))[0]
    diff[pos_idx] = diff[pos_idx] - 0.5
    diff[neg_idx] = np.power(diff[neg_idx], 2) * 0.5
    loss = np.sum(diff, axis=-1)
    return loss