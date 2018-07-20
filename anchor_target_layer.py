import bbox
import numpy as np


BG_LOW_THRES = 0.3
FG_HIGH_THRES = 0.7
SAMPLE_NUMBER = 256
FG_RATIO = 0.5


def anchor_target_layer(gt_boxes, all_anchors, image_shape, feature_map_shape, k):
    """
    
    :param gt_boxes: 
    :param all_anchors: 
    :param image_shape: 
    :param feature_map_shape: 
    :param k: 
    :return: 
    """

    # If there is no object in the image
    if len(gt_boxes) == 0:
        labels = np.zeros((len(all_anchors),), dtype=np.int32)
        targets = np.zeros(all_anchors.shape, dtype=np.float32)
        return labels, targets

    num_total_anchors = all_anchors.shape[0]

    # Keep anchors that inside the image
    valid_idx = np.where((all_anchors[:, 0] >= 0) &
                         (all_anchors[:, 1] >= 0) &
                         (all_anchors[:, 2] < image_shape[1]) &
                         (all_anchors[:, 3] < image_shape[0]))[0]

    anchors = all_anchors[valid_idx, :]

    labels = np.empty((len(valid_idx),), dtype=np.int32)
    labels.fill(-1)

    overlaps = bbox.bbox_overlaps(anchors, gt_boxes)
    argmax_overlaps = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(0, len(valid_idx), 1), argmax_overlaps]
    gt_argmax_overlaps = np.argmax(overlaps, axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(0, gt_boxes.shape[0], 1)]
    gt_max_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    labels[np.where(max_overlaps < BG_LOW_THRES)[0]] = 0
    labels[gt_max_overlaps] = 1
    labels[np.where(max_overlaps > FG_HIGH_THRES)[0]] = 1

    targets = bbox.bbox_transform(anchors, gt_boxes[argmax_overlaps, :])

    # Sampling positive and negative anchors
    fg_cnt = int(SAMPLE_NUMBER * FG_RATIO)
    fg_idxs = np.where(labels == 1)[0]
    if len(fg_idxs) > fg_cnt:
        disable_inds = np.random.choice(
            fg_idxs, size=(len(fg_idxs) - fg_cnt), replace=False)
        labels[disable_inds] = -1

    bg_cnt = SAMPLE_NUMBER - np.sum(labels == 1)
    bg_idxs = np.where(labels == 0)[0]
    if len(bg_idxs) > bg_cnt:
        disable_inds = np.random.choice(
            bg_idxs, size=(len(bg_idxs) - bg_cnt), replace=False)
        labels[disable_inds] = -1

    labels = _unmap(labels, num_total_anchors, valid_idx, -1)
    targets = _unmap(targets, num_total_anchors, valid_idx, 0)

    return labels, targets


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.int32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret