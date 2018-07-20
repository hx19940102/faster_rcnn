import numpy as np
import tensorflow as tf


def bbox_overlaps(anchors, gt_boxes):
    m = anchors.shape[0]
    n = gt_boxes.shape[0]
    overlaps = np.zeros((m, n), dtype=np.float32)
    for i in range(n):
        gt_box_area = (gt_boxes[i][2] - gt_boxes[i][0] + 1) * (gt_boxes[i][3] - gt_boxes[i][1] + 1)
        for j in range(m):
            w = min(gt_boxes[i][2], anchors[j][2]) - max(gt_boxes[i][0], anchors[j][0]) + 1
            h = min(gt_boxes[i][3], anchors[j][3]) - max(gt_boxes[i][1], anchors[j][1]) + 1
            if w > 0 and h > 0:
                overlap = w * h
                total_area = (anchors[j][2] - anchors[j][0] + 1) * (anchors[j][3] - anchors[j][1] + 1) + gt_box_area - overlap
                overlaps[j][i] = overlap / total_area
    return overlaps


def bbox_transform(anchors, gt_boxes):
    width_anchors = anchors[:, 2] - anchors[:, 0] + 1.0
    height_anchors = anchors[:, 3] - anchors[:, 1] + 1.0
    x_center_anchors = anchors[:, 0] + 0.5 * width_anchors
    y_center_anchors = anchors[:, 1] + 0.5 * height_anchors

    width_boxes = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    height_boxes = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    x_center_boxes = gt_boxes[:, 0] + 0.5 * width_boxes
    y_center_boxes = gt_boxes[:, 1] + 0.5 * height_boxes

    x_center = np.divide((x_center_boxes - x_center_anchors), width_anchors)
    y_center = np.divide((y_center_boxes - y_center_anchors), height_anchors)
    width = np.log(np.divide(width_boxes, width_anchors))
    height = np.log(np.divide(height_boxes, height_anchors))

    return np.vstack((x_center, y_center, width, height)).transpose()


def bbox_transform_inv(anchors, proposals):
    width_anchors = anchors[:, 2] - anchors[:, 0] + 1.0
    height_anchors = anchors[:, 3] - anchors[:, 1] + 1.0
    x_center_anchors = anchors[:, 0] + 0.5 * width_anchors
    y_center_anchors = anchors[:, 1] + 0.5 * height_anchors

    x_center_boxes = proposals[:, 0] * width_anchors + x_center_anchors
    y_center_boxes = proposals[:, 1] * height_anchors + y_center_anchors
    width_boxes = np.exp(proposals[:, 2]) * width_anchors
    height_boxes = np.exp(proposals[:, 3]) * height_anchors

    return np.vstack((x_center_boxes - width_boxes * 0.5,
                      y_center_boxes - height_boxes * 0.5,
                      x_center_boxes + width_boxes * 0.5,
                      y_center_boxes + height_anchors * 0.5)).transpose()


def clip_boxes(boxes, im_shape):
  """
  Clip boxes to image boundaries.
  """

  # x1 >= 0
  boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
  return boxes


def remove_out_bound(bbox, image_shape):
    valid_idxs = np.where((bbox[:, 0] >= 0) &
                          (bbox[:, 1] >= 0) &
                          (bbox[:, 2] < image_shape[1]) &
                          (bbox[:, 3] < image_shape[0]))[0]
    return valid_idxs