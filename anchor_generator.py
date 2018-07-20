import numpy as np


def decode_anchor(anchor):
    """
    Translate anchor from [x0, y0, x1, y1] to [x_center, y_center, width, height]
    :param anchor: [x0, y0, x1, y1]
    :return: [x_c, y_c, width, height]
    """
    width = anchor[2] - anchor[0] + 1
    height = anchor[3] - anchor[1] + 1
    x_center = anchor[0] + (width - 1) * 0.5
    y_center = anchor[1] + (height - 1) * 0.5
    return x_center, y_center, width, height

def encode_anchor(x_centers, y_centers, widths, heights):
    """
    Translate anchors from [x_c, y_c, width, height] to [x0, y0, x1, y1]
    :param x_centers: a list of x_center's
    :param y_centers: a list of y_cetner's
    :param widths: a list of widths
    :param heights: a list of heights
    :return: a list of [x0, y0, x1, y1]
    """
    x0s = x_centers - (widths - 1) * 0.5
    y0s = y_centers - (heights - 1) * 0.5
    x1s = x_centers + (widths - 1) * 0.5
    y1s = y_centers + (heights - 1) * 0.5
    return np.hstack((np.expand_dims(x0s, -1),
                      np.expand_dims(y0s, - 1),
                      np.expand_dims(x1s, -1),
                      np.expand_dims(y1s, - 1)))

def ratio_enum(anchor, ratios):
    """
    Based on the anchor [x0, y0, x1, y1], generate a list of anchors with 
    same area but different ratios(height / width).
       height / width = ratio
       height * width = area
    -> width = sqrt(area / ratio)
    -> height = sqrt(area * ratio)
    :param anchor: [x0, y0, x1, y1]
    :param ratios: [ratio0, ratio1, ...]
    :return: a list of anchors [[x0, y0, x1, y1], ...]
    """
    x_c, y_c, width, height = decode_anchor(anchor)
    area = width * height
    widths = np.round(np.sqrt(area / ratios))
    heights = np.round(np.sqrt(area * ratios))
    return encode_anchor(x_c, y_c, widths, heights)

def scale_enum(anchor, scales):
    """
    Based on anchor, generate a list of anchors with different scales
    :param anchor: [x0, y0, x1, y1]
    :param scales: [s0, s1, ...]
    :return: a list of anchors [[x0, y0, x1, y1], ...]
    """
    x_center, y_center, width, height = decode_anchor(anchor)
    widths = scales * width
    heights = scales * height
    return encode_anchor(x_center, y_center, widths, heights)

def generator_anchor(feature_map_shape, ratios, scales):
    """
    Based on feature_map size, generate corresponding anchors with different ratios and scales
    :param feature_map_shape: 
    :param ratios: 
    :param scales: 
    :return: 
    """
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([8, 16, 32])
    x0s = np.arange(0, feature_map_shape[1])
    y0s = np.arange(0, feature_map_shape[0])
    x1s = x0s + 1
    y1s = y0s + 1
    x0s = np.tile(x0s, [feature_map_shape[0], 1])
    y0s = np.tile(np.expand_dims(y0s, -1), [1, feature_map_shape[1]])
    x1s = np.tile(x1s, [feature_map_shape[0], 1])
    y1s = np.tile(np.expand_dims(y1s, -1), [1, feature_map_shape[1]])
    base_anchors = np.stack([x0s, y0s, x1s, y1s], axis=2)
    base_anchors[:, :, 0] = 16 * base_anchors[:, :, 0]
    base_anchors[:, :, 1] = 16 * base_anchors[:, :, 1]
    base_anchors[:, :, 2] = 16 * base_anchors[:, :, 2] - 1
    base_anchors[:, :, 3] = 16 * base_anchors[:, :, 3] - 1
    all_anchors = np.zeros([feature_map_shape[0], feature_map_shape[1], len(ratios) * len(scales), 4], np.float32)
    for i in range(base_anchors.shape[0]):
        for j in range(base_anchors.shape[1]):
            anchors = ratio_enum(base_anchors[i][j], ratios)
            anchors = np.vstack((scale_enum(anchor, scales) for anchor in anchors))
            all_anchors[i][j] = anchors
    return all_anchors