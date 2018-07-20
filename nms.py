import numpy as np


def non_maximal_suppression(bbox, overlap_thres):
    if len(bbox) == 0:
        return []
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    area = np.multiply((x2 - x1 + 1), (y2 - y1 + 1))
    idxs = np.argsort(y2, axis=0)
    res = np.array([], np.int32)
    while len(idxs) > 0:
        idx = idxs[-1]
        x1_ = np.maximum(x1[idx], x1[idxs[:-1]])
        y1_ = np.maximum(y1[idx], y1[idxs[:-1]])
        x2_ = np.minimum(x2[idx], x2[idxs[:-1]])
        y2_ = np.minimum(y2[idx], y2[idxs[:-1]])
        w = np.maximum(0, x2_ - x1_ + 1)
        h = np.maximum(0, y2_ - y1_ + 1)
        overlap = np.multiply(w, h) / area[idxs[:-1]]
        res = np.hstack((res, np.array([idx], np.int32)))
        idxs = np.delete(idxs, [len(idxs) - 1])
        idxs = np.delete(idxs, np.where(overlap > overlap_thres)[0])
    return res