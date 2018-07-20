import xml.etree.ElementTree as et
import numpy as np


def read_annotation_from_file(filename):
    tree = et.parse(filename)
    root = tree.getroot()
    classes = {'person':0, 'bird':1, 'cat':2, 'cow':3, 'dog':4, 'horse':5, 'sheep':6,
               'aeroplane':7, 'bicycle':8, 'boat':9, 'bus':10, 'car':11, 'motorbike':12, 'train':13,
               'bottle':14, 'chair':15, 'dining table':16, 'potted plant':17, 'sofa':18, 'monitor':19}
    objects = []
    for child in root:
        if child.tag == 'filename':
            image_dir = child.text
        elif child.tag == 'size':
            for t in child:
                if t.tag == 'width': width = t.text
                elif t.tag == 'height': height = t.text
                elif t.tag == 'depth': depth = t.text
            image_shape = (int(height), int(width), int(depth))
        elif child.tag == 'object':
            for t in child:
                if t.tag == 'name': name = t.text
                elif t.tag == 'bndbox': bndbox = t
            for b in bndbox:
                if b.tag == 'xmin': x1 = int(float(b.text))
                elif b.tag == 'xmax': x2 = int(float(b.text))
                elif b.tag == 'ymin': y1 = int(float(b.text))
                elif b.tag == 'ymax': y2 = int(float(b.text))
            object = [x1, y1, x2, y2]
            if name in classes:
                object.append(classes[name])
            else:
                object.append(len(classes))
            objects.append(object)
    return image_dir, image_shape, objects


def smooth_l1_loss(rpn_proposals, anchor_targets):
    if len(rpn_proposals) == 0: return 0
    diff = rpn_proposals - anchor_targets
    diff = np.abs(diff)
    pos_idx = np.where(np.greater_equal(diff, 1))[0]
    neg_idx = np.where(np.less(diff, 1))[0]
    diff[pos_idx] = diff[pos_idx] - 0.5
    diff[neg_idx] = np.power(diff[neg_idx], 2) * 0.5
    loss = np.sum(diff, axis=-1)
    return loss