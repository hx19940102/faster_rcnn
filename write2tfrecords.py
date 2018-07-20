import tensorflow as tf
import os
import utils
import cv2
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


annotation_prefix = 'VOC2012/Annotations/'
image_prefix = 'VOC2012/JPEGImages/'

annotations = [annotation_prefix + dir for dir in os.listdir(annotation_prefix)]

tf_writer =tf.python_io.TFRecordWriter('voc2012.tfrecords')

for annotation_dir in annotations:
    image_dir, image_shape, objects = utils.read_annotation_from_file(annotation_dir)
    image = cv2.imread(image_prefix + image_dir)
    image_raw = image.tostring()
    cnt_of_objects = len(objects)
    objects_raw = []
    for object in objects: objects_raw.extend(object)
    height, width, depth = image_shape[0], image_shape[1], image_shape[2]

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            'cnt_of_objects': _int64_feature(cnt_of_objects),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'image_raw': _bytes_feature(image_raw)
        }),
        feature_lists=tf.train.FeatureLists(feature_list={
            'objects_raw': _int64_feature_list(objects_raw)
        })
    )

    tf_writer.write(example.SerializeToString())

tf_writer.close()