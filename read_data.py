import tensorflow as tf

def read_data_from_tfrecords(data_dir, is_training, batch_size):
    # Build the filename queue
    if is_training is True:
        filename_queue = tf.train.string_input_producer(data_dir, shuffle=True)
    else:
        filename_queue = tf.train.string_input_producer(data_dir, num_epochs=1)

    reader = tf.TFRecordReader()
    # Read the sequence example from tfrecords file
    _, serialized_example = reader.read(filename_queue)

    # Parse the sequence example back to raw data
    context, sequence = tf.parse_single_sequence_example(
        serialized_example,
        context_features={
            'cnt_of_objects': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        },
        sequence_features={
            'objects_raw': tf.FixedLenSequenceFeature([], dtype=tf.int64)
        })

    cnt_of_objects = context['cnt_of_objects']
    height = context['height']
    width = context['width']
    depth = context['depth']
    image_raw = context['image_raw']
    objects_raw = sequence['objects_raw']

    # Decode and process raw data
    cnt_of_objects = tf.cast(cnt_of_objects, tf.int32)
    height = tf.cast(height, tf.int32)
    width = tf.cast(width, tf.int32)
    depth = tf.cast(depth, tf.int32)
    image = tf.decode_raw(image_raw, tf.uint8)
    gt_box = tf.reshape(objects_raw, tf.stack([cnt_of_objects, 5]))
    image = tf.reshape(image, tf.stack([height, width, 3]))

    return image, gt_box