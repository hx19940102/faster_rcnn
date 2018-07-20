import tensorflow as tf
import read_data
import rpn_model
import tensorflow.contrib.slim as slim
import numpy as np

######################################
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
training_data_dir = "voc2012.tfrecords"
batch_size = 1
learning_rate = 0.00001
vgg_paramater_dir = 'vgg_16.ckpt'
rpn_paramater_dir = 'rpn.ckpt'
training_rounds = 20000
######################################


image, gt_box = read_data.read_data_from_tfrecords([training_data_dir], True, batch_size)
gt_box_without_class = tf.slice(gt_box, [0, 0], [-1, 4])
gt_box_without_class = tf.cast(gt_box_without_class, tf.float32)
image = tf.expand_dims(image, axis=0)
image = tf.cast(image, tf.float32)
image = image - [_R_MEAN, _G_MEAN, _B_MEAN]
rpn = rpn_model.regional_proposal_network(is_training=True)
labels, anchor_targets, rpn_cls_score, rpn_bbox_pred, vgg_variables = rpn.build_model(image, gt_box_without_class)
cls_loss, reg_loss = rpn.loss_func(labels, anchor_targets, rpn_cls_score, rpn_bbox_pred)
combined_loss = tf.add(cls_loss, reg_loss)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=combined_loss)

# Initialization
pretrained_init_op = slim.assign_from_checkpoint_fn(vgg_paramater_dir, vgg_variables)
global_vars_init_op = tf.global_variables_initializer()
local_vars_init_op = tf.local_variables_initializer()
combined_op = tf.group(local_vars_init_op, global_vars_init_op)
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(combined_op)
    # pretrained_init_op(sess)
    # If restart training after break
    saver.restore(sess, rpn_paramater_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(training_rounds):
        _, error  = sess.run([train_op, combined_loss])
        print("Round %d, Loss = %f" % (i, error))
        if i % 299 == 0:
            saver.save(sess, rpn_paramater_dir)
    coord.request_stop()
    coord.join(threads)

saver.save(sess, rpn_paramater_dir)