# -*- coding:UTF-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from task1 import *

tf.logging.set_verbosity(tf.logging.INFO)


def CNN_FC_model(batch,mode):
    # input layer, features has 6000 triplets,batchsize多大训练时调用函数的时候再指定
    #input_layer = tf.reshape(batch, [-1, 64, 64, 3])
    input_layer = batch

    # Conv Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[8, 8],
        padding="valid",
        activation=tf.nn.relu
    )

    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

    # Conv Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=7,
        kernel_size=[5,5],
        padding="valid",
        activation=tf.nn.relu
    )

    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=1)

    # FC Layer 1
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 7])  # reshape into a vector and then feed it into FC layer
    dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)

    # 训练的时候这个mode打开
    dropout = tf.nn.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # FC Layer 2
    descriptor = tf.layers.dense(inputs=dropout,units=16)


    return descriptor


# m is margin
def compute_loss(anchor_descriptor,puller_descriptor,pusher_descriptor,margin):
    diff_pos = anchor_descriptor - puller_descriptor
    diff_neg = anchor_descriptor - pusher_descriptor
    Ltriplet = np.maximum(0, 1 - tf.square(diff_neg)/(tf.square(diff_pos) + margin))
    Lpair = tf.maximum(diff_pos)

    Loss = Ltriplet + Lpair

    return Loss


def main(unused_argv):
    batchsize = 100
    margin = 0.01

    # load batch and split into anchor, puller, pusher
    train_anchor = tf.placeholder(tf.float32, shape=[batchsize,64,64,3], name="anchor")
    train_puller = tf.placeholder(tf.float32, shape=[batchsize,64,64,3], name="puller")
    train_pusher = tf.placeholder(tf.float32, shape=[batchsize,64,64,3], name="pusher")


    # feed data to the CNN and computer descriptor
    anchor_descriptor = CNN_FC_model(train_anchor,tf.estimator.ModeKeys.TRAIN)
    puller_descriptor = CNN_FC_model(train_puller,tf.estimator.ModeKeys.TRAIN)
    pusher_descriptor = CNN_FC_model(train_pusher,tf.estimator.ModeKeys.TRAIN)

    # compute loss
    loss = compute_loss(anchor_descriptor,puller_descriptor,pusher_descriptor,margin)

    # define Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss,global_step=tf.train.get_global_step())


    # Create the Estimator, use it to train the network
    # model_dir是放模型参数的地方
    my_model = tf.estimator.Estimator(
        model_fn=CNN_FC_model, model_dir="/Users/gaoyingqiang/Documents/GitHub/Master-TUM/TDCV/exercise_3/ex3")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        batch,
        mode=tf.estimator.ModeKeys.TRAIN,
        batch_size=10,
        num_epochs=None,
        shuffle=True)

    my_model.train(
        input_fn=train_input_fn,
        steps=10)


if __name__ == "__main__":
  tf.app.run()







