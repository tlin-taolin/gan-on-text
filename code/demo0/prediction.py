# -*- coding: utf-8 -*-
import os
import datetime

import numpy as np
import tensorflow as tf

import code.utils.initData as init_data
from code.utils.myprint import myprint
from code.model.gan import GAN

import parameters as para


def init():
    """some basic init."""
    np.random.seed(para.SEED)

    # init real data pdf
    mean_matrix = np.zeros(2)
    covariance_matrix = 0.01 * np.eye(2) + [[0., 0.0], [0.05, 0.]]
    real_x_samples = init_data.init_x_real(
        para, mean_matrix, covariance_matrix)

    # init sample
    z_samples = init_data.init_z(para)
    return real_x_samples, z_samples


def run(MODEL):
    """run the model."""
    start_time = datetime.datetime.now()

    # init data
    real_x, z = init()

    # init placeholder for statistics
    all_losses = []

    # setup the model and then train.
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=para.ALLOW_SOFT_PLACEMENT,
            log_device_placement=para.LOG_DEVICE_PLACEMENT)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # init model
            model = MODEL(para)

            # setup the model as well as the training procedure.
            model.define_inference()
            model.define_loss()
            model.define_training()

            # define the training procedure.
            model.keep_tracking(sess)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # define best parameter.
            best_loss_D, best_loss_G, best_epoch, best_model_index \
                = - float('inf'), float('inf'), 0, 0

            # train the model in loop.
            for epoch in range(para.MAX_EPOCHS):
                myprint("Epoch {}" . format(epoch))

                losses, loss_D, loss_G = model.run_epoch(
                    sess, real_x, z, train=True)
                all_losses.append(losses)

                if loss_G < best_loss_G and loss_D > best_loss_D:
                    best_epoch = epoch
                    best_model_index += 1
                    best_model = model.best_model + "-" + str(best_model_index)
                    model.saver.save(sess, best_model)
                    myprint("save best model to: {}.\n".format(best_model))

                if epoch - best_epoch > para.EARLY_STOPPING and \
                        epoch > para.MIN_EPOCHS and best_epoch != 0:
                    break

    exection_time = (datetime.datetime.now() - start_time).total_seconds()
    myprint("execution time: {t:.3f} seconds" . format(t=exection_time))


if __name__ == '__main__':
    model = GAN
    run(model)
