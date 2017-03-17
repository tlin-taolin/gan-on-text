# -*- coding: utf-8 -*-
import os
import datetime

import tensorflow as tf

import parameters as para
from code.utils.logger import log
from code.dataset.dataLoaderBBC import DataLoaderBBC

from code.model.textG import TextG


def init(data_loader):
    log('use {} to init data.'.format(data_loader.__name__))
    loader = data_loader(para)
    return loader


def main(data_loader_fn, MODEL):
    start_time = datetime.datetime.now()

    data_loader = init(data_loader_fn)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=para.ALLOW_SOFT_PLACEMENT,
            log_device_placement=para.LOG_DEVICE_PLACEMENT)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # init model
            model = MODEL(para, data_loader, sess)
            model.inference()
            # pretrain stuff
            model.define_loss()
            model.define_train_op()
            model.define_keep_tracking()

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # pretrain the model a bit.
            log('------ training ------ \n')
            model_index = 0

            for cur_epoch in range(para.EPOCH_TRAIN):
                log('train epoch {}'.format(cur_epoch))
                avg_l_g, duration = model.run_epoch(model.train_step)
                log('train loss: {}, execution speed: {:.2f} seconds/batch\n'.format(
                    avg_l_g, duration))

                avg_l_g, duration = model.run_epoch(model.val_step)
                log('val loss: {}, execution speed: {:.2f} seconds/batch\n'.format(
                    avg_l_g, duration))

                if cur_epoch % para.CHECKPOINT_EVERY == 0:
                    model.saver.save(
                        sess, model.best_model, global_step=model_index)
                    model_index += 1
                    log("save {}-th bestmodel to path: {}.\n".format(
                        model_index, model.best_model))

    end_time = datetime.datetime.now()
    log('total execution time: {}'.format((end_time - start_time).seconds))

if __name__ == '__main__':
    data_loader = DataLoaderBBC

    model = TextG
    main(data_loader, model)
