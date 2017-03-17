# -*- coding: utf-8 -*-
import os
import shutil
import datetime

import tensorflow as tf

import parameters as para
from code.utils.logger import log
from code.dataset.dataLoaderChildrenStory import DataLoaderChildrenStory
from code.dataset.dataLoaderBBC import DataLoaderBBC

from code.model.textGAN import TextGAN


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
            model.define_pretrain_loss()
            model.define_pretraining_op()
            # train stuff
            model.define_train_loss()
            model.define_training_op()
            model.define_keep_tracking()

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # pretrain the model a bit.
            pretrain_model_index = 0

            if para.IF_PRETRAIN or not os.path.exists(model.pretrain_dir):
                log('------ do the pretrain ------ \n')
                log('....empty pretrain folder.')
                shutil.rmtree(model.pretrain_dir)
                os.makedirs(model.pretrain_dir)

                for cur_epoch in range(para.EPOCH_PRETRAIN):
                    log('pretrain epoch {}'.format(cur_epoch))
                    avg_l_d, avg_l_g, duration = model.run_pretrain_epoch()
                    log('pretrain loss d: {}, pretrain loss g: {}, execution speed: {:.2f} seconds/batch\n'.format(
                        avg_l_d, avg_l_g, duration))

                    if cur_epoch % para.CHECKPOINT_EVERY == 0:
                        model.saver.save(
                            sess, model.pretrain_model,
                            global_step=pretrain_model_index)
                        pretrain_model_index += 1
                        log("save {}-th bestmodel to path: {}.\n".format(
                            pretrain_model_index, model.pretrain_model))
            else:
                checkpoint = tf.train.latest_checkpoint(model.pretrain_dir)
                saver = tf.train.import_meta_graph(
                    "{}.meta".format(checkpoint))
                saver.restore(sess, checkpoint)

            log('------ do the standard GAN training ------ \n')
            best_model_index = 0

            for cur_epoch in range(para.EPOCH_TRAIN):
                log('train epoch {}'.format(cur_epoch))
                avg_l_d, avg_l_g, duration = model.run_train_epoch()
                log('train loss d: {}, train loss g: {}, execution speed: {:.2f} seconds/batch\n'.format(
                    avg_l_d, avg_l_g, duration))

                if cur_epoch % para.CHECKPOINT_EVERY == 0:
                    best_model_index += 1
                    model.saver.save(
                        sess, model.best_model,
                        global_step=best_model_index)
                    log("save {}-th bestmodel to path: {}.\n".format(
                        best_model_index, model.best_model))

    end_time = datetime.datetime.now()
    log('total execution time: {}'.format((end_time - start_time).seconds))

if __name__ == '__main__':
    data_loader = DataLoaderBBC

    model = TextGAN
    main(data_loader, model)
