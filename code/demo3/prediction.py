# -*- coding: utf-8 -*-
import datetime
from os.path import join

import tensorflow as tf

import parameters as para
from code.utils.logger import log
import code.utils.opfiles as opfile
from code.dataset.dataLoaderChildrenStory import DataLoaderChildrenStory

from code.model.textGAN import TextGAN
from code.model.textGANV1 import TextGANV1
from code.model.textGANV2 import TextGANV2


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
            model = MODEL(para, data_loader)
            model.define_inference()
            # pretrain stuff
            model.define_pretrain_loss()
            model.define_pretraining_op()
            # train stuff
            model.define_train_loss()
            model.define_training_op()
            model.define_keep_tracking(sess)

            # define the model saving.
            best_model_index = 0

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # pretrain the model a bit.
            log('------ do the pretrain ------ \n')
            for cur_epoch in range(para.EPOCH_PRETRAIN):
                avg_l_d, avg_l_g, duration = model.run_pretrain_epoch(sess)
                log('pretrain loss d:{}, pretrain loss g:{}, \
                    execution speed:{} second/batch'.format(
                    avg_l_d, avg_l_g, duration))

            best_model = model.best_model + '-' + str(best_model_index)
            model.saver.save(sess, best_model)
            best_model_index += 1
            log("save best model to: {}.\n".format(best_model))

            log('------ do the standard GAN training ------ \n')
            for cur_epoch in range(para.EPOCH_TRAIN):
                avg_l_d, avg_l_g, duration = model.run_train_epoch(sess)
                log('train loss d:{}, train loss g:{}, \
                    execution speed:{} second/batch'.format(
                    avg_l_d, avg_l_g, duration))

                if cur_epoch % para.CHECKPOINT_EVERY:
                    best_model_index += 1
                    best_model = model.best_model + '-' + str(best_model_index)
                    model.saver.save(sess, best_model)
                    log("save best model to: {}.\n".format(best_model))

            log('------ generate sentence from latent space / noice ------ \n')
            generated_sentences = []
            for cur_epoch in range(para.EPOCH_SENTENCE_GENERATION):
                generated_sentence = model.sample_from_latent_space(sess)
                generated_sentences.append(generated_sentence)

            generated_sentences_string = '\n\n'.join(
                [' '.join(s) for s in generated_sentences])
            opfile.write_txt(
                generated_sentences_string,
                join(model.out_dir, 'sentence_generation'))

    end_time = datetime.datetime.now()
    log('total execution time: {}'.format((end_time - start_time).seconds))

if __name__ == '__main__':
    data_loader = DataLoaderChildrenStory

    model = [TextGAN, TextGANV1, TextGANV2][0]

    main(data_loader, model)
