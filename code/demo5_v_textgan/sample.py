# -*- coding: utf-8 -*-
import datetime
from os.path import join

import numpy as np
import tensorflow as tf
from six.moves import cPickle

import parameters as para
from code.utils.logger import log
from code.dataset.dataLoaderBBT import DataLoaderBBT
from code.model.textGANV0 import TextGANV0


def init(args, data_loader_fn):
    log('use {} to init data.'.format(data_loader_fn.__name__))
    loader = data_loader_fn(args)
    return loader


def check_training_status(args, data_loader):
    if args.INIT_FROM is None:
        raise 'for sampling, you must ensure the init_from is not None!'
    else:
        log('init from the path {}'.format(args.INIT_FROM))
        init_from = join(args.INIT_FROM, 'checkpoints')
        ckpt = tf.train.get_checkpoint_state(init_from)

        if ckpt.model_checkpoint_path is None:
            raise 'no checkpoints path can be found!'

        with open(join(init_from, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        return ckpt, saved_args


def inference(model):
    with tf.variable_scope('generator'):
        model.embedding_model()
        model.language_model()
        model.prepare_generator()
        embedded_x_enc = model.embedding_model(model.x_enc, reuse=True)
        embedded_x_dec = model.embedding_model(model.x_dec, reuse=True)
        embedded_x = (embedded_x_enc, embedded_x_dec)

        model.embedded_G, model.logits_G, model.probs_G, \
            model.outputs_G, model.enc_state_G, model.dec_state_G \
            = model.define_generator(z=model.z, embedded_x=embedded_x)


def main(args, data_loader_fn, MODEL):
    start_time = datetime.datetime.now()
    data_loader = init(args, data_loader_fn)

    with tf.Session() as sess:
        model = MODEL(args, data_loader, sess, infer=True)

        tf.global_variables_initializer().run()
        inference(model)
        model.build_dir_for_tracking()

        ckpt, saved_args = check_training_status(args, data_loader)

        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            data = data_loader.x_batches
            data = data[np.random.randint(len(data))]
            data = data[np.random.randint(len(data))]
            true_question, true_answer = data
            faked_answer = model.sample_from_latent_space(true_question)

            log('true question: {}'.format(
                ' '.join([str(data_loader.words[x]) for x in true_question])))
            log('true answer: {}'.format(
                ' '.join([str(data_loader.words[x]) for x in true_answer])))
            log('faked answer: {}'.format(
                ' '.join([str(data_loader.words[x]) for x in faked_answer])))

    end_time = datetime.datetime.now()
    log('total execution time: {}'.format((end_time - start_time).seconds))


if __name__ == '__main__':
    # change the execution path here!
    args = para.get_args()

    data_loader_fn = DataLoaderBBT

    model = TextGANV0
    main(args, data_loader_fn, model)
