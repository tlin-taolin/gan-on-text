# -*- coding: utf-8 -*-
import datetime
from os.path import join

import tensorflow as tf
from six.moves import cPickle

import parameters as para
from code.utils.logger import log
from code.dataset.dataLoaderBBC import DataLoaderBBC
from code.model.textGAN import TextGAN
from code.model.textGANV3 import TextGANV3


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

        model.embedded_G, model.logits_G, model.probs_G, \
            model.outputs_G, model.state_G = model.define_generator(z=model.z)

        model.top_value, model.top_index = tf.nn.top_k(
            model.logits_G, k=model.para.BEAM_SEARCH_SIZE, sorted=True)


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

            print(model.sample_from_latent_space(num=200))
            log('above content is generated from lstm.')
    end_time = datetime.datetime.now()
    log('total execution time: {}'.format((end_time - start_time).seconds))


if __name__ == '__main__':
    # change the execution path here!
    args = para.get_args()

    data_loader_fn = DataLoaderBBC

    model = TextGAN
    model = TextGANV3
    main(args, data_loader_fn, model)
