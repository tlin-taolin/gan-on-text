# -*- coding: utf-8 -*-
import datetime
from os.path import join

import numpy as np
import tensorflow as tf
from six.moves import cPickle

import parameters as para
from code.utils.logger import log
from code.dataset.dataLoaderBBT import DataLoaderBBT
from code.dataset.dataLoaderBBTV1 import DataLoaderBBTV1
from code.model.textGANV0 import TextGANV0
from code.model.textGANV1 import TextGANV1
from code.model.textGANV2 import TextGANV2


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


def main(args, data_loader_fn, MODEL):
    start_time = datetime.datetime.now()
    data_loader = init(args, data_loader_fn)

    with tf.Session() as sess:
        model = MODEL(args, data_loader, sess, infer=True)

        tf.global_variables_initializer().run()
        model.define_inference()
        model.build_dir_for_tracking()

        ckpt, saved_args = check_training_status(args, data_loader)

        if ckpt and ckpt.model_checkpoint_path:
            # restore saved model.
            model.saver.restore(sess, ckpt.model_checkpoint_path)

            model.print_out_comparison(ckpt)

    end_time = datetime.datetime.now()
    log('total execution time: {} s'.format((end_time - start_time).seconds))


if __name__ == '__main__':
    # change the execution path here!
    args = para.get_args()

    if args.MODEL_TYPE == 'TextGANV0':
        model = TextGANV0
        data_loader_fn = DataLoaderBBT
    elif args.MODEL_TYPE == 'TextGANV1':
        model = TextGANV1
        data_loader_fn = DataLoaderBBTV1
    elif args.MODEL_TYPE == 'TextGANV2':
        model = TextGANV2
        data_loader_fn = DataLoaderBBT
    else:
        raise NotImplementedError

    main(args, data_loader_fn, model)
