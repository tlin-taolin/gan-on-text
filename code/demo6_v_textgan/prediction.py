# -*- coding: utf-8 -*-
import os
import datetime
from os.path import join, isdir, isfile

import tensorflow as tf
from six.moves import cPickle

import parameters as para
from code.utils.logger import log
from code.dataset.dataLoaderBBT import DataLoaderBBT
from code.dataset.dataLoaderBBTV1 import DataLoaderBBTV1
from code.model.textGANV0 import TextGANV0
from code.model.textGANV1 import TextGANV1
from code.model.textGANV2 import TextGANV2
from code.model.textGANV3 import TextGANV3


def init(args, data_loader_fn):
    log('use {} to init data.'.format(data_loader_fn.__name__))
    loader = data_loader_fn(args)
    return loader


def check_training_status(args, data_loader):
    """check the training status"""
    # check compatibility if training is continued from previously saved model
    if args.INIT_FROM is not None:
        # check if all necessary files exist
        init_from = join(args.INIT_FROM, 'checkpoints')
        assert isdir(init_from), " %s must be a path" % init_from
        assert isfile(join(init_from, "config.pkl")), \
            "config.pkl file does not exist in path %s" % init_from
        ckpt = tf.train.get_checkpoint_state(init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(join(init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["EMBEDDING_SIZE", "RNN_LAYER"]

        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[checkme],\
                "parameters disagree on '%s' " % checkme
        return ckpt
    else:
        return None


def main(args, data_loader_fn, MODEL):
    start_time = datetime.datetime.now()

    data_loader = init(args, data_loader_fn)
    ckpt = check_training_status(args, data_loader)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.ALLOW_SOFT_PLACEMENT,
            log_device_placement=args.LOG_DEVICE_PLACEMENT)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # init model and save configuration.
            model = MODEL(args, data_loader, sess)

            # inference procedure.
            model.define_inference()
            model.define_loss()
            model.define_train_op()
            model.define_keep_tracking()

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if ckpt is not None:
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            model.save_model()

            # train the model a bit.
            model.running()

    end_time = datetime.datetime.now()
    log('total execution time: {}'.format((end_time - start_time).seconds))

    os.system("cp {o} {d}".format(o='record', d=model.out_dir))


if __name__ == '__main__':
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
    elif args.MODEL_TYPE == 'TextGANV3':
        model = TextGANV3
        data_loader_fn = DataLoaderBBTV1
    else:
        raise NotImplementedError

    main(args, data_loader_fn, model)
