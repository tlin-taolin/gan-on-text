# -*- coding: utf-8 -*-
import os
from os import listdir

import tensorflow as tf
from optparse import OptionParser

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


def main(data_loader_fn, MODEL, checkpoint_dir):
    data_loader = init(data_loader_fn)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=para.ALLOW_SOFT_PLACEMENT,
            log_device_placement=para.LOG_DEVICE_PLACEMENT)
        sess = tf.Session(config=session_conf)

        # setup the model.
        model = MODEL(para, data_loader, training=False)
        model.define_inference()
        sess.run(tf.global_variables_initializer())

        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph(
            "{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        log('restore checkpoint_file from path: {}'.format(checkpoint_file))

        # define vocab that from word to index and that from index to word.
        vocab_word2index = data_loader.vocab
        vocab_index2word = dict((v, k) for k, v in vocab_word2index.items())

        generated_sentences = []
        for cur_epoch in range(para.EPOCH_SENTENCE_GENERATION):
            generated_sentence = model.sample_from_latent_space(
                sess, vocab_word2index, vocab_index2word,
                sampling_type=2, pick=1)
            generated_sentences.append(generated_sentence)

        log('generate sentence and write to path: {}'.format(checkpoint_dir))
        generated_sentences_string = '\n'.join(
            [' '.join(s) for s in generated_sentences]) + '\n\n'
        opfile.write_txt(
            generated_sentences_string,
            os.path.join('..', checkpoint_dir, 'generated_sentences'),
            type='a')

if __name__ == '__main__':
    # change the execution path here!
    execution = os.path.join(
        para.TRAINING_DIRECTORY,
        'runs', 'code.model.textGAN.TextGAN',
        '1489361968', 'checkpoints')

    data_loader = DataLoaderChildrenStory
    model = [TextGAN, TextGANV1, TextGANV2][0]
    main(data_loader, model, execution)
