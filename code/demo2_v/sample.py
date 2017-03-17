# -*- coding: utf-8 -*-
import os
from os import listdir

import tensorflow as tf
from optparse import OptionParser

import parameters as para
from code.utils.logger import log
import code.utils.opfiles as opfile
from code.dataset.dataLoaderShakespeare import DataLoaderShakespeare
from code.dataset.dataLoaderBBC import DataLoaderBBC
from code.model.textG import TextG


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
        model = MODEL(para, data_loader, sess, training=False)
        model.inference()
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
                vocab_word2index, vocab_index2word)
            generated_sentences.append(generated_sentence)

        log('generate sentence and write to path: {}'.format(checkpoint_dir))
        generated_sentences_string = '.\\\\ \n'.join(
            [' '.join(s) for s in generated_sentences]) + '.\n\n'

        print(generated_sentences_string)
        opfile.write_txt(
            generated_sentences_string,
            os.path.join('..', checkpoint_dir, 'generated_sentences'),
            type='w')

if __name__ == '__main__':
    # change the execution path here!
    execution = os.path.join(
        para.TRAINING_DIRECTORY, 'runs', 'DataLoaderShakespeare',
        'code.model.textG.TextG', '1489687743', 'checkpoints')

    data_loader = DataLoaderShakespeare
    data_loader = DataLoaderBBC
    model = TextG
    main(data_loader, model, execution)
