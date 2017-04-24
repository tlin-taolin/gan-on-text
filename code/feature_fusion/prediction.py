# -*- coding: utf-8 -*-
import re
import pickle
from os.path import join, exists

import numpy as np
import tensorflow as tf


def make_square(seq, size):
    return zip(*[iter(seq)] * size)


def read_txt(path, size):
    with open(path, 'r') as f:
        data = f.read()
    data = re.findall(r'\[(.*?)\]', data, re.S)
    data = make_square(data, size)

    data = [
        map(lambda x:
            map(int,
                filter(lambda y: y is not '',
                       x.replace(']', '').replace('[', '').strip().split(', '))
                ),
            instance)
        for instance in data]

    print('----------------------')
    print('number of instances: {}'.format(len(data)))
    return data


def load_pickle(path):
    """load data by pickle."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def write_pickle(data, path):
    """dump file to dir."""
    print("write --> data to path: {}\n".format(path))
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def load_data(force):
    """load dataset."""
    # define path.
    path_root = 'data/can_emb'
    path_emb = join(path_root, 'emb', 'total_emb.npy')
    path_candidate = join(path_root, 'candidates')
    path_candidates_argmax = join(path_candidate, 'argmax_list.txt')
    path_candidates_argmax_p = join(path_candidate, 'argmax_list.pickle')
    path_candidates_pick = join(path_candidate, 'weighted_pick_list.txt')
    path_candidates_pick_p = join(path_candidate, 'weighted_pick_list.pickle')
    path_candidates_beam = join(path_candidate, 'beam_search_list.txt')
    path_candidates_beam_p = join(path_candidate, 'beam_search_list.pickle')

    # load embedding.
    embedding = np.load(path_emb)

    # load candidates
    candidate_argmax, candidates_beam, candidates_pick = None, None, None
    # if force or not exists(path_candidates_argmax_p):
    #     candidate_argmax = read_txt(path_candidates_argmax, size=3)
    #     write_pickle(candidate_argmax, path_candidates_argmax_p)
    # else:
    #     candidate_argmax = load_pickle(path_candidates_argmax_p)

    if not force or not exists(path_candidates_beam_p):
        candidates_beam = read_txt(path_candidates_beam, size=3)
        write_pickle(candidates_beam, path_candidates_beam_p)
    else:
        print "Loading existing pickle..."
        candidates_beam = load_pickle(path_candidates_beam_p)

    # if force or not exists(path_candidates_pick_p):
    #     candidates_pick = read_txt(path_candidates_pick, size=3)
    #     write_pickle(candidates_pick, path_candidates_pick_p)
    # else:
    #     candidates_pick = load_pickle(path_candidates_pick_p)
    ''' Embedding size: 256
        vocab_size: 20525
        Totoal beam candidates: 742239
        None for candidate_argmax, candidates_pick
    '''
    '''
       Form of beam candidates:
            _________
                     |
                     |
                     |
           [Q, A, A']| 21 pairs. A is the truth answering.
                     |
                     |
            _________|

    '''
    return embedding, candidate_argmax, candidates_beam, candidates_pick


def build_up_candidates(cand_beam):
    # Note that I use `0` to denote padding symbols. please correct it.
    candidates = []
    candidate_max_length = max([len(candidate[2]) for candidate in cand_beam])
    tuples = make_square(cand_beam, 21)

    for tuple in tuples:
        head = tuple[0][0: 2]
        candidate = map(lambda line: line[2], tuple)
        # candidate_length_max = max(map(lambda c: len(c), candidate))
        candidate = [
            c + [0] * (candidate_max_length - len(c)) for c in candidate]
        candidates.append(head + candidate)
    return candidates, candidate_max_length


def build_batches(candidates, batch_size, num_batches):
    Q_list = [candidate[0] for candidate in candidates]
    A_list = [candidate[1] for candidate in candidates]
    Generated_A_list = [candidate[2:] for candidate in candidates]

    Q_batches = [
        Q_list[batch_th * batch_size: (batch_th + 1) * batch_size]
        for batch_th in range(num_batches)]
    A_batches = [
        A_list[batch_th * batch_size: (batch_th + 1) * batch_size]
        for batch_th in range(num_batches)]
    GA_batches = [
        Generated_A_list[batch_th * batch_size: (batch_th + 1) * batch_size]
        for batch_th in range(num_batches)]
    return Q_batches, A_batches, GA_batches


def define_placeholder(
        batch_size, candidate_size, candidate_max_length, embedding):
    input_ph = tf.placeholder(
        tf.int32, [batch_size, candidate_size, candidate_max_length],
        name="input_candidate")
    embedding_ph = tf.placeholder(
        tf.float32, embedding.shape, name="embedding_matrix")
    return input_ph, embedding_ph


def weight_variable(
        shape, initmethod=tf.truncated_normal, name="W", trainable=True):
    """init weight."""
    initial = initmethod(shape, stddev=0.1)
    return tf.Variable(initial, name=name, trainable=trainable)


def bias_variable(shape, name="b"):
    """init bias variable."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def max_pool(x, ksize, strides, padding='SAME', name="pool"):
    """do max pooling.
        x: [batch, height, width, channels]
    """
    return tf.nn.max_pool(
        x, ksize=ksize, strides=strides, padding=padding, name=name)


def conv2d(x, W, strides, padding='SAME', name="conv"):
    """do convolution.
        x: [batch, in_height, in_width, in_channels]
        W: [filter_height, filter_width, in_channels, out_channels]
    """
    return tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)


def leakyrelu(conv, b, alpha=0.01, name="leaky_relu"):
    """use lrelu as the activation function."""
    tmp = tf.nn.bias_add(conv, b)
    return tf.maximum(tmp * alpha, tmp)


def define_fusion(input, embedding):
    # embedding related stuff.
    embedded_input = tf.nn.embedding_lookup(embedding, input)
    vocab_size, embedding_size = embedding.get_shape().as_list()
    batch_size, candidate_size, candidate_length = input.get_shape().as_list()

    print('vocab size:{},embedding size:{}'.format(vocab_size, embedding_size))
    print('batch size:{},candidate size:{}'.format(batch_size, candidate_size))
    print('size of embedded_input: {}'.format(embedded_input.get_shape()))

    # define CNN architecture:
    conv_spatials = [2, 2]
    conv_depths = [32, 32]
    archits = zip(conv_spatials, conv_depths)

    pooled_outputs = []
    for i, (conv_spatial, conv_depth) in enumerate(archits):
        with tf.variable_scope("conv-lrelu-pooling-%s" % i):

            W = weight_variable(
                shape=[1, conv_spatial, embedding_size, conv_depth]
                )

            b = bias_variable(shape=[conv_depth])

            conv = conv2d(
                embedded_input,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID'
                )

            h = leakyrelu(conv, b, alpha=1.0/5.5)

            pooled = max_pool(
                h,
                ksize=[1, 1, candidate_length - conv_spatial + 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID")
            pooled_outputs.append(pooled)

    num_filters_total = sum([x[1] for x in archits])
    features = tf.reshape(
        tf.concat(pooled_outputs, 3),
        [batch_size, candidate_size, num_filters_total])

    print('size of fused feature: {}'.format(features.get_shape()))

    # for final fusion.
    features = tf.transpose(features, [0, 2, 1])
    print('size of feature for step 1: {}'.format(features.get_shape()))
    features = tf.reshape(features, shape=[batch_size * num_filters_total, -1])
    print('size of feature for step 2: {}'.format(features.get_shape()))

    W = tf.get_variable(
        "W",
        shape=[candidate_size, 1],
        initializer=tf.contrib.layers.xavier_initializer())

    weighted_features = tf.matmul(features, W)
    final_features = tf.reshape(
        weighted_features, [batch_size, num_filters_total])

    print('size of the final feature: {}'.format(final_features.get_shape()))
    return final_features


def main(force):
    """the main entry."""
    # define some parameters.
    batch_size = 64
    candidate_size = 21
    num_batches = 1

    # load and prepare data.
    embedding, _, cand_beam, _ = load_data(force)
    print "Finish loading data."
    candidates, candidate_max_length = build_up_candidates(cand_beam)
    Q_batches, A_batches, GA_batches = build_batches(candidates, batch_size, num_batches)
    '''
    Q_batches: Questions within this batch
    A_batches: Truth answering
    GA_batches: candidates
    print (np.array(GA_batches)).shape -- (1, 64, 21, x) where x is number of words after padding
    '''

    # define placeholder.
    input_ph, embedding_ph = define_placeholder(batch_size, candidate_size, candidate_max_length, embedding)

    # # start inference.
    fused_feature = define_fusion(input_ph, embedding_ph)

if __name__ == '__main__':
    force = True
    main(force)
    # load_data(force)
