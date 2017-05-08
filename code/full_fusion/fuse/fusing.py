import numpy as np
import tensorflow as tf
import fuse_utils as fu


def define_placeholder(batch_size, candidate_size, candidate_max_length, embedding):
    input_ph = tf.placeholder(tf.int32,
                             [batch_size, candidate_size, candidate_max_length],
                             name="input_candidate")
    embedding_ph = tf.placeholder(
                              tf.float32,
                              embedding.shape,
                              name="embedding_matrix")
    return input_ph, embedding_ph


def weight_variable(shape, initmethod=tf.truncated_normal, name="W", trainable=True):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name="b"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def max_pool(x, ksize, strides, padding="SAME", name="pool"):
    '''
    max pooling. x -> [batch, height, width, channels]
    '''
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, name=name)


def conv2d(x, W, strides, padding="SAME", name="conv"):
    '''
    x -> [batch, in_height, in_width, in_channels] treat as image but in_channels=1
    W -> [filter_height, filter_width, in_channels, out_channels]
    '''
    return tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)


def leakyrelu(conv, b, alpha=0.01, name="leaky_relu"):
    '''
    use relu as activation
    '''
    temp = tf.nn.bias_add(conv, b)
    return tf.maximum(temp * alpha, temp)


def define_fusion(input, embedding):
    vocab_size, embedding_size = embedding.get_shape().as_list()
    batch_size, candidate_size, candidate_length = input.get_shape().as_list()
    embedded_input = tf.nn.embedding_lookup(embedding, input)

    print('vocab size:{},embedding size:{}'.format(vocab_size, embedding_size))
    print('batch size:{},candidate size:{}'.format(batch_size, candidate_size))
    print('size of embedded_input: {}'.format(embedded_input.get_shape()))

    # convolution
    conv_spatials = [2, 2]
    conv_depths = [32, 32]
    archits = zip(conv_spatials, conv_depths)

    pooled_outputs = []
    for i, (conv_spatial, conv_depth) in enumerate(archits):
        with tf.variable_scope("conv-lrelu-pooling-%s"%i):
            W = weight_variable(shape=[1, conv_spatial, embedding_size, conv_depth])
            print "This is parameter W:"
            print W
            b = bias_variable(shape=[conv_depth])
            conv = conv2d(
                        embedded_input,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID")
            h = leakyrelu(conv, b, alpha=1.0/5.5)
            pooled = max_pool(
                        h,
                        ksize=[1, 1, candidate_length-conv_spatial+1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID")
            pooled_outputs.append(pooled)
    num_filters_total = sum([x[1] for x in archits])
    features = tf.reshape(
                         tf.concat(pooled_outputs, 3),
                         [batch_size, candidate_size, num_filters_total])
    print('size of fused feature: {}'.format(features.get_shape()))

    features = tf.transpose(features, [0, 2, 1])
    print('size of feature for step 1: {}'.format(features.get_shape()))
    features = tf.reshape(features, shape=[batch_size * num_filters_total, -1])
    print('size of feature for step 2: {}'.format(features.get_shape()))

    W = tf.get_variable(
        "W",
        shape=[candidate_size, 1],
        initializer=tf.contrib.layers.xavier_initializer())

    weighted_features = tf.matmul(features, W)
    final_features = tf.reshape(weighted_features, [batch_size, num_filters_total])

    print('size of the final feature: {}'.format(final_features.get_shape()))
    return final_features


def get_candidates(GA_batches):
    '''
    Currently for testing the gradient backpropagation.
    '''
    return np.squeeze(np.array(GA_batches), axis=(0,))


def main(force, sess):
    batch_size = 1
    candidate_size = 21
    num_batches = 1

    embedding, _, cand_beam, _ = fu.load_data(force)
    print "Finish loading data."

    '''
    Q_batches: Questions within this batch  --- num_batch x batch_size
    A_batches: Truth answering
    GA_batches: candidates
    print (np.array(GA_batches)).shape -- (1, 64, 21, x) where x is number of words after padding
                                       -- (num_batch, batch_size, candidates, max_sentences_length)
    '''

    candidates, candidate_max_length = fu.build_up_candidates(cand_beam)
    Q_batches, A_batches, GA_batches = fu.build_batches(candidates, batch_size, num_batches)
    candidates_tofeed = get_candidates(GA_batches)

    input_ph, embedding_ph = define_placeholder(batch_size, candidate_size, candidate_max_length, embedding)
    fused_feature = define_fusion(input_ph, embedding_ph)

    return embedding, fused_feature, candidates_tofeed

if __name__ == "__main__":
    _ = main(True)
