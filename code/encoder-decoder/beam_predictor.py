import numpy as np
import tensorflow as tf
import helpers
import sys

def beam_predictor(enc_inp, encoder_decoder, top_value,
                               top_index, dec_states,
                               top_k,  max_seq_len, sess, logit=None, probs=None, signal=None):
    sequence = [2] # start token
    # append user inputs if necessary
    if signal != None:
        sequence.append(signal)

    dec_inp = helpers.build_input(sequence)
    candidates = []
    options = []

    feed_dict = {
    encoder_decoder.enc_inputs: enc_inp,
    encoder_decoder.dec_inputs: dec_inp
    }
    values, indexs, state = sess.run([top_value, top_index, dec_states], feed_dict)
    # top beam_size first words

    # print "check the sum of prob"
    # print sess.run(tf.reduce_sum(probs), feed_dict)
    #
    # print "Get the prob array"
    # prob_list = sess.run(probs, feed_dict)
    # print prob_list.shape

    for i in xrange(len(values)):
        candidates.append([values[i], [indexs[i]]])

    # print candidates
    best_sequence = None
    highest_score = -sys.maxint-1
    A_ = []

    while True:
        for i in xrange(len(candidates)):
            sequence = candidates[i][1]
            score = candidates[i][0]

            # if reach EOF
            if sequence[-1] == 3 or len(sequence) >= max_seq_len:
                if score > highest_score:
                    highest_score = score
                    best_sequence = sequence
                continue

            dec_inp = helpers.build_input(sequence)
            feed_dict = {
            encoder_decoder.enc_states: state,
            encoder_decoder.dec_inputs: dec_inp
            }

            values, indexs = sess.run([top_value, top_index], feed_dict)

            for j in xrange(len(values)):
                new_seq = list(sequence)
                new_seq.append(indexs[j])
                # print "values of each option:"
                # print values[j]
                options.append([score+values[j], new_seq])

        options.sort(reverse=True)
        candidates = []

        for i in xrange(min(len(options), top_k)):
            if options[i][0] > highest_score:
                candidates.append(options[i])
        # print candidates
        for item in candidates:
            A_.append(item[1])
        options = [] # clean up
        if len(candidates) == 0:
            break

    if signal:
        best_sequence = [signal] + best_sequence # output sequence input by users when running

    return best_sequence[:-1], A_ # remove eof
