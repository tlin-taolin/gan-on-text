import numpy as np
import os
import json
import sys

def create_buckets(list_bucket_size):
    '''
    return tuples containing the size of buckets. Note the +1 from the GO signal
    for decoding.
    '''
    buckets = []
    for i in xrange(len(list_bucket_size)):
        for j in xrange(len(list_bucket_size)):
            buckets.append((list_bucket_size[i], list_bucket_size[j]+1))
    return buckets

def data_processing(data, size, batch_size):
    enc_len = size[0]
    dec_len = size[1]

    enc_inp = np.zeros((enc_len, batch_size))
    dec_inp = np.zeros((dec_len, batch_size))
    dec_tar = np.zeros((dec_len, batch_size))

    for i in xrange(len(data)):
        pair = data[i]
        enc_inp[enc_len-len(pair[0]):enc_len, i] = pair[0][::-1]
        dec_inp[1:len(pair[1])+1, i] = pair[1]
        dec_tar[0:len(pair[1]), i] = pair[1]
        # start and end token
        dec_inp[0, i] = 2
        dec_tar[len(pair[1]), i] = 3

    return enc_inp, dec_inp, dec_tar

def build_input(sequence):
    dec_inp = np.zeros((1, len(sequence)))
    dec_inp[0][:] = sequence
    return dec_inp.T

def print_sentence(index_list, reader):
    for index in index_list:
        if index < 20525:
            sys.stdout.write(reader.id_dict[index])
            sys.stdout.write(" ")
    sys.stdout.write("\n")

def predict(enc_inp, sess, encoder_decoder, top_indexs, dec_states):
    dec_inp = np.zeros((1,1))
    dec_inp[0][0] = 2 # 2 is the start token
    index_output = []

    feed_dict = {
    encoder_decoder.enc_inputs: enc_inp,
    encoder_decoder.dec_inputs: dec_inp
    }

    indexs, state = sess.run([top_indexs, dec_states], feed_dict)
    index_output.append(indexs[0])

    while True:
        dec_inp[0][0] = indexs[0]

        feed_dict = {
        encoder_decoder.enc_states: state,
        encoder_decoder.dec_inputs: dec_inp
        }

        indexs, state = sess.run([top_indexs, dec_states], feed_dict)
        if indexs[0] == 3: # if meet EOF
            break
        index_output.append(indexs[0])

    return index_output

def translate(token_list, reader):
    enc = []
    for token in token_list:
        if token in reader.dict:
            enc.append(reader.dict[token])
        else:
            enc.append(reader.dict["[unk]"])
    return enc
