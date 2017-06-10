import numpy as np
import tensorflow as tf
import os
import re
import sys
import json

import beam_predictor
import helpers
import data_reader
import model

file_name = "raw_data"
expression = r"[0-9]+|[']*[\w]+" # re for input from users
batch_size = 1

signal = False

bucket_option = [5, 10, 15, 20, 25, 31] # there are len(bucket_option)^2 types
buckets = helpers.create_buckets(bucket_option)
# print buckets
reader = data_reader.reader(file_name=file_name, batch_size=batch_size,
buckets=buckets, bucket_option=bucket_option, clean_mode=True)

vocab_size = len(reader.dict)
# print vocab_size 20525 vocabulary

hidden_size = 64
projection_size = 32
embedding_size = 80
num_layers = 1

truncated_std = .1
keep_prob = .95
max_epoch = 50
norm_clip = 5.
adam_lr = .005

#output_size for sftmx layer. we could use project to reduce the size
output_size = hidden_size
if projection_size != None:
    output_size = projection_size

model_name = "p"+str(projection_size)+"_h"+str(hidden_size)+"_x"+str(num_layers)
save_path = file_name+"/"+model_name

# parameters for prediction
beam_size = 5
top_k = 10 # candidates
max_seq_len = 20

# restore trained model
encoder_decoder = model.seq2seq(batch_size, vocab_size, embedding_size,
                                  num_layers, keep_prob, output_size,
                                    truncated_std, hidden_size, projection_size)

encoder_decoder.initialize_input_layers()

# model returns
# self.total_loss, self.avg_loss, logits, self.enc_states, self.dec_outputs, self.dec_states
_, _, logits, _, _, dec_states, probs = encoder_decoder._seq2seq()

# flat logits and make prediction
logit = logits[-1]

top_value, top_index = tf.nn.top_k(logit, k=beam_size, sorted=True)

# load trained model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
cwd = os.getcwd()
saver.restore(sess, cwd+"/"+save_path+"/model.ckpt")
print "model restored"

interactive = False
if interactive:
    print("\n--------------------------")
    print("--Interactive mode is on--")
    print("--------------------------\n")
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print "\n"
            print "Session closed."
            break

        token_list = re.findall(expression, line.lower())

        sequence = helpers.translate(token_list, reader)
        enc_inp = helpers.build_input(sequence[::-1])

        response, A_ = beam_predictor.beam_predictor(enc_inp, encoder_decoder=encoder_decoder,
        top_value=top_value, top_index=top_index, dec_states=dec_states , top_k=top_k,
        max_seq_len=max_seq_len, sess=sess, probs=probs, signal=None)

        sys.stdout.write("src: ")
        helpers.print_sentence(sequence, reader)

        sys.stdout.write("res (beam search): ")
        helpers.print_sentence(response, reader)

        sys.stdout.write("res (beam candidates): ")
        for item in A_:
            helpers.print_sentence(item, reader)

else:
    total_emb = encoder_decoder.get_emb(sess)
    np.save("emb/total_emb.npy", total_emb)
    print "Embedding of vocabulary saved."
    # print total_emb.shape #(20525, 256)
    candidate_top = 10
    argmax_list = []
    weighted_pick_list = []
    beam_search_list = []
    counter = 0
    while True:
        if counter %10 == 0:
            print "Finished %s rows." %counter
        counter += 1
        try:
            data, idx = reader.next_batch()
            enc_inp, dec_inp, dec_tar = helpers.data_processing(data,
                                                        buckets[idx], batch_size)
            response, A_ = beam_predictor.beam_predictor(enc_inp, encoder_decoder=encoder_decoder,
            top_value=top_value, top_index=top_index, dec_states=dec_states , top_k=top_k,
            max_seq_len=max_seq_len, sess=sess, probs=probs, signal=None)

            beam_search_list.append([data[0][0], data[0][1], response])
            for idx, item in enumerate(A_):
                if idx < candidate_top:
                    beam_search_list.append([data[0][0], data[0][1], item])

        except Exception:
            print "Finish storing, writing.."

            sess.close()
            print "\n session closed"
            break
