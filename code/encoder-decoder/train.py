import numpy as np
import tensorflow as tf
import os
import shutil

import helpers
import data_reader
import model
import argparse

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-embsize", type=int, default=32)
    return parser.parse_args()

file_name = "raw_data"
save_root = "saved_model"

batch_size = 128
bucket_option = [5, 10, 15, 20, 25, 31] # there are len(bucket_option)^2 types
buckets = helpers.create_buckets(bucket_option)
reader = data_reader.reader(file_name, batch_size, buckets, bucket_option)

args = Parser()
vocab_size = len(reader.dict)
hidden_size = 64
projection_size = 32
embedding_size = args.embsize
num_layers = 1

#output_size for sftmx layer. we could use project to reduce the size
output_size = hidden_size
if projection_size != None:
    output_size = projection_size

model_name = "p"+str(projection_size)+"_h"+str(hidden_size)+"_x"+str(num_layers)
save_path = save_root+"/"+model_name

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

# params for training
truncated_std = .1
keep_prob = .95
max_epoch = 1
norm_clip = 5.
adam_lr = .005

# ---- build model -----
encoder_decoder = model.seq2seq(embedding_size=embedding_size,
                                output_size=output_size,
                                hidden_size=hidden_size,
                                projection_size=projection_size)
encoder_decoder.initialize_input_layers()

# model returns
# self.total_loss, self.avg_loss, logits, self.enc_states, self.dec_outputs, self.dec_states
# avg_loss is calculated as the softmax crossentropy over current batch
_, avg_loss, _, _, _, _, _ = encoder_decoder._seq2seq()

optimizer = tf.train.AdamOptimizer(adam_lr)
gradients = optimizer.compute_gradients(avg_loss)
capped_gradients = [(tf.clip_by_norm(grad, norm_clip), var) for grad, var in gradients]
train_op = optimizer.apply_gradients(capped_gradients)

# ---- training ----
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

count = 0
epoch_loss = 0.
epoch_count = 0
losses = []
saver = tf.train.Saver()

while True:
    current_epoch = reader.epoch
    data, index = reader.next_batch()
    enc_inp, dec_inp, dec_tar = helpers.data_processing(data, buckets[index], batch_size)

    if reader.epoch != current_epoch:
        print "\n----------end of epoch:" + str(reader.epoch-1) + "----------"
        print "    avg loss: " + str(epoch_loss/epoch_count)
        print "\n"

        losses.append(epoch_loss/epoch_count) # avg loss among this epoch
        epoch_loss = 0.
        epoch_count = 0

        cwd = os.getcwd()
        saver.save(sess, cwd+"/"+save_path+"/model.ckpt")
        print "Model saved"

        if reader.epoch == (max_epoch+1):
            break

    feed_dict = {
    encoder_decoder.enc_inputs: enc_inp,
    encoder_decoder.dec_inputs: dec_inp,
    encoder_decoder.targets: dec_tar
    }

    _, loss_t = sess.run([train_op, avg_loss], feed_dict)
    epoch_loss += loss_t # total loss among this epoch
    epoch_count += 1 # batch count among this epoch

    count += 1 # printing counter
    if count % 10 == 0:
        print str(loss_t) + " @ epoch: " + str(reader.epoch) + " count: "+ str(epoch_count * batch_size)

with open("./records/"+str(embedding_size)+"_training_loss.txt", "w") as f:
    for item in losses:
        f.write("%s\n" % item)
f.close
sess.close()
