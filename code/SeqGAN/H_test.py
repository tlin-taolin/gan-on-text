import numpy as np

def significance_test(sess, target_lstm, data_loader, output_file):
    loss = []
    data_loader.reset_pointer()
    for it in xrange(data_loader.num_batch):
        Batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: Batch})
        # print g_loss
        loss.extend([g_loss])
    with open(output_file, "w") as fout:
        for item in loss:
            connexion = str(item) + "\n"
            fout.write(connexion)
    fout.close()
