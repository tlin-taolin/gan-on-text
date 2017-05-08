import numpy as np
import json
import simplejson


class g_data_loader(object):
    '''
    A dataloader for generator. The inputs are sentences and will be parsed to
    the same length (largest sequence length).
    '''
    def __init__(self, batch_size, largest_len, data_path):
        self.batch_size = batch_size
        self.largest_len = largest_len
        self.data_path = data_path

    def create_batches(self):
        self.sentences_stream = []
        f = open(self.data_path, "r")
        whole = simplejson.load(f)
        temp = []
        for item in whole:
            if len(temp) == self.largest_len:
                self.sentences_stream.append(temp)
                temp = []
            else:
                temp.append(item)
        f.close()
        self.num_batch = int(len(self.sentences_stream) / self.batch_size)
        self.sentences_stream = self.sentences_stream[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(np.array(self.sentences_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


if __name__ == "__main__":
    path = "./datasets/bbt_concate.txt"
    G = g_data_loader(batch_size=64, largest_len=20, data_path=path)
    G.create_batches()
