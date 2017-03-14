# from overrides import overrides

import numpy as np

class Gen_Data_loader(object):
    def __init__(self, batch_size, SEQ_LEN):
        self.batch_size = batch_size
        self.token_stream = []
        self.SEQ_LEN = SEQ_LEN
    def create_batches(self, data_file):
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.SEQ_LEN:
                    self.token_stream.append(parse_line)

        print len(self.token_stream) * 20
        self.num_batch = len(self.token_stream) / self.batch_size

        # remove small remainders, if necessary
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
            retval = self.sequence_batch[self.pointer]
            self.pointer = (self.pointer+1) % self.num_batch
            return retval

    def reset_pointer(self):
            self.pointer = 0

class Likelihood_Data_loader(Gen_Data_loader):
    # @overrides
    def __init__(self, batch_size, SEQ_LEN):
        Gen_Data_loader.__init__(self, batch_size, SEQ_LEN)
        self.likeli_flag = 1

if __name__ == "__main__":
    SEQ_LENGTH = 20
    # file path
    positive_file = 'save/real_data.txt'

    gen_loader = Gen_Data_loader(128, 20)
    gen_loader.create_batches(positive_file)

    # likely_loader = Likelihood_Data_loader(128)
    # likely_loader.create_batches(positive_file)
