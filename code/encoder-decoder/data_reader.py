import numpy as np
import os
import json

class reader(object):
    def __init__(self, file_name, batch_size, buckets, bucket_option,
                                               signal=False, clean_mode=True):
        self.epoch = 1
        self.batch_size = batch_size
        self.file_name = file_name
        self.buckets = buckets
        self.bucket_option = bucket_option
        self.output = []
        self.bucket_list = []
        self.clean_stock = False
        self.clean_mode = clean_mode
        self.bucket_dict = self.build_bucket_dict()

        for i in xrange(len(buckets)):
            self.bucket_list.append([])

        self.file = open(file_name+"/filtered_text.txt", "r")
        with open(file_name+"/dict.json", "r") as f:
            self.dict = json.load(f)

        if signal:
            with open(file_name+"/signal.json", "r") as f:
                self.signal_dict = json.load(f)

        self.id_dict = {}
        for key in self.dict:
            self.id_dict[self.dict[key]] = key

    def next_batch(self):
        if not self.clean_stock:
            index = self.fill_bucket()
            if index >= 0:
                output = self.bucket_list[index]
                self.bucket_list[index] = []
                return output, index
            else:
                self.clean_stock = True
                for i in xrange(len(self.bucket_list)):
                    if len(self.bucket_list[i]) > 0:
                        output = self.bucket_list[i]
                        self.bucket_list[i] = []
                        return output, i
        elif self.clean_mode:
            for i in xrange(len(self.bucket_list)):
                if len(self.bucket_list[i]) > 0:
                    output = self.bucket_list[i]
                    self.bucket_list[i] = []
                    return output, i
            self.clean_stock = False
            self.reset()
            return self.next_batch()

    def reset(self):
        self.epoch += 1
        self.file.close()
        self.file = open(self.file_name + "/filtered_text.txt", "r")

    def build_bucket_dict(self):
        bucket_dict = {}
        for i in xrange(1, self.bucket_option[-1] + 1):
            count = len(self.bucket_option) - 1
            for options in reversed(self.bucket_option):
                if options >= i:
                    bucket_dict[i] = count
                count -= 1
        return bucket_dict

    def check_bucket(self, pair):
        best_i = self.bucket_dict[len(pair[0])]
        best_j = self.bucket_dict[len(pair[1])]
        return best_i * len(self.bucket_option) + best_j

    def fill_bucket(self):
        while True:
            line = self.file.readline()
            if not line:
                break
            pair = json.loads(line)
            index = self.check_bucket(pair)
            if index == -1:
                continue
            self.bucket_list[index].append(pair)
            if len(self.bucket_list[index]) == self.batch_size:
                return index
        return -1
