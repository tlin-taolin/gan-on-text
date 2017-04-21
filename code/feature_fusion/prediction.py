# -*- coding: utf-8 -*-

import re
from os.path import join
import numpy as np


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
    print('number of sentence in first instance: {}'.format(len(data[0])))
    return data


def load_data():
    """load dataset."""
    # define path.
    path_root = 'data'
    path_emb = join(path_root, 'emb', 'total_emb.npy')
    path_candidate = join(path_root, 'candidates')
    path_candidates_argmax = join(path_candidate, 'argmax_list.txt')
    path_candidates_pick = join(path_candidate, 'weighted_pick_list.txt')
    path_candidates_beam = join(path_candidate, 'beam_search_list.txt')

    # load data.
    # load embedding.
    embedding = np.load(path_emb)

    # load candidates
    candidate_argmax = read_txt(path_candidates_argmax, size=3)
    candidates_beam = read_txt(path_candidates_beam, size=3)
    candidates_pick = read_txt(path_candidates_pick, size=3)
    return embedding, candidate_argmax, candidates_beam, candidates_pick


def main():
    """the main entry."""
    load_data()


if __name__ == '__main__':
    main()
