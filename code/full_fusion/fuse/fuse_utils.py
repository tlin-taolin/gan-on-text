from __future__ import division
from os.path import join, exists

import re
import pickle
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
    return data


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def write_pickle(data, path):
    print("write --> data to path: {}\n".format(path))
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def load_data(force):
    # path_root = '/Users/mzhao/Desktop/nnMisc/full/fuse/data/can_emb'
    path_root = '/home/tlin/notebooks/code/full_fusion/fuse/data/can_emb/'
    path_emb = join(path_root, 'emb', 'total_emb.npy')
    path_candidate = join(path_root, 'candidates')
    path_candidates_argmax = join(path_candidate, 'argmax_list.txt')
    path_candidates_argmax_p = join(path_candidate, 'argmax_list.pickle')
    path_candidates_pick = join(path_candidate, 'weighted_pick_list.txt')
    path_candidates_pick_p = join(path_candidate, 'weighted_pick_list.pickle')
    path_candidates_beam = join(path_candidate, 'beam_search_list.txt')
    path_candidates_beam_p = join(path_candidate, 'beam_search_list.pickle')

    # load embedding.
    embedding = np.load(path_emb)

    # load candidates
    candidate_argmax, candidates_beam, candidates_pick = None, None, None

    if not force or not exists(path_candidates_beam_p):
        candidates_beam = read_txt(path_candidates_beam, size=3)
        write_pickle(candidates_beam, path_candidates_beam_p)
    else:
        print "Loading existing pickle..."
        candidates_beam = load_pickle(path_candidates_beam_p)

    ''' Embedding size: 256
        vocab_size: 20525
        Totoal beam candidates: 742239
        None for candidate_argmax, candidates_pick
    '''
    '''
       Form of beam candidates:
            _________
                     |
                     |
                     |
           [Q, A, A']| 21 pairs. A is the truth answering.
                     |
                     |
            _________|

    '''
    return embedding, candidate_argmax, candidates_beam, candidates_pick


def build_up_candidates(cand_beam):
    # Note that I use `99999` to denote padding symbols. please correct it.
    candidates = []
    candidate_max_length = max([len(candidate[2]) for candidate in cand_beam])
    tuples = make_square(cand_beam, 21)

    for tuple in tuples:
        head = tuple[0][0: 2]
        candidate = map(lambda line: line[2], tuple)
        # candidate_length_max = max(map(lambda c: len(c), candidate))
        candidate = [
            c + [0] * (candidate_max_length - len(c)) for c in candidate]
        candidates.append(head + candidate)
    return candidates, candidate_max_length


def build_batches(candidates, batch_size, num_batches):
    Q_list = [candidate[0] for candidate in candidates]
    A_list = [candidate[1] for candidate in candidates]
    Generated_A_list = [candidate[2:] for candidate in candidates]

    Q_batches = [
        Q_list[batch_th * batch_size: (batch_th + 1) * batch_size]
        for batch_th in range(num_batches)]
    A_batches = [
        A_list[batch_th * batch_size: (batch_th + 1) * batch_size]
        for batch_th in range(num_batches)]
    GA_batches = [
        Generated_A_list[batch_th * batch_size: (batch_th + 1) * batch_size]
        for batch_th in range(num_batches)]
    return Q_batches, A_batches, GA_batches
