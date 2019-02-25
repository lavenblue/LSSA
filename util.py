import sys
import copy
import random
import numpy as np
from collections import defaultdict
import pandas as pd


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def batch_sequences_lengths(batch_sequences):
    lengths = []
    max_length = 0
    for seq in batch_sequences:
        lengths.append(len(seq))
        if len(seq) > max_length:
            max_length = len(seq)
    return max_length, lengths


def zero_padding_post(sequences, target_seq_length):
    for index, sub_seq in enumerate(sequences):
        sub_seq_length = len(sub_seq)
        if sub_seq_length < target_seq_length:
            minus_length = target_seq_length - sub_seq_length
            plus_sequence = [0]*minus_length
            sequences[index] += plus_sequence


def zero_padding_pre(sequences, target_seq_length):
    pad_seq = []
    for index, sub_seq in enumerate(sequences):
        seq = np.zeros([target_seq_length], dtype=np.int32)
        idx = target_seq_length - 1
        for i in reversed(sub_seq):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        pad_seq.append(seq)

    return pad_seq


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def generate_train_data(user_train, itemnum, maxlen):
    inputs_seq = []
    inputs_pos = []
    inputs_neg = []
    inputs_user = []
    print('generate_train_data-----')

    def sample(user):
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return (user, seq, pos, neg)

    for userid in user_train:
        [user, seq, pos, neg] = sample(userid)
        inputs_user.append(user)
        inputs_seq.append(seq)
        inputs_pos.append(pos)
        inputs_neg.append(neg)

    return inputs_user, inputs_seq, inputs_pos, inputs_neg


def generate_test_data(train, test, usernum, maxlen):
    inputs_seq = []
    inputs_user = []
    real_items = []
    print('generate_test_data-----')
    users = range(0, usernum)
    for u in users:
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        item_idx = test[u]

        inputs_user.append(u)
        inputs_seq.append(seq)
        real_items.append(item_idx)
    return inputs_user, inputs_seq, real_items


def sas_partition(fname):
    print('----------------split data-----------------')
    user_train = {}
    user_test = {}

    usernum = 0
    itemnum = 0
    data = pd.read_csv(fname, names=['user', 'sessions'], dtype='str')
    is_first_line = 1
    for line in data.values:
        if is_first_line:
            usernum = int(line[0])
            itemnum = int(line[1])
            is_first_line = 0
        else:
            user_id = int(line[0])
            sessions = [i for i in line[1].split('@')]
            size = len(sessions)
            the_first_session = [int(i)+1 for i in sessions[0].split(':')]
            tmp = copy.deepcopy(the_first_session)
            user_train[user_id] = tmp
            for j in range(1, size - 1):
                current_session = [int(it)+1 for it in sessions[j].split(':')]
                tmp = copy.deepcopy(current_session)
                user_train[user_id].extend(tmp)

            current_session = [int(it) + 1 for it in sessions[size - 1].split(':')]
            tmp = copy.deepcopy(current_session)
            user_train[user_id].extend(tmp[:-1])
            # item = random.choice(current_session)
            user_test[user_id] = tmp[-1]
    return [user_train, user_test, usernum-1, itemnum]


def precision_k(pre_top_k, true_items, top_k):
    user_number = len(true_items)
    hits = len(set(pre_top_k[:top_k]).intersection(set(true_items)))
    num_hits = 0.0
    NDCG = 0.0
    for i, p in enumerate(pre_top_k[:top_k]):
        if p in true_items and p not in pre_top_k[:i]:
            num_hits += 1.0
            NDCG += 1.0 / np.log2(i + 2.0)

    rec = hits / user_number
    prec = hits / top_k
    ndcg = NDCG / user_number
    return prec, rec, ndcg


def compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)