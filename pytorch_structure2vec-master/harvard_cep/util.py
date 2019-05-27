from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
import cPickle as cp

def load_raw_data():
    print('loading data')
    raw_data_dict = {}
    for fname in ['train', 'valid', 'test']:
        d = []
        with open('./data/%s.txt' % fname, 'r') as f:
            for row in f:
                row = row.split()
                d.append( (row[0].strip(), float(row[1].strip())) )
        raw_data_dict[fname] = d
        print('%s: %d' % (fname, len(d)))
    return raw_data_dict

def find_weight_idx(lower, upper, weights, pce):
    for i in range(len(lower)):
        if pce >= lower[i] and pce < upper[i]:
            return i
    return -1

def resampling_idxes(d):
    print('resampling indices')
    labels = []
    for t in d['train']:
        labels.append(t[1])
    width = 0.05
    labels = np.array(labels, float)

    lower = min(labels)
    upper = max(labels)

    cnt = []
    for i in np.arange(lower, upper, width):
        num = ((labels >= i) & (labels < i + width)).sum()
        cnt.append(num)

    cnt = np.array(cnt, float)
    max_cnt = max(cnt)
    
    cur = lower
    region_tuples = []
    for i in range(len(cnt)):
        region_tuples.append((cur, cur + width, max_cnt / cnt[i]))
        cur += width

	pce_values = []
    for p in d['train']:
		pce_values.append(p[1])

    lower, upper, weights = zip(*region_tuples)
    sample_idxes = {}
    output_cnt = {}
    for i in range(len(lower)):
        sample_idxes.setdefault(i, [])
        output_cnt.setdefault(i, 0)
        
    for i in range(len(pce_values)):
        idx = int(pce_values[i] / width)
        sample_idxes[idx].append(i)

    total_samples = 0
    for i in sample_idxes:
        if len(sample_idxes[i]) > total_samples:
            total_samples = len(sample_idxes[i])

    total_samples = int(total_samples * len(weights))
    train_idxes = []
    for i in tqdm(range(total_samples)):
        idx = random.randint(0, len(weights) - 1)
        if output_cnt[idx] < len(sample_idxes[idx]):
            train_idxes.append(sample_idxes[idx][output_cnt[idx]])
        else:
            sample = random.randint(0, len(sample_idxes[idx]) - 1)
            train_idxes.append(sample_idxes[idx][sample])
        output_cnt[idx] += 1
	
    random.shuffle(train_idxes)
    return train_idxes

