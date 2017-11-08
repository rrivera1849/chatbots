
import collections
import json

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

def trim_inputs(l, max_length):
    for i, elem in enumerate(l):
        if len(elem) > max_length:
            l[i] = elem[:max_length]

    return l

class UDC(Dataset):
    def __init__(self, data_path, vocabulary_path, max_length=160, is_train=True):
        self.data = json.load(open(data_path, 'r'))
        self.vocabulary = json.load(open(vocabulary_path, 'r'))
        self.max_length = max_length
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            context, utterance = trim_inputs(self.data[index][:2], self.max_length)
            context = torch.from_numpy(np.asarray(context))
            utterance = torch.from_numpy(np.asarray(utterance))
            label = torch.zeros(1).fill_(self.data[index][2])

            data = collections.OrderedDict()
            data['context'] = context
            data['utterance'] = utterance
            data['label'] = label

            return data
        else:
            trimmed = trim_inputs(self.data[index], self.max_length)
            data = collections.OrderedDict()
            data['context'] = torch.from_numpy(np.asarray(trimmed[0]))
            data['utterance'] = torch.from_numpy(np.asarray(trimmed[1]))
            for i, distractor in enumerate(trimmed[2:]):
                data['distractor_{}'.format(i)] = torch.from_numpy(np.asarray(distractor))

            return data

    def __len__(self):
        return len(self.data)
