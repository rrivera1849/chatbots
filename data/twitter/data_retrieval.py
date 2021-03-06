
import os
import pickle
import pdb
import sys
from random import shuffle

import numpy as np
import pandas as pd

def convert_to_numpy(series):
    result = []
    for index, value in series.iteritems():
        result.append([int(x) for x in value.split()])
    return np.array(result)

def load_data(path):
    df = pd.read_csv(os.path.join(path, 'twitter_retrieval.csv'))

    with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    return metadata, convert_to_numpy(df['context']), convert_to_numpy(df['utterance']), np.array(df['label'])

def load_np_data():
    q = np.load('idx_q.npy')
    a = np.load('idx_a.npy')

    data_len = len(q)
    train_len = int(0.7 * data_len)
    trainX, trainY = q[:train_len], a[:train_len]

    return trainX.tolist(), trainY.tolist()

def to_string(l):
    result = []
    for elem in l:
        elem = [str(x) for x in elem]
        result.append(' '.join(elem))
    return result

def main():
    trainX, trainY = load_np_data()
    trainX = to_string(trainX)
    trainY = to_string(trainY)

    data_correct = pd.DataFrame()
    data_correct['context'] = trainX
    data_correct['utterance'] = trainY
    data_correct['label'] = 1

    shuffle(trainY)

    data_incorrect = pd.DataFrame()
    data_incorrect['context'] = trainX
    data_incorrect['utterance'] = trainY
    data_incorrect['label'] = 0

    result = pd.concat([data_correct, data_incorrect]).reset_index(drop=True)
    result.to_csv('twitter_retrieval.csv')

    return 0

if __name__ == '__main__':
    sys.exit(main())
