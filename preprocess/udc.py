"""
This file does all of the preprocessing necessary to ingest data from the 
Ubuntu Dialog Corpus.
"""

import os
import logging
import json
import sys
from collections import defaultdict
from optparse import OptionParser

import pandas as pd
from tqdm import tqdm

def build_vocabulary(df, min_frequency):
    word_count = defaultdict(int)
    pbar = tqdm(total=len(df))
    for i, row in df.iterrows():
        context = row['Context']
        for word in context.split():
            word_count[word] += 1
        pbar.update(1)

    words = [word for word, count in word_count.items() if count >= min_frequency]

    vocabulary = dict(zip(words, range(len(words))))
    vocabulary['<UNK>'] = len(words)
    vocabulary['<PAD>'] = len(words) + 1
    return vocabulary

def text_to_idx(text, vocabulary):
    indices = []
    for word in  text.split():
        if word in vocabulary:
            indices.append(vocabulary[word])
        else:
            indices.append(vocabulary['<UNK>'])

    return indices

def process_train(df, vocabulary):
    dataset = []
    pbar = tqdm(total=len(df))
    for i, row in df.iterrows():
        context_indices = text_to_idx(row['Context'], vocabulary)
        utterance_indices = text_to_idx(row['Utterance'], vocabulary)
        dataset.append((context_indices, utterance_indices, int(row['Label'])))
        pbar.update(1)

    return dataset

def process_test(df, vocabulary):
    dataset = []
    pbar = tqdm(total=len(df))
    for i, row in df.iterrows():
        context_indices = text_to_idx(row['Context'], vocabulary)
        utterance_indices = text_to_idx(row['Ground Truth Utterance'], vocabulary)
        example = [context_indices, utterance_indices]

        for j in range(9):
            distractor_indices = text_to_idx(row['Distractor_{}'.format(j)], vocabulary)
            example.append(distractor_indices)

        dataset.append(tuple(example))
        pbar.update(1)

    return dataset

def main(options, args):
    train_path = os.path.join(options.dataset_path, 'train.csv')
    validation_path = os.path.join(options.dataset_path, 'valid.csv')
    test_path = os.path.join(options.dataset_path, 'test.csv')

    if not os.path.exists(train_path) or not os.path.exists(validation_path) \
            or not os.path.exists(test_path):
        logging.error('One of train.csv, valid.csv or test.csv does not exist in the directory provided')

    train = pd.read_csv(train_path)

    logging.info('Creating vocabulary from training data')
    vocabulary = build_vocabulary(train, options.min_word_frequency)

    logging.info('Preprocessing training data')
    preprocessed_train = process_train(train, vocabulary)
    logging.info('Preprocessing validation data')
    preprocessed_validation = process_test(pd.read_csv(validation_path), vocabulary)
    logging.info('Preprocessing test data')
    preprocessed_test = process_test(pd.read_csv(test_path), vocabulary)

    logging.info('Saving preprocessed data to: {}'.format(options.output_path))
    json.dump(vocabulary,
              open(os.path.join(options.output_path, 'vocabulary.json'), 'w+'))
    json.dump(preprocessed_train, 
              open(os.path.join(options.output_path, 'train_preprocessed.json'), 'w+'))
    json.dump(preprocessed_validation, 
              open(os.path.join(options.output_path, 'validation_preprocessed.json'), 'w+'))
    json.dump(preprocessed_test, 
              open(os.path.join(options.output_path, 'test_preprocessed.json'), 'w+'))

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-d', '--dataset-path', dest='dataset_path', type=str, \
                      help='Path to UDC, where train.csv, valid.csv and test.csv are located')
    parser.add_option('-m', '--min-word-frequency', dest='min_word_frequency', type=int, default=5, \
                      help='Minimum number of times that word must appear to be added to the vocabulary.')
    parser.add_option('-o', '--output-path', dest='output_path', type=str, \
                      help='Path to store preprocessed dataset')
    (options, args) = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if not options.dataset_path:
        parser.error('Must provide dataset_path where train.csv, valid.csv and test.csv are located')

    if not options.output_path:
        parser.error('Must provide output_path to store the preprocessed dataset')

    sys.exit(main(options, args))
