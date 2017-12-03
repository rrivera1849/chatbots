"""
This file does all of the preprocessing necessary to ingest data from the twitter datasets.
"""
import collections
import json
import os
import logging
import sys
from optparse import OptionParser

from tqdm import tqdm

def build_vocabulary(dataset, min_word_frequency):
    word_count = collections.defaultdict(int)

    logging.info('Getting word count from dataset')
    pbar = tqdm(total=len(dataset))
    for dialog_1, dialog_2 in dataset:
        for word in dialog_1:
            word_count[word] += 1
        for word in dialog_2:
            word_count[word] += 1
        pbar.update(1)
    pbar.close()

    logging.info('Creating vocabulary')
    words = [word for word, count in word_count.items() if count >= min_word_frequency]

    vocabulary = dict(zip(words, range(len(words))))
    vocabulary['<UNK>'] = len(vocabulary)
    vocabulary['<SOL>'] = len(vocabulary)
    vocabulary['<EOL>'] = len(vocabulary)
    vocabulary['<PAD>'] = len(vocabulary) 
    return vocabulary

def dialog_to_idx(dialog, vocabulary):
    result = []

    for i, word in enumerate(dialog):
        result.append(vocabulary.get(word, vocabulary['<UNK>']))
    result = [vocabulary['<SOL>']] + result + [vocabulary['<EOL>']]

    return result

def dataset_to_idx(dataset, vocabulary):
    logging.info('Converting dataset words to vocabulary indexes')
    result = []

    pbar = tqdm(total=len(dataset))
    for i, dialogs in enumerate(dataset):
        dialog_1, dialog_2 = dialogs
        result.append((dialog_to_idx(dialog_1, vocabulary), dialog_to_idx(dialog_2, vocabulary)))
        pbar.update(1)
    pbar.close()

    return result

def main(options, args):
    dataset_path = options.dataset_path
    if options.big:
        dataset_name = 'twitter_en_big.txt'
        suffix = 'big_en'
    else:
        dataset_name = 'twitter_en.txt'
        suffix = 'en'

    dataset_path = os.path.join(dataset_path, dataset_name)
    dataset = []
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            dataset.append((lines[i].split(), lines[i+1].split()))

    vocabulary = build_vocabulary(dataset, options.min_word_frequency)
    dataset = dataset_to_idx(dataset, vocabulary)

    logging.info('Saving vocabulary and preprocessed dataset to: {}'.format(options.output_path))
    json.dump(vocabulary, \
              open(os.path.join(options.output_path, 'twitter_word_voc_{}.json'.format(suffix)), 'w'))
    json.dump(dataset, \
              open(os.path.join(options.output_path, 'twitter_word_data_{}.json'.format(suffix)), 'w'))

    return 0

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-d', '--dataset-path', dest='dataset_path', type=str, \
                      help='Path to Twitter Dataset, where twitter_en.txt and twitter_en_big.txt are located')
    parser.add_option('-b', dest='big', default=False, action='store_true', \
                      help='Whether or not to use twitter_en_big.txt')
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
