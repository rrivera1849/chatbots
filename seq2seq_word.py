
import os
import json
import math
import pdb
import sys
from optparse import OptionParser

import keras
import seq2seq
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from seq2seq.models import AttentionSeq2Seq

def get_data_iter(num_epochs, batch_size, dataset, voc,
                 max_encoder_seq_len, max_decoder_seq_len, 
                 shuffle=True):

    input_texts, target_texts = zip(*dataset)
    num_samples = len(input_texts)
    num_batches_per_epoch = int(math.floor(float(num_samples) / float(batch_size))) + 1

    if num_epochs is None:
        num_epochs = sys.maxsize**10

    for epoch in range(num_epochs):
        if shuffle:
            perm = np.random.permutation(num_samples)
        else:
            perm = list(range(num_samples))

        for batch in range(num_batches_per_epoch):
            start = batch * batch_size
            end = min(start + batch_size, num_samples)

            encoder_input_data = pad_sequences(input_texts[start:end], maxlen=max_encoder_seq_len, padding='post', truncating='post', value=voc['<PAD>'])

            decoder_input = pad_sequences(target_texts[start:end], maxlen=max_decoder_seq_len, padding='post', truncating='post', value=voc['<PAD>'])
            decoder_input_data = decoder_input[:,:-1]
            decoder_target_data = decoder_input[:,1:]

            yield (encoder_input_data, decoder_input_data, decoder_target_data)


def build_model():
    pass

def main(options, args):
    dataset = json.load(open(os.path.join(options.dataset_path, 'twitter_word_data_en.json'), 'r'))
    voc = json.load(open(os.path.join(options.dataset_path, 'twitter_word_voc_en.json'), 'r'))
    max_encoder_seq_len = max([len(x[0]) for x in dataset])
    max_decoder_seq_len = max([len(x[1]) for x in dataset])

    data_iter = get_data_iter(None, options.batch_size, dataset, voc,
                              max_encoder_seq_len, max_decoder_seq_len)

    for (encoder_input_data, decoder_input_data, decoder_target_data) in data_iter:
        pdb.set_trace()

    return 0

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--dataset-path', dest='dataset_path', type=str, default='./datasets/preprocessed/')

    parser.add_option('--rnn-dim', dest='rnn_dim', type=int, default=256)
    parser.add_option('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_option('--num-epochs', dest='num_epochs', type=int, default=100)
    (options, args) = parser.parse_args()

    sys.exit(main(options, args))

