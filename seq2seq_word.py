
import os
import json
import math
import pdb
import sys
from optparse import OptionParser

import keras
import numpy as np
import seq2seq
from keras.models import Model
from keras.layers import Embedding, Input
from keras.preprocessing.sequence import pad_sequences
from seq2seq.models import Seq2Seq, AttentionSeq2Seq

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
            decoder_input = decoder_input[:,1:]

            decoder_target_data = np.zeros((end - start, max_decoder_seq_len, len(voc)), 
                                           dtype='float32')

            for i in range(end - start):
                for t in range(max_decoder_seq_len-1):
                    index = decoder_input[i][t]
                    decoder_target_data[i][t][index] = 1

            yield (encoder_input_data, decoder_target_data)


def build_model(batch_size, voc_size, embed_size, 
                max_encoder_seq_len, max_decoder_seq_len,
                rnn_dim, depth, bidirectional):

    encoder_inputs = Input(shape=(max_encoder_seq_len,))
    embedding = Embedding(voc_size, embed_size)

    seq2seq = AttentionSeq2Seq(input_dim=embed_size, input_length=max_encoder_seq_len, 
                               output_dim=voc_size, output_length=max_decoder_seq_len, 
                               hidden_dim=rnn_dim, depth=options.depth, bidirectional=options.bidirectional)

    embedded_input = embedding(encoder_inputs)
    output = seq2seq(embedded_input)

    model = Model([encoder_inputs], output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def main(options, args):
    dataset = json.load(open(os.path.join(options.dataset_path, 'twitter_word_data_en.json'), 'r'))
    voc = json.load(open(os.path.join(options.dataset_path, 'twitter_word_voc_en.json'), 'r'))

    encoder_seq_len = [len(x[0]) for x in dataset]
    decoder_seq_len = [len(x[1]) for x in dataset]
    max_encoder_seq_len = max(encoder_seq_len)
    max_decoder_seq_len = max(decoder_seq_len)

    print('num_samples: {}'.format(len(dataset)))
    print('max_encoder_seq_len: {}'.format(max_encoder_seq_len))
    print('max_decoder_seq_len: {}'.format(max_decoder_seq_len))

    model = build_model(options.batch_size, len(voc), options.embed_size, 
                        max_encoder_seq_len, max_decoder_seq_len, 
                        options.rnn_dim, options.depth, options.bidirectional)

    train_iter = get_data_iter(None, options.batch_size, dataset, voc,
                               max_encoder_seq_len, max_decoder_seq_len)
    train_batches_per_epoch = int(math.floor(float(len(dataset)) / options.batch_size)) + 1 
    print('train_batches_per_epoch: {}'.format(train_batches_per_epoch))

    model.fit_generator(train_iter, steps_per_epoch=train_batches_per_epoch, epochs=options.num_epochs)

    return 0

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--dataset-path', dest='dataset_path', type=str, default='./datasets/preprocessed/')

    parser.add_option('--rnn-dim', dest='rnn_dim', type=int, default=256)
    parser.add_option('--embed-size', dest='embed_size', type=int, default=128)
    parser.add_option('--depth', dest='depth', type=int, default=1)
    parser.add_option('--no-bidirectional', dest='bidirectional', action='store_false', default=True)
    parser.add_option('--batch-size', dest='batch_size', type=int, default=64)
    parser.add_option('--num-epochs', dest='num_epochs', type=int, default=100)
    (options, args) = parser.parse_args()

    sys.exit(main(options, args))

