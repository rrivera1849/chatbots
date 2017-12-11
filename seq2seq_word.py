
import os
import json
import math
import pdb
import sys
from optparse import OptionParser

import keras
import numpy as np
import seq2seq
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.sequence import pad_sequences

def get_data_iter(num_epochs, batch_size, dataset, voc,
                 max_encoder_seq_len, max_decoder_seq_len, 
                 shuffle=True):
    """Creates a data iterator that can be used to feed in batches of data to the 
       Seq2Seq model.
    """

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

def build_stacked_lstm(rnn_dim, depth, input_shape, return_sequences=False):
    layers = []
    if depth <= 1:
        layers.append(LSTM(rnn_dim, input_shape=input_shape, return_state=True, return_sequences=return_sequences))
        return layers
    
    layers.append(LSTM(rnn_dim, input_shape=input_shape, return_sequences=True))
    for i in range(depth-2):
        layers.append(LSTM(rnn_dim, return_sequences=True))
    layers.append(LSTM(rnn_dim, return_state=True, return_sequences=return_sequences))

    return layers

def run_lstm(layers, inputs, initial_state=None):
    if len(layers) == 1:
        return layers[0](inputs, initial_state=initial_state)

    print('NumLayers: {}'.format(len(layers)))
    last_output = layers[0](inputs, initial_state=initial_state)
    for l in range(1, len(layers)-1):
        last_output = layers[l](last_output)

    return layers[-1](last_output)

def build_seq2seq(voc_size, embed_dim, num_encoder_tokens, num_decoder_tokens, depth, sampling=True):
    """Builds a Seq2Seq model that operates on words.

    Keyword Arguments:
        voc_size: size of the vocabulary 
    """
    encoder_inputs = Input(shape=(num_encoder_tokens,))
    embedding = Embedding(voc_size, embed_dim)

    encoder = build_stacked_lstm(embed_dim, depth, input_shape=(num_encoder_tokens, embed_dim))
    encoder_embedded_inputs = embedding(encoder_inputs)
    encoder_outputs, state_h, state_c = run_lstm(encoder, encoder_embedded_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(num_decoder_tokens,))
    decoder = build_stacked_lstm(embed_dim, depth, input_shape=(num_decoder_tokens, embed_dim), return_sequences=True)
    decoder_embedded_inputs = embedding(decoder_inputs) 
    decoder_outputs, _, _= run_lstm(decoder, decoder_embedded_inputs, initial_state=encoder_states)
    decoder_dense = Dense(voc_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    if sampling:
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(embed_dim,))
        decoder_state_input_c = Input(shape=(embed_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = run_lstm(decoder, decoder_embedded_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return model, encoder_model, decoder_model

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
    model = build_seq2seq(len(voc), options.embed_size, 
                        max_encoder_seq_len, max_decoder_seq_len, 
                        options.depth)

    pdb.set_trace()

    train_iter = get_data_iter(None, options.batch_size, dataset, voc,
                               max_encoder_seq_len, max_decoder_seq_len)
    train_batches_per_epoch = int(math.floor(float(len(dataset)) / options.batch_size)) + 1 
    print('train_batches_per_epoch: {}'.format(train_batches_per_epoch))

    model.fit_generator(train_iter, steps_per_epoch=train_batches_per_epoch, epochs=options.num_epochs)

    return 0

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--dataset-path', dest='dataset_path', type=str, default='./datasets/preprocessed/')

    parser.add_option('--embed-size', dest='embed_size', type=int, default=1024)
    parser.add_option('--depth', dest='depth', type=int, default=3)
    parser.add_option('--batch-size', dest='batch_size', type=int, default=32)
    parser.add_option('--num-epochs', dest='num_epochs', type=int, default=100)
    (options, args) = parser.parse_args()

    sys.exit(main(options, args))

