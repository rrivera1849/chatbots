
import os
import math
import pickle
import sys
from optparse import OptionParser

import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def get_data_iter(num_epochs, batch_size,
                 input_texts, target_texts, 
                 max_encoder_seq_len, num_encoder_tokens,
                 max_decoder_seq_len, num_decoder_tokens, 
                 input_token_index, target_token_index,
                 shuffle=True):
    """Creates a data iterator that can be used to feed in batches of data to the 
       Seq2Seq model.
    """
    num_samples = len(input_texts)
    num_batches_per_epoch = int(math.floor(float(num_samples) / float(batch_size))) + 1

    if num_epochs is None:
        num_epochs = sys.maxsize**10

    for epoch in range(num_epochs):
        perm = np.random.permutation(len(input_texts))

        for batch in range(num_batches_per_epoch):
            start = batch * batch_size
            end = min(start + batch_size, len(input_texts))

            encoder_input_data = np.zeros(
                    (end - start, max_encoder_seq_len, num_encoder_tokens), 
                    dtype='float32')
            decoder_input_data = np.zeros(
                    (end - start, max_decoder_seq_len, num_decoder_tokens),
                    dtype='float32')
            decoder_target_data = np.zeros(
                    (end - start, max_decoder_seq_len, num_decoder_tokens),
                    dtype='float32')

            for i, (input_text, target_text) in enumerate(zip(input_texts[start:end], target_texts[start:end])):
                for t, char in enumerate(input_text):
                    encoder_input_data[i, t, input_token_index[char]] = 1

                for t, char in enumerate(target_text):
                    decoder_input_data[i, t, target_token_index[char]] = 1
                    if t > 0:
                        decoder_target_data[i, t-1, target_token_index[char]] = 1

            yield ([encoder_input_data, decoder_input_data], decoder_target_data)

def build_seq2seq(rnn_dim, num_encoder_tokens, num_decoder_tokens, depth=2, sampling=True):
    """Builds a basic Seq2Seq model.

    Keyword Arguments:
        rnn_dim: hidden dimensionality of the LSTM cells
        num_encoder_tokens: total number of tokens in the encoder input
        num_decoder_tokens: total number of tokens in the decoder input
        sampling: if true, will define models that can be used to sample
    """
    # TODO: Add depth parameter
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(rnn_dim, return_state=True, input_shape=(None, num_encoder_tokens))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder = LSTM(rnn_dim, return_sequences=True, return_state=True, input_shape=(None,  num_decoder_tokens))
    decoder_outputs, _, _= decoder(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    if sampling:
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(rnn_dim,))
        decoder_state_input_c = Input(shape=(rnn_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return model, encoder_model, decoder_model

    return model

def main(options, args):
    experiment_dir = os.path.join('./checkpoints', options.experiment_id)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    dataset = pickle.load(open(options.dataset_path, 'rb'))
    input_texts = dataset['input_texts']
    target_texts = dataset['target_texts']
    val_input_texts = dataset['val_input_texts'] 
    val_target_texts = dataset['val_target_texts']
    num_encoder_tokens = dataset['num_encoder_tokens']
    num_decoder_tokens = dataset['num_decoder_tokens']
    max_encoder_seq_len = dataset['max_encoder_seq_len']
    max_decoder_seq_len = dataset['max_decoder_seq_len']
    input_token_index = dataset['input_token_index']
    target_token_index = dataset['target_token_index']
    reverse_input_token_index = dataset['reverse_input_token_index']
    reverse_target_token_index = dataset['reverse_target_token_index']

    model, encoder_model, decoder_model = build_seq2seq(options.rnn_dim, num_encoder_tokens, num_decoder_tokens)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    loss_history = LossHistory()

    train_iter = get_data_iter(None, options.batch_size, input_texts, target_texts,
                            max_encoder_seq_len, num_encoder_tokens, 
                            max_decoder_seq_len, num_decoder_tokens,
                            input_token_index, target_token_index)

    val_iter = get_data_iter(None, options.batch_size, val_input_texts, val_target_texts,
                            max_encoder_seq_len, num_encoder_tokens, 
                            max_decoder_seq_len, num_decoder_tokens,
                            input_token_index, target_token_index)

    train_steps_per_epoch = int(math.floor(float(len(input_texts)) / float(options.batch_size))) + 1
    val_steps_per_epoch = int(math.floor(float(len(val_input_texts)) / float(options.batch_size))) + 1

    print('input_texts: {}'.format(len(input_texts)))
    print('train_steps_per_epoch: {}'.format(train_steps_per_epoch))
    print('val_input_texts: {}'.format(len(val_input_texts)))
    print('val_steps_per_epoch: {}'.format(val_steps_per_epoch))

    history = model.fit_generator(train_iter, train_steps_per_epoch, epochs=options.num_epochs,
                                validation_data=val_iter, validation_steps=val_steps_per_epoch,
                                callbacks=[loss_history])
    model.save(os.path.join(experiment_dir, 'model.hdf5'))
    encoder_model.save(os.path.join(experiment_dir, 'encoder_model.hdf5'))
    decoder_model.save(os.path.join(experiment_dir, 'decoder_model.hdf5'))

    losses = {}
    losses['loss'] = loss_history.losses
    losses['val_loss'] = history.history['val_loss']
    pickle.dump(losses, open(os.path.join(experiment_dir, 'seq2seq_loss_history.pkl'), 'wb'))

    return 0 

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--dataset-path', dest='dataset_path', type=str, default='./data/twitter/twitter_char_100000.pkl')
    parser.add_option('--experiment-id', dest='experiment_id', type=str, default='seq2seq_char')

    parser.add_option('--rnn-dim', dest='rnn_dim', type=int, default=256)
    parser.add_option('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_option('--num-epochs', dest='num_epochs', type=int, default=100)
    (options, args) = parser.parse_args()

    if not options.experiment_id:
        parser.error('Must provide an experiment identifier')

    sys.exit(main(options, args))
