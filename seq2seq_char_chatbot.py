# Code inspired from: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

import os
import numpy as np
import pickle
import sys
from optparse import OptionParser

import keras

# Inference Mode:
# 1. Encode the input and retrieve the initial decoder state
# 2. Run one step of the decoder with this initial state and the SOL token
# 3. Repeat with the current target token and current states
def decode_sequence(encoder_model, decoder_model, input_seq, 
                    num_decoder_tokens, max_decoder_seq_len,
                    target_token_index, reverse_target_token_index):
    """Given an |encoder_model| and a |decoder_model| built for sampling. This 
       method will decode a sequence given the |input_seq|

    Keyword Arguments:
        encoder_model: model used to encode then input sequence into a context vector
        decoder_model: model used to decode response given context vector
        input_seq: sequence to use as input
    """
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_token_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_len):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1

        # Update states
        states_value = [h, c]

    return decoded_sentence

def main(options, args):
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

    model = keras.models.load_model(os.path.join(options.checkpoint_path, 'model.hdf5'))
    encoder_model = keras.models.load_model(os.path.join(options.checkpoint_path, 'encoder_model.hdf5'))
    decoder_model = keras.models.load_model(os.path.join(options.checkpoint_path, 'decoder_model.hdf5'))

    while True:
        print('-')
        user_input = input('Speak to the chatbot!\n')
        if len(user_input) <= 0:
            break

        input_seq = np.zeros((1, max_encoder_seq_len, num_encoder_tokens))
        for t, char in enumerate(user_input):
            input_seq[0][t][input_token_index[char]] = 1
            if t >= max_encoder_seq_len:
                break

        decoded_sentence = decode_sequence(encoder_model, decoder_model, input_seq, 
                                           num_decoder_tokens, max_decoder_seq_len,
                                           target_token_index, reverse_target_token_index)
        print('>>> ', user_input)
        print('<<< ', decoded_sentence)


    return 0

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--dataset-path', dest='dataset_path', type=str, default='./data/twitter/twitter_char_100000.pkl')
    parser.add_option('--checkpoint-path', dest='checkpoint_path', type=str, default='./checkpoints')
    (options, args) = parser.parse_args()

    sys.exit(main(options, args))
