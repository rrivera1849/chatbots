
import os
import pickle
import numpy as np
from optparse import OptionParser

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

parser = OptionParser()

parser.add_option('--dataset-path', dest='dataset_path', type=str, default='./datasets/generative')
parser.add_option('--num-samples', dest='num_samples', type=int, default=5000)

parser.add_option('--rnn-dim', dest='rnn_dim', type=int, default=256)
parser.add_option('--batch-size', dest='batch_size', type=int, default=128)
parser.add_option('--num-epochs', dest='num_epochs', type=int, default=100)

(options, args) = parser.parse_args()

################################################################
# Data Preprocessing BEGIN
################################################################
text_path = os.path.join(options.dataset_path, 'twitter_en.txt')
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(text_path, 'r').read().split('\n')[:options.num_samples * 2]
for i, (input_text, target_text) in enumerate(zip(lines[::2], lines[1::2])):
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_len = max([len(x) for x in input_texts]) 
max_decoder_seq_len = max([len(x) for x in target_texts])

# Lookup to get index given a character
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Reverse lookup to decode sequences back into characters
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_len)
print('Max sequence length for outputs:', max_decoder_seq_len)
################################################################
# Data Preprocessing END
################################################################
encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_len, num_encoder_tokens), 
        dtype='float32')
decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_len, num_decoder_tokens),
        dtype='float32')
decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_len, num_decoder_tokens),
        dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1

    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1
        if t > 0:
            decoder_target_data[i, t-1, target_token_index[char]] = 1

def build_seq2seq(rnn_dim, num_encoder_tokens, num_decoder_tokens, sampling=True):
    """Builds a basic Seq2Seq model.

    Keyword Arguments:
        rnn_dim: hidden dimensionality of the LSTM cells
        num_encoder_tokens: total number of tokens in the encoder input
        num_decoder_tokens: total number of tokens in the decoder input
        sampling: if true, will define models that can be used to sample
    """
    # TODO: Add depth parameter
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(rnn_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder = LSTM(rnn_dim, return_sequences=True, return_state=True)
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

model, encoder_model, decoder_model = build_encoder_model(options.rnn_dim, num_encoder_tokens, num_decoder_tokens)
model.compile(optimizer='adam', loss='categorical_crossentropy')

class LossHistory(keras.callbacks.Callback): def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

loss_history = LossHistory()
model_checkpoint = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True)

history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    epochs=options.num_epochs,
                    validation_split=0.2,
                    batch_size=options.batch_size,
                    callbacks=[loss_history])
                    # callbacks=[model_checkpoint, loss_history])

losses = {}
losses['loss'] = loss_history.losses
losses['val_loss'] = history.history['val_loss']
pickle.dump(losses, open('seq2seq_loss_history.pkl', 'wb'))

# Inference Mode:
# 1. Encode the input and retrieve the initial decoder state
# 2. Run one step of the decoder with this initial state and the SOL token
# 3. Repeat with the current target token and current states
def decode_sequence(encoder_model, decoder_model, input_seq):
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
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_len):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1

        # Update states
        states_value = [h, c]

    return decoded_sentence

for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(encoder_model, decoder_model, input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

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

    decoded_sentence = decode_sequence(input_seq)
    print('>>> ', user_input)
    print('<<< ', decoded_sentence)
