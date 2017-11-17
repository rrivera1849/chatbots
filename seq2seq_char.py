
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np
from scipy.sparse import csr_matrix

dataset_path = './datasets/generative'
rnn_dim = 256
batch_size = 128
num_epochs = 10
num_samples = 50000
text_path = os.path.join(dataset_path, 'twitter_en.txt')

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(text_path, 'r').read().split('\n')[:num_samples * 2]
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

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_len)
print('Max sequence lenght for outputs:', max_decoder_seq_len)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

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
        if t >= max_encoder_seq_len:
            continue

        encoder_input_data[i, t, input_token_index[char]] = 1

    for t, char in enumerate(target_text):
        if t >= max_decoder_seq_len:
            continue

        decoder_input_data[i, t, target_token_index[char]] = 1
        if t > 0:
            decoder_target_data[i, t, target_token_index[char]] = 1

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(rnn_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder = LSTM(rnn_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

model_checkpoint = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[model_checkpoint])
