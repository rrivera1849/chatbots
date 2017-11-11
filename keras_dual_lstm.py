
import os
import json
import sys
from optparse import OptionParser

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, LSTM, dot
from keras.preprocessing.sequence import pad_sequences
import numpy as np

parser = OptionParser()

parser.add_option('--num-epochs', type=int, default=10,
                  help='Number of epochs to use during training')
parser.add_option('--batch-size', type=int, default=128,
                  help='Number of batches to use during training and evaluation')
parser.add_option('--validate-every', type=int, default=1000, 
                  help='Validate model every |num| iterations')

parser.add_option('--embedding-dim', type=int, default=100,
                  help='Dimensionality of the word embedding vectors')
parser.add_option('--rnn-dim', type=int, default=256, 
                  help='Dimensionality of the LSTM hidden/cell vector')
parser.add_option('--max-context-length', type=int, default=160, 
                  help='Maximum length for each context')
parser.add_option('--max-utterance-length', type=int, default=80,
                  help='Maximum length for each utterance/distractor')

(options, args) = parser.parse_args()

dataset_path = 'datasets/ubuntu_2.0'
train_path = os.path.join(dataset_path, 'train_preprocessed.json')
validation_path = os.path.join(dataset_path, 'validation_preprocessed.json')
vocabulary_path = os.path.join(dataset_path, 'vocabulary.json')

train = np.array(json.load(open(train_path, 'r')))
validation = np.array(json.load(open(validation_path, 'r')))
vocab = json.load(open(vocabulary_path, 'r'))

contexts, utterances, labels = train[:,0], train[:,1], train[:,2]
val_contexts, val_utterances, val_distractors = validation[:,0], validation[:,1], validation[:,2:]

# Pad all of our train / validation sequences
contexts = pad_sequences(contexts, maxlen=options.max_context_length, value=len(vocab)) 
utterances = pad_sequences(utterances, maxlen=options.max_utterance_length, value=len(vocab))

val_contexts = pad_sequences(val_contexts, maxlen=options.max_context_length, value=len(vocab)) 
val_utterances = pad_sequences(val_utterances, maxlen=options.max_utterance_length, value=len(vocab))

distractors = []
for i in range(val_distractors.shape[1]):
    padded_distractor = pad_sequences(val_distractors[:, i], maxlen=options.max_utterance_length, value=len(vocab)) 
    distractors.append(padded_distractor)

##################################################
# Build Model START
##################################################
# Placeholders for context and utterance inputs
context_sequence = Input((options.max_context_length,))
utterance_sequence = Input((options.max_utterance_length,))

# Add Embedding Layer and LSTM network to encode our inputs
encoder = Sequential()
encoder.add(Embedding(len(vocab)+1, output_dim=options.embedding_dim))
encoder.add(LSTM(options.rnn_dim))

context_encoded = encoder(context_sequence)
utterance_encoded = encoder(utterance_sequence)

# M is a rnn_dim x rnn_dim matrix used to get the predicted utterance
M = Dense(options.rnn_dim, input_shape=(None, options.rnn_dim), use_bias=False)

# Get similarity and score for our inputs
predicted_utterance = M(context_encoded)
similarity = dot([predicted_utterance, utterance_encoded], axes=(1,1))
score = Activation('sigmoid')(similarity)

model = Model([context_sequence, utterance_sequence], score)
##################################################
# Build Model END 
##################################################

def recall_at_k(predictions, target=0, k=[1,2,5,10]):
    results = {}
    order = np.argsort(predictions, axis=1)

    for k_ in k:
        num_correct = 0 
        for row in order[:,-k_:]:
            num_correct += 1 if target in row else 0

        results['recall@{}'.format(k_)] = float(num_correct) / float(order.shape[0])

    return results

def evaluate_on_validation(epoch, logs, model):
    print('[{}/{}] Evaluating on validation:'.format(epoch, options.num_epochs))

    predictions = model.predict([val_contexts, val_utterances], batch_size=options.batch_size)
    for distractor in distractors:
        current_predictions = model.predict([val_contexts, distractor], batch_size=options.batch_size)
        predictions = np.concatenate((predictions, current_predictions), axis=1)

    results = recall_at_k(predictions)
    for k, v in results.items():
        print('\t{}: {}'.format(k,v))

# Compile the model in order to be ready for training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create callback to save a model checkpoint every epoch
filepath = 'dual_encoder_best.hdf5'
checkpointer = ModelCheckpoint(filepath, save_best_only=True, monitor='loss', mode='min')

# Create a callback to evaluate on our validation data every epoch
evaluate_model_cb = keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: evaluate_on_validation(epoch, logs, model))

model.fit([contexts, utterances], labels, batch_size=options.batch_size, epochs=options.num_epochs, callbacks=[checkpointer, evaluate_model_cb])
