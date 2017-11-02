
import array
import numpy as np
import tensorflow as tf
from collections import defaultdict

def load_vocab(filename):
    with open(filename) as f:
        vocab = f.readlines().splitlines()

    d = defaultdict(int)
    for idx, word in enumerate(vocab):
        d[word] = idx
    return [vocab, idx]

FLAGS = tf.flags.FLAGS

def dual_encoder_model(
        hparams,
        mode,
        context,
        context_len,
        utterance,
        utterance_len,
        targets):
    
    embeddings_W = tf.get_variable('word_embeddings', 
            shape=[hparams.vocab_size, hparams.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.25, 0.25))

    context_embedded = tf.nn.embedding_lookup(embeddings_W,
            context, name='embed_context')
    utterance_embedded = tf.nn.embedding_lookup(embeddings_W,
            utterance, name='embed_utterance')

    print('context_embedded: {}'.format(context_embedded.get_shape()))
    print('utterance_embedded: {}'.format(utterance_embedded.get_shape()))
    print('targets: {}'.format(targets.get_shape()))

    with tf.variable_scope('rnn') as vs:
        cell = tf.nn.rnn_cell.LSTMCell(
                hparams.rnn_dim,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell,
                tf.concat([context_embedded, utterance_embedded], 0),
                sequence_length=tf.concat([context_len, utterance_len], 0),
                dtype=tf.float32)
        print('rnn_states.h: {}'.format(rnn_states.h.get_shape()))
        encoding_context, encoding_utterance = tf.split(rnn_states.h, 2, axis=0)
        print('encoding_context: {}'.format(encoding_context.get_shape()))
        print('encoding_utterance: {}'.format(encoding_utterance.get_shape()))

    with tf.variable_scope('prediction') as vs:
        M = tf.get_variable('M',
                shape=[hparams.rnn_dim, hparams.rnn_dim],
                initializer=tf.truncated_normal_initializer())

        print('M: {}'.format(M.get_shape()))
        generated_response = tf.matmul(encoding_context, M)
        generated_response = tf.expand_dims(generated_response, 2)
        encoding_utterance = tf.expand_dims(encoding_utterance, 2)
        print('generated_response: {}'.format(generated_response.get_shape()))
        print('encoding_utterance: {}'.format(encoding_utterance.get_shape()))

        logits = tf.matmul(generated_response, encoding_utterance, True)
        print('logits: {}'.format(logits.get_shape()))
        logits = tf.squeeze(logits, [2])

        probs = tf.sigmoid(logits)
        losses = tf.losses.sigmoid_cross_entropy(tf.to_float(targets), logits)

    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    return probs, mean_loss
