
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
        encoding_context, encoding_utterance = tf.split(rnn_states.h, 2, axis=0)

        with tf.variable_scope('prediction') as vs:
            M = tf.get_variable('M',
                    shape=[hparams.rnn_dim, hparams.rnn_dim],
                    initializer=tf.truncated_normal_initializer())

            generated_response = tf.matmul(encoding_context, M)
            generated_response = tf.expand_dims(generated_response, 2)
            encoding_utterance = tf.expand_dims(encoding_utterance, 2)

            logits = tf.matmul(generated_response, encoding_utterance, True)
            logits = tf.squeeze(logits, [2])

            probs = tf.sigmoid(logits)

            if mode == tf.contrib.learn.ModeKeys.INFER:
                return probs, None

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))

        mean_loss = tf.reduce_mean(losses, name='mean_loss')
        return probs, mean_loss
