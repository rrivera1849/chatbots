
import tensorflow as tf
from collections import namedtuple

tf.flags.DEFINE_integer('vocab_size', 91595, 'Size of the vocabulary')

# Model Parameters
tf.flags.DEFINE_integer('embedding_dim', 100, 'Dimensionality of the embeddings')
tf.flags.DEFINE_integer('rnn_dim', 256, 'Dimensionality of the RNN cell')
tf.flags.DEFINE_integer('max_context_len', 160, 'Truncate contexts to this length')
tf.flags.DEFINE_integer('max_utterance_len', 80, 'Truncate utterance to this length')

# Training Parameters
tf.flags.DEFINE_string('vocab_path', None, 'Path to vocabulary.txt file')
tf.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size during training')
tf.flags.DEFINE_integer('eval_batch_size', 16, 'Batch size during evaluation')
tf.flags.DEFINE_string('optimizer', 'Adam', 'Optimizer Name (Adam, Adagrad, etc)')

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  'HParams',
  [
    'batch_size',
    'embedding_dim',
    'eval_batch_size',
    'learning_rate',
    'max_context_len',
    'max_utterance_len',
    'optimizer',
    'rnn_dim',
    'vocab_size',
    'vocab_path'
  ])

def create_hparams():
    return HParams(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        vocab_size=FLAGS.vocab_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        embedding_dim=FLAGS.embedding_dim,
        max_context_len=FLAGS.max_context_len,
        max_utterance_len=FLAGS.max_utterance_len,
        vocab_path=FLAGS.vocab_path,
        rnn_dim=FLAGS.rnn_dim
        )
