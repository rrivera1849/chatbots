"""
This script formats the Ubuntu Dialog Corpus data into TFRecords 
for ease of use within our models.
"""

import array
import csv
import itertools
import functools
import os

import numpy as np
import tensorflow as tf

tf.flags.DEFINE_integer(
    'min_word_frequency', 5, 'Minimum frequency of words in the vocabulary')

tf.flags.DEFINE_integer('max_sentence_len', 160, 'Maximum sentence length')

tf.flags.DEFINE_string('input_dir', 
        os.path.abspath('./datasets/ubuntu_2.0'), 
        'Input directory containing original CSV data files. (default=\'./datasets/ubuntu_2.0\')')

tf.flags.DEFINE_string('output_dir',
        os.path.abspath('./datasets/ubuntu_2.0'),
        'Output directory for TFRecord files (default=\'./datasets/ubuntu_2.0\')')

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, 'train.csv')
VALIDATION_PATH = os.path.join(FLAGS.input_dir, 'valid.csv')
TEST_PATH = os.path.join(FLAGS.input_dir, 'test.csv')

def tokenizer_fn(iterator):
    return (x.split() for x in iterator)

def create_csv_iter(filename):
    """Creates a CSV file iterator, skips the header.
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            yield row

def create_vocab(input_iter, min_frequency):
    """Creates a TF VocabularyProcessor and fits it using the input data.
    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
            FLAGS.max_sentence_len,
            min_frequency=min_frequency,
            tokenizer_fn=tokenizer_fn
            )
    vocab_processor.fit(input_iter)

    return vocab_processor

def transform_sentence(sequence, vocab_processor):
    """Transform words in sequence to id's using the vocabulary processor.
    """
    return next(vocab_processor.transform([sequence])).tolist()

def create_example_train(row, vocab):
    """Creates a tf.train.Example object for the current row of 
       training data.
    """
    context, utterance, label = row
    context_transformed = transform_sentence(context, vocab)
    utterance_transformed = transform_sentence(utterance, vocab)
    context_len = len(next(vocab._tokenizer([context])))
    utterance_len = len(next(vocab._tokenizer([utterance])))
    label = int(float(label))

    example = tf.train.Example()
    example.features.feature['context'].int64_list.value.extend(context_transformed)
    example.features.feature['utterance'].int64_list.value.extend(utterance_transformed)
    example.features.feature['context_len'].int64_list.value.extend([context_len])
    example.features.feature['utterance_len'].int64_list.value.extend([utterance_len])
    example.features.feature['label'].int64_list.value.extend([label])

    return example

def create_example_test(row, vocab):
    """Creates a tf.train.Example object for the current row of
       testing data.
    """
    context, utterance = row[:2]
    distractors = row[2:]
    context_transformed = transform_sentence(context, vocab)
    utterance_transformed = transform_sentence(utterance, vocab)
    context_len = len(next(vocab._tokenizer([context])))
    utterance_len = len(next(vocab._tokenizer([utterance])))

    example = tf.train.Example()
    example.features.feature['context'].int64_list.value.extend(context_transformed)
    example.features.feature['utterance'].int64_list.value.extend(utterance_transformed)
    example.features.feature['context_len'].int64_list.value.extend([context_len])
    example.features.feature['utterance_len'].int64_list.value.extend([utterance_len])

    for i, distractor in enumerate(distractors):
        dis_key = 'distractor_{}'.format(i)
        dis_len_key = 'distractor_{}_len'.format(i)

        dis_transformed = transform_sentence(distractor, vocab)
        example.features.feature[dis_key].int64_list.value.extend(dis_transformed)

        dis_len = len(next(vocab._tokenizer([distractor])))
        example.features.feature[dis_len_key].int64_list.value.extend([dis_len])

    return example

def create_tfrecords_file(input_filename, output_filename, example_fn):
    """Creates the tfrecords file for the given input CSV file from the 
       ubuntu dialog corpus.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    print('Creating TFRecords file at: {}'.format(output_filename))
    for i, row in enumerate(create_csv_iter(input_filename)):
        x = example_fn(row)
        writer.write(x.SerializeToString())
    writer.close()
    print('Wrote to: {}'.format(output_filename))

def write_vocabulary(vocab_processor, outfile):
    """Writes the inverse vocabulary to a tex file.
    """
    vocab_size = len(vocab_processor.vocabulary_)
    with open(outfile, 'w') as vocabfile:
        for id in range(vocab_size):
            word = vocab_processor.vocabulary_._reverse_mapping[id]
            vocabfile.write(word + '\n')
    print('Saved vocabulary to: {}'.format(outfile))

if __name__ == '__main__':
    print('Creating vocabulary')
    input_iter = create_csv_iter(TRAIN_PATH)
    input_iter = (x[0] + " " + x[1] for x in input_iter)
    vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency) 
    write_vocabulary(vocab, os.path.join(FLAGS.output_dir, 'vocabulary.txt'))

    create_tfrecords_file(
        input_filename=VALIDATION_PATH,
        output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"),
        example_fn=functools.partial(create_example_test, vocab=vocab)
        )

    create_tfrecords_file(
        input_filename=TEST_PATH,
        output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"),
        example_fn=functools.partial(create_example_test, vocab=vocab)
        )

    create_tfrecords_file(
        input_filename=TRAIN_PATH,
        output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
        example_fn=functools.partial(create_example_train, vocab=vocab)
        )
