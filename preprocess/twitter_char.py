
import logging
import os
import pickle
import sys
from optparse import OptionParser

from tqdm import tqdm

def preprocess_lines(lines):
    logging.info('Preprocessing raw lines')

    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    pbar = tqdm(total=len(lines[::2]))
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

        pbar.update(1)
    pbar.close()

    return input_texts, target_texts, input_characters, target_characters

def split_data(input_texts, target_texts, val_split=0.2):
    # Split data into train and validation
    logging.info('Splitting data into training and validation')
    val_split_index = int(val_split * len(input_texts))

    val_input_texts = input_texts[:val_split_index]
    val_target_texts = target_texts[:val_split_index]

    input_texts = input_texts[val_split_index:]
    target_texts = target_texts[val_split_index:]

    return input_texts, target_texts, val_input_texts, val_target_texts

def main(options, args):
    if options.num_samples:
        lines = open(options.dataset_path, 'r').read().split('\n')[:options.num_samples*2]
    else:
        lines = open(options.dataset_path, 'r').read().split('\n')

    input_texts, target_texts, input_characters, target_characters = preprocess_lines(lines)

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_len = max([len(x) for x in input_texts]) 
    max_decoder_seq_len = max([len(x) for x in target_texts])

    input_texts, target_texts, val_input_texts, val_target_texts = split_data(input_texts, target_texts)

    # Lookup to get index given a character
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    # Reverse lookup to decode sequences back into characters
    reverse_input_token_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_token_index = dict((i, char) for char, i in target_token_index.items())

    logging.info('Number of samples {}'.format(len(input_texts) + len(val_input_texts)))
    logging.info('Number of unique input tokens {}'.format(num_encoder_tokens))
    logging.info('Number of unique output tokens {}'.format(num_decoder_tokens))
    logging.info('Max sequence length for inputs {}'.format(max_encoder_seq_len))
    logging.info('Max sequence length for outputs {}'.format(max_decoder_seq_len))

    dataset = {}
    dataset['input_texts'] = input_texts
    dataset['target_texts'] = target_texts
    dataset['val_input_texts'] = val_input_texts
    dataset['val_target_texts'] = val_target_texts
    dataset['num_encoder_tokens'] = num_encoder_tokens
    dataset['num_decoder_tokens'] = num_decoder_tokens
    dataset['max_encoder_seq_len'] = max_encoder_seq_len
    dataset['max_decoder_seq_len'] = max_decoder_seq_len
    dataset['input_token_index'] = input_token_index
    dataset['target_token_index'] = target_token_index
    dataset['reverse_input_token_index'] = reverse_input_token_index
    dataset['reverse_target_token_index'] = reverse_target_token_index

    name = 'twitter_char_{}.pkl'.format(len(input_texts) + len(val_input_texts))
    path = os.path.join(options.output_path, name)
    logging.info('Saving data to {}'.format(path))
    pickle.dump(dataset, open(path, 'wb'))

    return 0

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--dataset-path', dest='dataset_path', type=str, default='./datasets/generative/twitter_en.txt')
    parser.add_option('--output-path', dest='output_path', type=str, default='./datasets/preprocessed')
    parser.add_option('--num-samples', dest='num_samples', type=int, default=None)
    (options, args) = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    sys.exit(main(options, args))
