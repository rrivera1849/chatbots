
import os
import pdb
import time
import sys
from optparse import OptionParser

import numpy as np
import keras
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from data.twitter import data
from data.twitter import data_retrieval
from seq2seq_word import build_seq2seq_word, load_data, prepare_vocab

parser = OptionParser()
parser.add_option('--data-path', type=str, default='./data/twitter/')
parser.add_option('--checkpoint-path', type=str, default='./lstm.npz')
parser.add_option('--dual-encoder-path', type=str, default='./twitter_dual_encoder_best.hdf5')

parser.add_option('--batch-size', type=int, default=32)
parser.add_option('--embedding-dim', type=int, default=1024)
parser.add_option('--dropout', type=float, default=0.5)
parser.add_option('--nlayers', type=int, default=3)

def pad(x, maxlen):
    pad_width = maxlen - len(x)
    if pad_width >= 1:
        arr = np.array(x)
        arr = np.pad(arr, (0, pad_width), mode='constant', constant_values=(0,))
    else:
        x = x[:maxlen]
        arr = np.array(x)

    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    if options.dual_encoder_path:
        print('Loading Dual Encoder')
        metadata, _, _, _ = data_retrieval.load_data('./data/twitter')
        de_vocab = metadata['w2idx']
        model = keras.models.load_model(options.dual_encoder_path)

    trainX, trainY, testX, testY, validX, validY, metadata = load_data(options.data_path)

    xseq_len = len(trainX)
    yseq_len = len(trainY)
    assert xseq_len == yseq_len

    n_step = int(xseq_len / options.batch_size)

    w2idx, idx2w, xvocab_size = prepare_vocab(metadata['w2idx'], metadata['idx2w'])
    unk_id = w2idx['unk']
    pad_id = w2idx['_'] 
    start_id = w2idx['start_id']
    end_id = w2idx['end_id']

    target_seqs = tl.prepro.sequences_add_end_id([trainY[10]], end_id=end_id)[0]
    decode_seqs = tl.prepro.sequences_add_start_id([trainY[10]], start_id=start_id, remove_last=False)[0]
    target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]

    print("encode_seqs", [idx2w[id] for id in trainX[10]])
    print("target_seqs", [idx2w[id] for id in target_seqs])
    print("decode_seqs", [idx2w[id] for id in decode_seqs])
    print("target_mask", target_mask)
    print(len(target_seqs), len(decode_seqs), len(target_mask))

    # Initialize seq2seq_word for training
    encode_seqs = tf.placeholder(dtype=tf.int64, shape=[options.batch_size, None], name="encode_seqs")
    decode_seqs = tf.placeholder(dtype=tf.int64, shape=[options.batch_size, None], name="decode_seqs")
    target_seqs = tf.placeholder(dtype=tf.int64, shape=[options.batch_size, None], name="target_seqs")
    target_mask = tf.placeholder(dtype=tf.int64, shape=[options.batch_size, None], name="target_mask")

    net_out, _ = build_seq2seq_word(encode_seqs, decode_seqs, xvocab_size, 
            options.batch_size, options.embedding_dim, options.nlayers, options.dropout, is_train=True, reuse=False)

    # Initialize seq2seq_word for inference
    encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
    net, net_rnn = build_seq2seq_word(encode_seqs2, decode_seqs2, xvocab_size, 
            options.batch_size, options.embedding_dim, options.nlayers, options.dropout, is_train=False, reuse=True)
    y = tf.nn.softmax(net.outputs)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name='lstm.npz', network=net)

    custom_contexts = [
            'happy birthday have a nice day',
            'donald trump won last nights presidential debate according to snap online polls',
            'how are you',
            'do you love me',
            'game of thrones is the best show ever',
            'republicans are really smart',
            'you are just jealous',
            'is it ok to hit kids',
            'are you conscious'
            ]

    it = 0
    while True:
        if it < len(custom_contexts):
            seed = custom_contexts[it]
            it += 1
        else:
            print('Talk to Chatbot')
            seed= input()
            seed = seed.strip()

            if seed == "":
                break

        print("Query >", seed, flush=True)
        seed_id = [w2idx.get(w, w2idx['unk']) for w in seed.split(" ")]

        sentences = []
        total_iter = 0
        while True:
            state = sess.run(net_rnn.final_state_encode,
                            {encode_seqs2: [seed_id]})

            o, state = sess.run([y, net_rnn.final_state_decode],
                            {net_rnn.initial_state_decode: state,
                            decode_seqs2: [[start_id]]})

            w_id = tl.nlp.sample_top(o[0], top_k=3)
            w = idx2w[w_id]

            sentence = [w]
            for _ in range(20): # max sentence length
                o, state = sess.run([y, net_rnn.final_state_decode],
                                {net_rnn.initial_state_decode: state,
                                decode_seqs2: [[w_id]]})
                w_id = tl.nlp.sample_top(o[0], top_k=2)
                w = idx2w[w_id]
                if w_id == end_id:
                    break
                sentence = sentence + [w]

            if sentence not in sentences:
                sentences.append(sentence)

            if total_iter > 1000 or len(sentences) > 50:
                break

            total_iter += 1

        ranked_sentences = []
        for s in sentences:
            encoded_q = [de_vocab.get(x, de_vocab['unk']) for x in seed]
            encoded_s = [de_vocab.get(x, de_vocab['unk']) for x in s]
            padded_encoded_q = pad(encoded_q, 20)
            padded_encoded_s = pad(encoded_s, 20)

            score = model.predict([padded_encoded_q, padded_encoded_s])[0][0]
            ranked_sentences.append((s, score))

        ranked_sentences = sorted(ranked_sentences, key = lambda x: x[1], reverse=True)
        for s, score in ranked_sentences:
            print('----- {} -> {}'.format(' '.join(s), score))

if __name__ == '__main__':
    global options, args
    (options, args) = parser.parse_args()

    sys.exit(main())
