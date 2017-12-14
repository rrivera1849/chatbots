
import os
import pdb
import sys
import time
from optparse import OptionParser

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from data.twitter import data
from sklearn.utils import shuffle

parser = OptionParser()
parser.add_option('--data-path', type=str, default='./data/twitter/')
parser.add_option('--checkpoint-path', type=str, default='./lstm.npz')
parser.add_option('--print-every', type=int, default=200)
parser.add_option('--eval-every', type=int, default=1000)

parser.add_option('--batch-size', type=int, default=32)
parser.add_option('--num-epochs', type=int, default=50)
parser.add_option('--embedding-dim', type=int, default=1024)
parser.add_option('--dropout', type=float, default=0.5)
parser.add_option('--nlayers', type=int, default=3)
parser.add_option('--lr', type=float, default=0.0001)

def load_data(path):
    metadata, idx_q, idx_a = data.load_data(path)
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

    trainX = trainX.tolist()
    trainY = trainY.tolist()
    testX = testX.tolist()
    testY = testY.tolist()
    validX = validX.tolist()
    validY = validY.tolist()

    trainX = tl.prepro.remove_pad_sequences(trainX)
    trainY = tl.prepro.remove_pad_sequences(trainY)
    testX = tl.prepro.remove_pad_sequences(testX)
    testY = tl.prepro.remove_pad_sequences(testY)
    validX = tl.prepro.remove_pad_sequences(validX)
    validY = tl.prepro.remove_pad_sequences(validY)

    return trainX, trainY, testX, testY, validX, validY, metadata

def prepare_vocab(w2idx, idx2w):
    xvocab_size = len(idx2w)
    start_id = xvocab_size
    end_id = xvocab_size + 1
    w2idx.update({'start_id': start_id})
    w2idx.update({'end_id': end_id})
    idx2w = idx2w + ['start_id', 'end_id']

    xvocab_size = xvocab_size + 2

    return w2idx, idx2w, xvocab_size

def build_seq2seq_word(encode_seqs, decode_seqs, xvocab_size, 
                       batch_size, embedding_dim, nlayers, dropout, is_train=True, reuse=False):

    with tf.variable_scope("model", reuse=reuse):
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = embedding_dim,
                name = 'seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = embedding_dim,
                name = 'seq_embedding')

        net_rnn = Seq2Seq(net_encode, net_decode,
                cell_fn = tf.contrib.rnn.BasicLSTMCell,
                n_hidden = embedding_dim,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                dropout = (dropout if is_train else None),
                n_layer = nlayers,
                return_seq_2d = True,
                name = 'seq2seq')
        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')

    return net_out, net_rnn

def main():
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

    loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, 
            input_mask=target_mask, return_details=False, name='cost')
    net_out.print_params(False)

    train_op = tf.train.AdamOptimizer(learning_rate=options.lr).minimize(loss)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=options.checkpoint_path, network=net)

    for epoch in range(options.num_epochs):
        epoch_time = time.time()
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        total_err, n_iter = 0, 0

        for X, Y in tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=options.batch_size, shuffle=False):
            step_time = time.time()

            X = tl.prepro.pad_sequences(X)
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)

            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            _, err = sess.run([train_op, loss],
                            {encode_seqs: X,
                            decode_seqs: _decode_seqs,
                            target_seqs: _target_seqs,
                            target_mask: _target_mask})

            if n_iter % options.print_every == 0:
                print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % 
                        (epoch, options.num_epochs, n_iter, n_step, err, time.time() - step_time), flush=True)

            total_err += err; n_iter += 1

            if n_iter % options.eval_every == 0:
                seeds = ["happy birthday have a nice day",
                        "donald trump won last nights presidential debate according to snap online polls",
                        "how are you",
                        "do you love me",
                        "game of thrones is the best show ever",
                        "republicans are really smart",
                        "you are just jealous",
                        "is it ok to hit kids",
                        "are you conscious"]
                for seed in seeds:
                    print("Query >", seed, flush=True)
                    seed_id = [w2idx[w] for w in seed.split(" ")]
                    for _ in range(5):  # 1 Query --> 5 Reply
                        # 1. Encode the input sentence
                        state = sess.run(net_rnn.final_state_encode,
                                        {encode_seqs2: [seed_id]})

                        # 2. Feed start_id and get first word
                        o, state = sess.run([y, net_rnn.final_state_decode],
                                        {net_rnn.initial_state_decode: state,
                                        decode_seqs2: [[start_id]]})
                        w_id = tl.nlp.sample_top(o[0], top_k=3)
                        w = idx2w[w_id]
                        # 3. Decode until end_id or max_size
                        sentence = [w]
                        for _ in range(30): # max sentence length
                            o, state = sess.run([y, net_rnn.final_state_decode],
                                            {net_rnn.initial_state_decode: state,
                                            decode_seqs2: [[w_id]]})
                            w_id = tl.nlp.sample_top(o[0], top_k=2)
                            w = idx2w[w_id]
                            if w_id == end_id:
                                break
                            sentence = sentence + [w]
                        print(" >", ' '.join(sentence), flush=True)

        print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % 
                (epoch, options.num_epochs, total_err/n_iter, time.time()-epoch_time), flush=True)

        tl.files.save_npz(net.all_params, name='lstm.npz', sess=sess)

    return 0

if __name__ == '__main__':
    global options, args
    (options, args) = parser.parse_args()

    sys.exit(main())

