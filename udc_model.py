
import sys
import tensorflow as tf

import udc_metrics

def get_id_feature(features, key, len_key, max_len):
    ids = features[key]
    ids_len = tf.squeeze(features[len_key], [1])
    ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
    return ids, ids_len

def create_train_op(loss, hparams):
    train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=hparams.learning_rate,
            clip_gradients=10.0,
            optimizer=hparams.optimizer)
    return train_op

def create_model_fn(hparams, model_impl):

    def model_fn(features, labels, mode):
        context, context_len = get_id_feature(
                features, 'context', 'context_len', hparams.max_context_len)

        utterance, utterance_len = get_id_feature(
                features, 'utterance', 'utterance_len', hparams.max_utterance_len)

        batch_size = labels.get_shape().as_list()[0]

        if mode == tf.estimator.ModeKeys.TRAIN:
            probs, loss = model_impl(
                    hparams,
                    mode,
                    context,
                    context_len,
                    utterance,
                    utterance_len,
                    labels)
            train_op = create_train_op(loss, hparams)

            # return probs, loss, train_op
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op)

        if mode == tf.estimator.ModeKeys.PREDICT:
            probs, loss = model_impl(
                    hparams,
                    mode,
                    context,
                    context_len,
                    utterance,
                    utterance_len,
                    None)

            # return probs, 0.0, None

            return tf.estimator.EstimatorSpec(mode=mode)

        if mode == tf.estimator.ModeKeys.EVAL:
            all_contexts = [context]
            all_context_lens = [context_len]
            all_utterances = [utterance]
            all_utterance_lens = [utterance_len]
            all_labels = [tf.ones([batch_size, 1], dtype=tf.int64)]

            for i in range(9):
                distractor, distractor_len = get_id_feature(features,
                    'distractor_{}'.format(i),
                    'distractor_{}_len'.format(i),
                    hparams.max_utterance_len)
                all_contexts.append(context)
                all_context_lens.append(context_len)
                all_utterances.append(distractor)
                all_utterance_lens.append(distractor_len)
                all_labels.append(tf.zeros([batch_size, 1], dtype=tf.int64))

            probs, loss = model_impl(
                hparams,
                mode,
                tf.concat(all_contexts, 0),
                tf.concat(all_context_lens, 0),
                tf.concat(all_utterances, 0),
                tf.concat(all_utterance_lens, 0),
                tf.concat(all_labels, 0))

            split_probs = tf.split(probs, 10, axis=0)
            shaped_probs = tf.concat(split_probs, 1)

            tf.summary.histogram('eval_correct_probs_hist', split_probs[0])
            tf.summary.histogram('eval_correct_probs_average', tf.reduce_mean(split_probs[0]))
            tf.summary.histogram('eval_incorrect_probs_hist', split_probs[1])
            tf.summary.histogram('eval_incorrect_probs_average', tf.reduce_mean(split_probs[1]))

            print('shaped_probs: {}'.format(shaped_probs.get_shape()))

            eval_metrics = {}
            for k in [1,2,5,10]:
                eval_metrics['recall_at_{}'.format(k)] = \
                        tf.contrib.metrics.streaming_sparse_recall_at_k(
                            shaped_probs, 
                            tf.zeros([batch_size, 1], dtype=tf.int64), k=k
                            )

            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=None,
                    eval_metric_ops=eval_metrics)

    return model_fn
