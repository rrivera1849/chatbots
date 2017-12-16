# Building Conversational Agents

Developing conversational agents, also called chatbots, is a new hot topic within the artificial intelligence community. 
The purpose of these agents is to be able to learn how to carry out a conversation with a human user. These agents have the 
potential of changing how customers interact with companies and open up new business opportunities where the primary mode 
of customer interaction is through textual conversation. In this work, we explore retrieval-based models on the Ubuntu Dialog Corpus 
and Sequence to Sequence models on a publicly available dataset. Finally, we show how both paradigms may be combined to produce 
more robust conversational agents.

## Getting Started

Or code has the following dependencies:
* pandas
* tqdm
* numpy
* keras
* tensorflow v1.3
* tensorlayer
* sklearn

We recommend that you install a [Miniconda](https://conda.io/miniconda.html) to run the code. 

Once you've created an environment, run the following commands to get a copy of the code and install all dependencies:

```
git clone https://github.com/rrivera1849/chatbots.git
pip install -r requirements.txt
```

## What's in the repository?

This section is a breakdown of each file included in the repository and their functions:

* preprocess/udc.py -- Used to preprocess data from the [Ubuntu Dialog Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
* preprocess/twitter_char.py -- Used to preprocess data from the [Twitter Corpus](https://github.com/Marsan-Ma/chat_corpus)
* data/twitter/data.py -- Utility used to load twitter dataset easily
* data/twitter/data_retrieval.py -- Utility used to create and load twitter retrieval dataset easily
* dual_encoder.py -- Implements model described in this [paper](https://arxiv.org/abs/1510.03753)
* seq2seq_char.py -- Implements a Sequence to Sequence model that operates on characters, see [this](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py) for an example
* seq2seq_char_chatbot.py -- Takes a seq2seq character model and lets the user interact with it
* seq2seq_word.py -- Implements a Sequence to Sequence model that operates on words, see [this](https://github.com/zsdonghao/seq2seq-chatbot/blob/master/main_simple_seq2seq.py) for an example
* seq2seq_word_chatbot.py -- Takes a seq2seq word model and lets the user interact with it

In general, this repository contains implementations of various conversational agent models. Some of the models included are:
* Dual Encoder model described in this [paper](https://arxiv.org/abs/1510.03753) for retrieval-based chatbots in the Ubuntu Dialog Corpus
* [Sequence to Sequence](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) models that operate on both characters 
  and words
* It also allows you to interact with both of these chatbots and includes a dual paradigm chatbot that first uses a generative Sequence to Sequence
  model to generate K responses and then scores each one with the Dual Encoder model. In this way we leverage the dynamic nature of generative responses 
  while taking advantage of retrieval based models.

## Training Models

To train any model on the gpu you must first set an environment variable to indicate which GPU to use:

```
export CUDA_VISIBLE_DEVICES=<gpu-id>
```

### Dual Encoder

To train this model on the Ubuntu Dialog Corpus you must first [download](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
the dataset and save it to data/udc. Then, use the preprocess/udc.py script to preprocess it:

```
python udc.py --dataset-path ./data/udc --output-path ./data/udc
```

This will save the preprocessed dataset to data/udc. You can then run the model as follows:

```
python dual_encoder.py
```

If you want the train on the twitter dataset then simply do the following:

```
python dual_encoder.py --twitter --max-context-length 20 --max-utterance-length 20
```

You can also tune the following parameters:

    Options:
    -h, --help            show this help message and exit
    --num-epochs=NUM_EPOCHS
                            Number of epochs to use during training
    --batch-size=BATCH_SIZE
                            Number of batches to use during training and
                            evaluation
    --validate-every=VALIDATE_EVERY
                            Validate model every |num| iterations
    --twitter
    --embedding-dim=EMBEDDING_DIM
                            Dimensionality of the word embedding vectors
    --rnn-dim=RNN_DIM     Dimensionality of the LSTM hidden/cell vector
    --max-context-length=MAX_CONTEXT_LENGTH
                            Maximum length for each context
    --max-utterance-length=MAX_UTTERANCE_LENGTH
                            Maximum length for each utterance/distractor

### Seq2Seq Character


The data to train this model is already preprocessed and is located at data/twitter/twitter_char_100000.pkl. Run the following command
to train a model:

```
python seq2seq_char.py
```

The output will be in checkpoints/seq2seq_char. You can also tune the following parameters:

    Options:
    -h, --help            show this help message and exit
    --dataset-path=DATASET_PATH
    --experiment-id=EXPERIMENT_ID
    --rnn-dim=RNN_DIM
    --batch-size=BATCH_SIZE
    --num-epochs=NUM_EPOCHS

### Seq2Seq Word

The data to train this model is already preprocessed and is located at data/twitter/idx_a.npy and data/twitter/idx_q.npy. Run the following 
command to train a model:

```
python seq2seq_word.py
```

The output will be saved in your current working directory. You can also tune the following parameters:

    Options:
    -h, --help            show this help message and exit
    --data-path=DATA_PATH
    --checkpoint-path=CHECKPOINT_PATH
    --print-every=PRINT_EVERY
    --eval-every=EVAL_EVERY
    --batch-size=BATCH_SIZE
    --num-epochs=NUM_EPOCHS
    --embedding-dim=EMBEDDING_DIM
    --dropout=DROPOUT
    --nlayers=NLAYERS
    --lr=LR

## Running Chatbots

The same rules for running in the GPU apply, you must set the environment variable as discussed above.

### Seq2Seq Character 

To talk to a Seq2Seq Character model run the following command:

```
python seq2seq_char_chatbot.py --checkpoint-path path/to/checkpoint
```

### Seq2Seq Word

To talk to a Seq2Seq Word model run the following command:

```
python seq2seq_word.py --checkpoint-path path/to/model
```

### Dual Paradigm Models

To talk to a Dual Paradigm model you must first train the Dual Encoder model on the twitter dataset and the Seq2Seq Word model. Once you 
have checkpoints for both you may run the following command:

```
python seq2seq_word.py --checkpoint-path path/to/seq2seq_model --dual-encoder-path /path/to/dual_encoder_model
```

## Authors

* **Rafael A. Rivera-Soto** (rivera43@stanford.edu)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
