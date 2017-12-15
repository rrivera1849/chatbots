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

## Authors

* **Rafael A. Rivera-Soto**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
