
import torch
import torch.nn as nn
from torch.autograd import Variable

class DualEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.5):
        super(DualEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, dropout=dropout, batch_first=True)
        self.cell_state = Variable(torch.zeros(1, 1, self.hidden_size))

        self.M = Variable(torch.randn(self.hidden_size, self.hidden_size), requires_grad=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, data, is_train, use_cuda=False):
        if is_train:
            context = Variable(data['context'], requires_grad=False)
            utterance = Variable(data['utterance'], requires_grad=False)

            if use_cuda and torch.cuda.is_available():
                context.cuda()
                utterance.cuda()

            context_embedded = self.embedding(context)
            utterance_embedded = self.embedding(utterance)

            if use_cuda and torch.cuda.is_available():
                context_embedded.cuda()
                utterance_embedded.cuda()

            context_encoded = self.encode_input_(context_embedded, use_cuda).squeeze(0)
            utterance_encoded = self.encode_input_(utterance_embedded, use_cuda).squeeze(0).transpose(0, 1)
            predicted_utterance = torch.matmul(context_encoded, self.M)
            score = self.sigmoid(torch.matmul(predicted_utterance, utterance_encoded))

            return score
        else:
            scores = []
            context = Variable(data['context'], requires_grad=False, volatile=True)
            
            if use_cuda and torch.cuda.is_available():
                context.cuda()

            for key, value in data.items():
                if key == 'context': 
                    continue

                value = Variable(value, requires_grad=False, volatile=True)
                if use_cuda and torch.cuda.is_available():
                    value.cuda()

                context_embedded = self.embedding(context)
                val_embedded = self.embedding(value)

                if use_cuda and torch.cuda.is_available():
                    context_embedded.cuda()
                    val_embedded.cuda()

                context_encoded = self.encode_input_(context_embedded, use_cuda, False, True).squeeze(0)
                val_encoded = self.encode_input_(val_embedded, use_cuda, False, True).squeeze(0).transpose(0, 1)

                predicted_val = torch.matmul(context_encoded, self.M)
                score = self.sigmoid(torch.matmul(predicted_val, val_encoded))
                scores.append(score)

            return scores

    def encode_input_(self, x, use_cuda, requires_grad=True, volatile=False):
        hidden_state = Variable(torch.zeros(1, 1, self.hidden_size), \
                requires_grad=requires_grad, volatile=volatile)
        if use_cuda and torch.cuda.is_available():
            hidden_state.cuda()

        _, (x_encoded, _) = self.lstm(x , (hidden_state, self.cell_state))

        return x_encoded

