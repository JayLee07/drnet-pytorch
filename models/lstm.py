import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):

    def __init__(self, args):
        super(lstm, self).__init__()
        self.args = args
        self.embed = nn.Linear(args.lstm_input_size, args.lstm_hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(args.hidden_size, args.hidden_size) for i in range(args.n_layers)])
        self.output = nn.Sequential(nn.Linear(args.hidden_size, args.output_size),
                                    nn.BatchNorm1d(args.output_size),
                                    nn.Tanh())
        self.hidden = self.init_hidden()
        self.normalize = False

    def init_hidden(self):
        hidden = []
        for i in range(self.args.n_layers):
            if len(self.args.n_gpu == 0):
                hidden.append(Variable(torch.zeros(self.args.batch_size, self.args.lstm_hidden_size)),
                              Variable(torch.zeros(self.args.batch_size, self.args.lstm_hidden_size)))
            else:
                hidden.append(Variable(torch.zeros(self.args.batch_size, self.args.lstm_hidden_size).cuda()),
                              Variable(torch.zeros(self.args.batch_size, self.args.lstm_hidden_size).cuda()))

    def forward(self, x):
        embedded = x.view(-1, self.args.hidden_size)
        hidden_in = embedded
        for i in range(self.args.n_layers):
            self.hidden[i] = self.lstm[i](hidden_in, self.hidden[i])
            hidden_in = self.hidden[i][0]
        output = self.output(hidden_in)
        if self.normalize:
            return nn.functional.normalize(output, p=2)
        else:
            return output