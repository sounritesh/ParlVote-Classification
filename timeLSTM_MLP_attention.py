from torch import nn
import torch

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, timestamps, reverse=False):
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)

        h = h.cuda()
        c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs

class TimeLSTM_ATT_MLP(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, lstm_hidden_size=768, ntargets=1, input_size=768, sequence_length=6):
        super(TimeLSTM_ATT_MLP, self).__init__()

        self.hidden_size = lstm_hidden_size
     
        self.lstm = TimeLSTM(input_size=input_size, hidden_size=lstm_hidden_size)
        self.attention = Attention(feature_dim=lstm_hidden_size, step_dim=sequence_length)

        layers = []
        layers.append(nn.Linear(lstm_hidden_size, hidden_size))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        for i in range(num_layers):
          layers.append(nn.Linear(hidden_size, hidden_size))
          layers.append(nn.Dropout(dropout))
          layers.append(nn.ReLU())
       
        layers.append(nn.Linear(hidden_size, ntargets))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, timestamps):
        out = self.lstm(x, timestamps)
        out = self.attention(out)
        out = self.mlp(out)
        return out