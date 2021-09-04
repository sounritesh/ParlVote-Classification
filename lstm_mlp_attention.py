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

class LSTM_MLP_ATT(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, lstm_layers, lstm_hidden_size, ntargets=1, input_size=768, sequence_length=6):
        super(LSTM_MLP_ATT, self).__init__()
        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
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


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = self.mlp(out)
        return out