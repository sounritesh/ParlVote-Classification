import torch
from torch import nn 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM_MLP(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, lstm_layers, lstm_hidden_size, ntargets=1, input_size=768):
        super(LSTM_MLP, self).__init__()

        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_layers
     
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, dropout=0)

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0,h0))
        out = out[:,-1,:]
        out = self.mlp(out)
        return out