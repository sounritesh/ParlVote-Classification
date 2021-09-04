from torch import nn 

class MLP(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, ntargets=1, input_size=768):
        super(MLP, self).__init__()
     
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        for i in range(num_layers):
          layers.append(nn.Linear(hidden_size, hidden_size))
          layers.append(nn.Dropout(dropout))
          layers.append(nn.ReLU())
       
        layers.append(nn.Linear(hidden_size, ntargets))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out