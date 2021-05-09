## Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
## Description : Python file containing transformer based classifier for timeseries data

import torch
import torch.nn as nn
from EncodeLayer import EncodeLayer
import dataset

class TransformerClassifier(nn.Module):
    
    def __init__(self, in_channels, output_size, d_model, nhead, period = 100, dim_feedforward=2048, dropout=0.3, num_layers = 3):
        super(TransformerClassifier, self).__init__()
        self.positional_embedding = dataset.generate_pe(in_channels, d_model, period = period)
        self.layers = nn.Sequential(
            *[EncodeLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)],
        )
        
        self.fcn =  nn.Sequential(nn.Flatten(),nn.ReLU(),nn.Dropout(p=0.2),
                                 nn.Linear(in_channels*d_model, output_size))
        
    def forward(self, x):
        y = self.layers(x + self.positional_embedding.to(x.device))
        y = self.fcn(y)
        return y