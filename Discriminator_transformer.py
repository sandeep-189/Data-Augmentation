## Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
## Description : Python file containing Self attention based discriminator(transformer discriminator) of GAN network

import torch
import torch.nn as nn
import numpy as np
import helper
from EncodeLayer import EncodeLayer

class Discriminator(nn.Module):
    
    def __init__(self, input_size = (27,100), nheads = 3, period = 50, dim_feedforward = 2048, num_layers = 1):
        super(Discriminator,self).__init__()
        self.input_size = input_size
        self.flat_input_size = np.prod(input_size)
        
        # Produce a learned embedding for timeseries (for word data this is a word bag embedding)
        self.embedding = nn.Conv1d(self.flat_input_size, self.flat_input_size, 1)
        
        # constant which is added to every timeseries data
        self.positional_embedding = helper.generate_pe(input_size[-2], input_size[-1], period = period) 
        
        self.layer = nn.Sequential(
            *[EncodeLayer(d_model = self.input_size[-1], nhead = nheads, dim_feedforward = dim_feedforward, dropout = 0.4) for _ in range(num_layers)],
        )
        
        self.fcn = nn.Sequential(nn.PReLU(), nn.Dropout(0.4),
                                 nn.Conv1d(self.flat_input_size, 1, 1), nn.Sigmoid(),
                                )
        
    def forward(self, x):
        batch_size = x.shape[0]
        y = self.embedding(x.view(batch_size, -1, 1))
        y = y.view(batch_size, self.input_size[-2], self.input_size[-1]) + self.positional_embedding.to(y.device) # broadcasting on batch dimension
        y = self.layer(y.view(batch_size, self.input_size[-2], self.input_size[-1]))
        y = self.fcn(y.view(batch_size, -1, 1))
        return y