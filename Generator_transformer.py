import torch
import torch.nn as nn
import numpy as np
import helper

class Generator(nn.Module):
    
    def __init__(self, noise_len = 100, output_size = (27,100), nheads = 3, period = 50, dim_feedforward = 2048):
        super(Generator,self).__init__()
        self.output_size = output_size
        self.noise_len = noise_len
        flat_output_size = np.prod(output_size)
        
        self.embedding = nn.Conv1d(noise_len, noise_len, 1)
        
        self.positional_embedding = helper.generate_pe(1, noise_len, period = period)
        
        self.layer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model = noise_len, nhead = nheads, dim_feedforward = dim_feedforward, dropout = 0.5, activation = "relu"),
            nn.Conv1d(1, 10, 1),
            nn.TransformerEncoderLayer(d_model = noise_len, nhead = nheads, dim_feedforward = dim_feedforward, dropout = 0.5, activation = "relu"),
            nn.Conv1d(10, 1, 1),
        )
        
        self.fcn = nn.Sequential(nn.PReLU(), nn.Dropout(0.5),
                                nn.Conv1d(noise_len, flat_output_size, 1)
                                )
        
        
    def forward(self, x):
        y = self.embedding(x.view(x.shape[0], -1, 1))
        y = y.view(x.shape[0], 1, self.noise_len) + self.positional_embedding.to(y.device) # broadcasting on batch dimension
        y = self.layer(y.view(y.shape[0], 1, self.noise_len))
        y = self.fcn(y.view(x.shape[0], -1, 1))
        y = y.view([x.shape[0]]+list(self.output_size))
        return y