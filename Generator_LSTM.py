import torch
import torch.nn as nn
import numpy as np
from BasicConvs import BasicConv1d
import helper

class Generator(nn.Module):
    
    def __init__(self, hidden_size = 256, noise_len = 100, output_size = (27,100), num_layers = 2, bidirectional = False):
        super(Generator,self).__init__()
        self.output_size = output_size
        self.noise_len = noise_len
        flat_output_size = np.prod(output_size)
        norm_layer = 0 # 0: no batchnorm 1: add batchnorm after conv
        ## Input : 100 noise_len
        self.prelstm = nn.Sequential(BasicConv1d(1, 256, 1, norm_layer = norm_layer),
                                    )
        ## 256 * 100(noise_len) 
        self.lstm = nn.Sequential(nn.LSTM(input_size = noise_len, hidden_size = hidden_size,
                                          num_layers = num_layers, batch_first = True, bidirectional = bidirectional),
                                  )
        ## 1(bidirectional) * 256 * 256(hidden_size)
        self.postlstm = nn.Sequential(BasicConv1d(256, 128, 3, stride = 1, padding = 1, norm_layer = norm_layer),
                                     BasicConv1d(128, 64, 3, stride = 1, padding = 1, norm_layer = norm_layer),
                                     )
        element = helper.conv1d_ele_size(helper.conv1d_ele_size(hidden_size, 3, 1, 1, 1), 3, 1, 1, 1)
        ## 1(bidirectional) * 64 * element
        self.fcn = nn.Sequential(nn.PReLU(), nn.Dropout(0.5),
                                 nn.Conv1d(64*element, flat_output_size, 1),
                                )
        ## Output : 27 * 100
        
        
    def forward(self, x):
        batch_size = x.shape[0]
        y = self.prelstm(x.view(batch_size,1,-1))
        y, _ = self.lstm(y.view(batch_size,-1,self.noise_len))
        y = self.postlstm(y)
        y = y.reshape(batch_size, -1, 1)
        y = self.fcn(y)
        y = y.view([batch_size]+list(self.output_size)) 
        return y