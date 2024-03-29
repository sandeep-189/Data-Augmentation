## Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
## Description : Python file containing LSTM based Discriminator of GAN network

import torch
import torch.nn as nn
from BasicConvs import BasicConv1d
import helper

class Discriminator(nn.Module):
    
    def __init__(self, hidden_size = 256, input_size = (27,100), num_layers = 2, bidirectional = False):
        super(Discriminator,self).__init__()
        self.input_size = input_size
        norm_layer = 0 # # 0: no batchnorm 1: add batchnorm after conv
        ## Input : 27 * 100 for PAMAP2
        self.prelstm = nn.Sequential(BasicConv1d(input_size[0], 500, 3, padding = 1, norm_layer = norm_layer),
                                    )
        ## 500 * 100 
        self.lstm = nn.Sequential(nn.LSTM(input_size = input_size[-1], hidden_size = hidden_size, 
                                          num_layers = num_layers, batch_first = True, bidirectional = bidirectional),
                                  )
        ## 1(bidirectional) * 500 * 256(hidden_size)
        self.postlstm = nn.Sequential(BasicConv1d(500, 256, 3, stride = 1, padding = 1, norm_layer = norm_layer),
                                      BasicConv1d(256, 128, 3, stride = 1, padding = 1, norm_layer = norm_layer),
                                    )
        element = helper.conv1d_ele_size(helper.conv1d_ele_size((1 + int(bidirectional))*hidden_size, 3, 1, 1, 1), 3, 1, 1, 1)
        ## 1(bidirectional) * 128 * element
        self.fcn = nn.Sequential(nn.PReLU(), nn.Dropout(0.3),
                                 nn.Conv1d(128*element, 1, 1), nn.Sigmoid(),
                                ) 
        ## Output : 1
        
    def forward(self, x):
        batch_size = x.shape[0]
        y = self.prelstm(x)
        y, _ = self.lstm(y.view(batch_size,-1,self.input_size[-1]))
        y = self.postlstm(y)
        y = y.reshape(batch_size, -1, 1)
        y = self.fcn(y)
        y = y.view(batch_size,1)
        return y