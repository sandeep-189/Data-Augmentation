## Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
## Description : Python file containing DeepConvLSTM model for classification purposes

import torch
import torch.nn as nn
from BasicConvs import BasicConv2d

class DeepConvNet(nn.Module):
    
    def __init__(self, in_channels = 3, input_size = 100, hidden_size = 128, output_size = 7, conv_filter = (5,9), conv_padding = (2,4)):
        super(DeepConvNet,self).__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        ## Input : 3 * 9 * 100 for PAMAP2
        self.conv = nn.Sequential(BasicConv2d(in_channels,64,conv_filter,padding = conv_padding),
                                  ## 64 * 9 * 100
                                 BasicConv2d(64,256,conv_filter,padding = conv_padding),
                                  ## 256 * 9 * 100
                                 BasicConv2d(256,512,conv_filter,padding = conv_padding),
                                  ## 512 * 9 * 100
                                 nn.AdaptiveAvgPool2d((10,10))
                                  ## 512 * 10 * 10
                                 )
        ## Input to LSTM : 512 * 100
        self.lstm = nn.LSTM(input_size = 100, hidden_size = hidden_size, num_layers = 2, batch_first = True)
        ## 512 * 128(hidden_size)
        self.fcn = nn.Sequential(nn.Flatten(),nn.ReLU(),nn.Dropout(p=0.2),
                                 nn.Linear(512*hidden_size, output_size))
        ## 7(output_size)
        
    def forward(self, input_seq):
        input_seq = input_seq.view(input_seq.shape[0],self.in_channels,-1,self.input_size)
        y = self.conv(input_seq)
        y, h = self.lstm(y.view(y.shape[0],-1,100))
        y = self.fcn(y)
        return y