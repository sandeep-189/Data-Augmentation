## Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
## Description : Python file containing module class for convolution, batchnorm(optional) and relu together for 1d and 2d

import torch
import torch.nn as nn

class BasicConv1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride = 1, padding = 0, bias = False, norm_layer = 1):
        super(BasicConv1d, self).__init__() 
            
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size, stride = stride, padding = padding, bias = bias)
        if norm_layer == 1:
            self.bn = nn.BatchNorm1d(out_planes,
                                     eps=0.001,
                                     momentum=0.1,
                                     affine=True)
        elif norm_layer == 2:
            self.bn = nn.InstanceNorm1d(out_planes,
                                     eps=0.001,
                                     momentum=0.1,
                                     affine=True)
        else:
            self.bn = nn.Identity()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
def make_list(val):
        if type(val) is int:
            return (val, val)
        elif type(val) is tuple and len(val) == 2:
            return val
        else:
            raise TypeError("Only int and tuples allowed")

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride = 1, padding = 0):
        super(BasicConv2d, self).__init__()
        
        side_pad, top_pad = make_list(padding)
        side_ker, top_ker = make_list(kernel_size)
        side_stride, top_stride = make_list(stride)
                
            
        self.conv = nn.Sequential(nn.Conv2d(in_planes, in_planes,
                                            kernel_size=1, bias=True),
                                  nn.Conv2d(in_planes, in_planes,
                                            kernel_size=(side_ker,1), stride=(side_stride,1),
                                            padding=(side_pad,0), bias=False),
                                  nn.Conv2d(in_planes, in_planes,
                                            kernel_size=(1,top_ker), stride=(1,top_stride),
                                            padding=(0,top_pad), bias=False),
                                  nn.Conv2d(in_planes, out_planes,
                                            kernel_size=1, bias=True),
                                 )
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,
                                 momentum=0.1,
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x