import torch
import torch.nn as nn

## Rewritten TransformerEncodeLayer from Pytorch library to use convolutions instead of linear layers

class EncodeLayer(nn.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(EncodeLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Conv1d(d_model, dim_feedforward,1)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Conv1d(dim_feedforward, d_model,1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        
    def forward(self, src):
        src2 = self.self_attn(src, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src.view(src.shape[0],self.d_model,-1)))).view(src.shape[0],self.dim_feedforward,-1))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
