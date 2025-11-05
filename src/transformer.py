import torch
import torch.nn as nn
import torch.nn.functional as F
from mha import MultiHeadAttention as MHA

class Transformer(nn.module):
    def __init__(self, d_model):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.mha = MHA(d_model, d_model, d_model, d_model, 16, True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*d_model, d_model)
        ) 
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        h = self.mha(self.layernorm1(x), self.layernorm1(x), self.layernorm1(x), None)
        x = x + self.dropout(h)
        h = self.ffn(self.layernorm2(x))
        x = x + self.dropout(h)
        return x