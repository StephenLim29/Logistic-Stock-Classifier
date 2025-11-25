import torch
import torch.nn as nn
import torch.nn.functional as F
from mha import MultiHeadAttention as MHA
import sys

class Transformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.inputprojection = nn.Linear(6, d_model)

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
        self.classifier = nn.Linear(d_model, 2)
       
    
    def forward(self, x, attn_mask):
        
        # Check BEFORE projection
        if torch.isnan(x).any():
            print("NaNs in raw inputs BEFORE projection")
            print("input NaN locations:", torch.isnan(x).nonzero()[:10])
            sys.exit(1)

        x = self.inputprojection(x)  # (B, T, d_model)

        # Check AFTER projection
        if torch.isnan(x).any():
            print("NaNs AFTER inputprojection")
            print("weight NaNs:", torch.isnan(self.inputprojection.weight).any().item())
            print("bias NaNs:", torch.isnan(self.inputprojection.bias).any().item())
            sys.exit(1)
        h = self.mha(self.layernorm1(x), self.layernorm1(x), self.layernorm1(x), attn_mask)
        
        x = x + self.dropout(h)
        
        h = self.ffn(self.layernorm2(x))
        x = x + self.dropout(h)
        pooled = x[:, -1, :]           
        logits = self.classifier(pooled) 
        return logits