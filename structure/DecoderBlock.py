import torch
import torch.nn as nn

from structure.SelfAttention import SelfAttention
from structure.TransformerBlock import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.normalization_layer = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, value, key, source_mask, target_mask):
        attention = self.attention(X, X, X, target_mask)
        query = self.dropout(self.normalization_layer(attention + X))
        output = self.transformer_block(value, key, query, source_mask)

        return output