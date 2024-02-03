import torch
import torch.nn as nn

from structure.SelfAttention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm_layer = nn.LayerNorm(embed_size)
        self.norm_block = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        X = self.dropout(self.norm_layer(attention + query))
        forward = self.feed_forward(X)

        output = self.dropout(self.norm_block(forward + X))
        return output