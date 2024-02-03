import torch
import torch.nn as nn

from structure.SelfAttention import SelfAttention
from structure.TransformerBlock import TransformerBlock

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device
        
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X, mask):
        N, seq_len = X.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        output = self.dropout(self.word_embedding(X) + self.positional_embedding(positions))
        
        for layer in self.layers:
            output = layer(output, output, output, mask)
        
        return output