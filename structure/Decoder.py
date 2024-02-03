import torch
import torch.nn as nn

from structure.SelfAttention import SelfAttention
from structure.TransformerBlock import TransformerBlock
from structure.DecoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_len):
        super(Decoder, self).__init__()

        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
        ])

        self.fully_connected_output = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, encoder_output, source_mask, target_mask):
        N, seq_len = X.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        X = self.dropout(self.word_embedding(X) + self.positional_embedding(positions))

        for layer in self.layers:
            X = layer(X, encoder_output, encoder_output, source_mask, target_mask)
        
        output = self.fully_connected_output(X)
        return output