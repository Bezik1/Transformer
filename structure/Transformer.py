import torch
import torch.nn as nn

from structure.Decoder import Decoder
from structure.Encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, 
        source_vocab_size, 
        target_vocab_size, 
        source_padding_index, 
        target_padding_index, 
        embed_size=256, 
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_len=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            source_vocab_size,
            embed_size,
            num_layers,
            heads,
            device, 
            forward_expansion,
            dropout,
            max_len
        )
        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_len
        )

        self.source_padding_index = source_padding_index
        self.target_padding_index = target_padding_index
        self.device = device
    
    def make_source_mask(self, src):
        source_mask = (src != self.source_padding_index).unsqueeze(1).unsqueeze(2)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones(target_len, target_len)).expand(N, 1, target_len, target_len)

        return target_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)

        encoder_source = self.encoder(source, source_mask)
        output = self.decoder(target, encoder_source, source_mask, target_mask)

        return output