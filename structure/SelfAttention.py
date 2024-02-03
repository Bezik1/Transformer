import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert(self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fully_connected_output = nn.Linear(heads*self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Parameters
        values = self.values(values)
        values = values.reshape(N, value_len, self.heads, self.head_dim)

        keys = self.keys(keys)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)

        queries = self.queries(query)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Concat keys and queries
        # energy[N, heads, query_len, key_len]
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # out[N, query_len, heads, head_dim] -> flatten to 3d array
        output = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        output = self.fully_connected_output(output)

        return output

