# -*- coding: utf-8 -*-

# libraries
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer

# It computes the relevance between a query and a set of key-value pairs
def scaled_dot_product(q, k, v, mask=None): # the query, key, values should be vectors of the same dimensions
    d_k = q.size()[-1] # dimension
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # dot product between the query and each key
    attn_logits = attn_logits / math.sqrt(d_k) # dividing the dot products by the square root of the dimension this helps in avoiding large values that can cause numerical instability
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1) # applying a softmax function to obtain the attention weights
    values = torch.matmul(attention, v) #attention weights are used to combine the values into a single output vector
    return values, attention

# Sometimes the input dimension could be a prime number. So it would be hard to change the number of heads
class MHADiv(nn.Module): # MHA allows the neural network to learn different aspects or perspectives of the data by using different attention heads
    def __init__(self, input_dim, embed_dim, num_heads,device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads  # Number of heads
        self.head_dim = embed_dim // num_heads # Number of dimensions in vectors in each head


        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim) # Linear layer for linear transform
        self.o_proj = nn.Linear(embed_dim, input_dim) # output Layer

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, input_dim = x.size()
        qkv = self.qkv_proj(x)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)

        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):
    def __init__(self, input_dim,emb_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MHADiv(input_dim, emb_dim, num_heads)

        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        # print(attn_out.shape)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

class TransformerModel(nn.Module):

  def __init__(self, input_size, emb_size, num_heads, dropout, num_layers, max_sequence_size, device):
    super(TransformerModel, self).__init__()

    self._input_size = input_size
    self._dropout = dropout
    self._device = device
    full_sequence_size = max_sequence_size + 1

    self._pos_encoder = Summer(PositionalEncodingPermute1D(full_sequence_size))

    encoder_layer = EncoderBlock(input_size,emb_size, num_heads, dim_feedforward, dropout)
    self._encoder = TransformerEncoder(num_layers=num_layers,
            emb_dim = emb_size,
            input_dim=input_size,
            dim_feedforward=dim_feedforward,
            num_heads= num_heads,
            dropout= dropout)
    self._linear = nn.Linear(input_size, 1)
    self.sigmoid = nn.Sigmoid()
  def forward(self, sequence):
    batch_size = sequence.shape[0]

    sequence = torch.cat([torch.zeros(batch_size, 1, self._input_size).to(self._device), sequence], dim=1)
    sequence = sequence * math.sqrt(self._input_size)
    sequence = self._pos_encoder(sequence)

    result = self._encoder(sequence)
    result = result[:, 0, :]
    result = F.dropout(result, self._dropout, self.training)
    result = self._linear(result)
    result=self.sigmoid(result)
    return result

def mod_input_dim(input_dim, num_heads):
  r = input_dim % num_heads
  embed_dim = input_dim - r
  return embed_dim

input_dim = 67
num_heads = 4
embed_dim = mod_input_dim(input_dim, num_heads)
dropout = 0.2
num_layers = 4
max_sequence_size = 100
dim_feedforward = 2048
device = "cuda:0"
model = TransformerModel(input_dim, embed_dim, num_heads, dropout, num_layers, max_sequence_size, device)

print(model)