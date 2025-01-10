import torch.nn as nn
import torch

def causal_mask(tensor_input):
    seq_length = tensor_input.size(1)  # Assume tensor_input has shape (batch_size, seq_length)
    return torch.triu(torch.ones(seq_length,seq_length), diagonal=1) == 0

def padding_mask(tensor_input):
    return (tensor_input == 0)


class DecoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, ff_dim, dropout = 0):
        super().__init__()

        # Decoder embedding and layer
        self.decoder_embedding = nn.Embedding(output_dim, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim, dropout)

        # Encoder/Decoder stack
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)


    def forward(self, src, tgt):

        # Target decoded with causal & padding mask
        tgt_emb = self.decoder_embedding(tgt)
        tgt_padding_mask = padding_mask(tgt).to(src.device)
        tgt_causal_mask = causal_mask(tgt).to(src.device)
        tgt_decoded = self.decoder(tgt_emb, memory = None)

        return self.fc_out(tgt_decoded)
    
    def train(self, data, criterion, optim, lr):
        pass