import torch.nn as nn
import torch

def causal_mask(tensor_input):
    seq_length = tensor_input.size(1)  # Assume tensor_input has shape (batch_size, seq_length)
    return torch.triu(torch.ones(seq_length,seq_length), diagonal=1) == 0

def padding_mask(tensor_input):
    return (tensor_input == 0)


class EncoderDecoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, ff_dim, dropout = 0):
        super().__init__()

        # Encoder embedding and layer
        self.encoder_embedding = nn.Embedding(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)

        # Decoder embedding and layer
        self.decoder_embedding = nn.Embedding(output_dim, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim, dropout)

        # Encoder/Decoder stack
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)


    def forward(self, src, tgt):
        # Source encoded with padding mask
        src_emb = self.encoder_embedding(src)
        src_padding_mask = padding_mask(src).to(src.device)
        src_encoded = self.encoder(src_emb)

        # Target decoded with causal & padding mask
        tgt_emb = self.decoder_embedding(tgt)
        tgt_padding_mask = padding_mask(tgt).to(src.device)
        tgt_causal_mask = causal_mask(tgt).to(src.device)
        tgt_decoded = self.decoder(tgt_emb, src_encoded)

        return self.fc_out(tgt_decoded)
    
    def train(self, data, criterion, optim, lr):
        pass