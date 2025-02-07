import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

torch.manual_seed(50)

PAD_TOKEN_ID = 0

# ----------------------
# 1) Custom Tokenizer
# ----------------------
class CustomTokenizer():
    def __init__(self):
        self.vocab = {
            ' ': 0,  # pad token
            '0': 1,
            '1': 2,
            '2': 3,
            '3': 4,
            '4': 5,
            '5': 6,
            '6': 7,
            '7': 8,
            '8': 9,
            '9': 10,
            '{': 11,
            '}': 12,
            ',': 13,
            '-': 14
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = ' '  # space character (index 0)
        self.pad_token_id = self.vocab[self.pad_token]
        self.EOS_token_id = self.vocab['}']
        self.BOS_token_id = self.vocab['{']

    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            match_found = False
            # Iterate over vocab keys to find the longest match starting at i
            for token_str in sorted(self.vocab.keys(), key=len, reverse=True):
                if text[i:i + len(token_str)] == token_str:
                    tokens.append(self.vocab[token_str])
                    i += len(token_str)  # Advance the pointer
                    match_found = True
                    break
            if not match_found:
                # If no match is found, use 0 or fallback
                tokens.append(self.vocab.get(text[i], 0))
                i += 1
        return tokens

    def detokenize(self, tokens):
        missing_tokens = []
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        for token in tokens:
            token_id = token if isinstance(token, int) else token.item()
            if token_id not in self.inv_vocab:
                missing_tokens.append(token_id)
        return (''.join([self.inv_vocab[t] for t in tokens if t not in missing_tokens])).replace(" ","")

# ----------------------
# 2) Open dat File -- along with stringy/tokenized data object
# ----------------------
def open_dat_file(folder_path):
    inputs = []
    outputs = []
    for file_name in os.listdir(folder_path):
            if file_name.endswith('.dat'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        input_text, output_text = line.split("->")
                        inputs.append(input_text.strip())
                        outputs.append(output_text.strip())
    return inputs, outputs

def pad_tokens(sequences, max_len, padding_value=PAD_TOKEN_ID):
    return  torch.tensor([token_seq + [padding_value]*(max_len - len(token_seq)) for token_seq in sequences], dtype=torch.long)
    
class FIRE6Dataset(Dataset):
    def __init__(self, folder_path, tokenizer, max_len=1000):
        self.input_strings, self.output_strings = open_dat_file(folder_path)

        self.tokenizer = tokenizer

        self.input_tokens = [self.tokenizer.tokenize(str) for str in self.input_strings]
        self.output_tokens = [self.tokenizer.tokenize(str) for str in self.output_strings]

        self.input_padded = pad_tokens(self.input_tokens,max_len)
        self.output_padded = pad_tokens(self.output_tokens,max_len)

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return self.input_padded[idx], self.output_padded[idx]

# ----------------------
# 3) Positional Encoding
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to `pe`
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (B, S, d_model)
        """
        seq_len = x.size(1)  # Sequence length is in dim 1 for (B, S, d_model)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Input sequence length ({seq_len}) exceeds max_len ({self.pe.size(1)}).")
        
        # Add positional encodings
        x = x + self.pe[:, :seq_len, :]  # Slice positional encodings to match input length
        return self.dropout(x)
    
# ---------------------------
# 4) Encoder & Decoder Stacks
# ---------------------------

class EncoderStack(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, num_layers, dropout):
        # Inherit the constructor of PyTorch Module
        super(EncoderStack, self).__init__()

        # Define the encoder layer and stack
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_ffn, nhead=nhead, dropout=dropout)
        self.layers = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.layers(src)

class DecoderStack(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, num_layers, dropout):
        # Inherit the constructor of PyTorch Module
        super(DecoderStack, self).__init__()
        
        # Define the decoder layer and stack
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_ffn, nhead=nhead, dropout=dropout)
        self.layers = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def causal_mask(self, n):
        mask = torch.triu(torch.ones(n, n), diagonal=1)  # Upper triangular mask
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0)  
        return mask.float()

    def forward(self, tgt, memory, maskQ = False):
        # Causal mask, tgt and src (i.e. encoder memory)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.causal_mask(tgt_seq_len).to(tgt.device)

        return self.layers(tgt, memory, tgt_mask = tgt_mask, tgt_is_causal = True)


# ----------------------
# 5) Core Model that embodies the space of integrals (this is what we train)
# ----------------------
class KernelModel(nn.Module):
    def __init__(self, vocab_size, d_model=96, d_ffn=96, nhead=8, encoder_layers=4, decoder_layers=4, dropout=0.1):
        # Causal mask, tgt and src (i.e. encoder memory)
        super(KernelModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = EncoderStack(d_model, d_ffn, nhead, encoder_layers, dropout)
        self.decoder = DecoderStack(d_model, d_ffn, nhead, decoder_layers, dropout)

        self.fc_out = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.d_fnn = d_ffn
        self.nhead = nhead
        self.encoder_layers = decoder_layers
        self.decoder_layers = decoder_layers
        self.vocab_size = vocab_size



    def forward(self, src, tgt):
        """
        If tgt is given, do encoder-decoder pass.
        If tgt is None, just do encoder pass (for inference).

        Expects src, tgt shapes = (batch_size, seq_len).
        """
        # Embed + positional encode
        src_emb = self.embedding(src)         # (B, S, d_model)
        src_emb = self.pos_encoder(src_emb)   # (B, S, d_model)

        tgt_emb = self.embedding(tgt)         # (B, T, d_model)
        tgt_emb = self.pos_decoder(tgt_emb)   # (B, T, d_model)

        # Encode source sequence
        memory = self.encoder(src_emb)  # => (B, S, d_model)
        
        output = self.decoder(tgt_emb, memory)  # => (B, T, d_model)
        return self.fc_out(output)    # => (B, T, vocab_size)

# ----------------------
# 6) Inference Engine builds the output from input sequence (this is what translates)
# ----------------------

class ProjectionModel(nn.Module):
    def __init__(self, kernelmodel, tokenizer, max_length):
        super(ProjectionModel, self).__init__()
        self.kernelmodel = kernelmodel
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.BOS_token = self.tokenizer.BOS_token_id
        self.EOS_token = self.tokenizer.EOS_token_id
        self.kernelmodel.dropout = 0

    def forward(self, input_string):
        index = 0
        input_tokens = pad_tokens([self.tokenizer.tokenize(input_string)], self.max_length)
        start_tokens = pad_tokens([[self.tokenizer.BOS_token_id]], self.max_length)
        while index < self.max_length-1:
            index += 1
            logits = self.kernelmodel(input_tokens, start_tokens)
            start_tokens[0][index] = torch.argmax(torch.softmax(logits[0][index],dim=0),dim=0)
            if start_tokens[0][index] == self.tokenizer.EOS_token_id:
                index = self.max_length
        return self.tokenizer.detokenize(start_tokens[0])