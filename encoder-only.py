from torch import nn

def padding_mask(tensor_input):
    return (tensor_input == 0)


class EncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, ff_dim, dropout = 0):
        super().__init__()

        # Encoder embedding and layer
        self.encoder_embedding = nn.Embedding(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)

        # Encoder/Decoder stack
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.fc_out_model = nn.Linear(d_model, output_dim)
        self.fc_out_seq = nn.Linear(d_model, 1)


    def forward(self, src, tgt):
        # Source encoded with padding mask
        src_emb = self.encoder_embedding(src)
        src_encoded = self.encoder(src_emb)
        tgt_compressed = self.fc_out_model(src_encoded)

        return self.fc_out_seq(tgt_compressed.transpose(0,1))
    
    def train(self, data, criterion, optim, lr):
        pass