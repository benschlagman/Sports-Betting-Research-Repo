import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, output_dim, dropout=0.1, max_len=1000):
        super(TransformerModel, self).__init__()

        # Embedding layer to transform input_dim to d_model as required by transformer
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout)

        # Output layer
        self.fc = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # Embed and add positional encoding
        src = self.embedding(src)
        src = self.positional_encoding(src.permute(1, 0, 2))

        # Generate mask for the transformer
        mask = self._generate_square_subsequent_mask(src.size(0)).to(src.device)

        # Pass through the transformer with src masking
        output = self.transformer(src, src, src_mask=mask, tgt_mask=mask)

        # Extract the output of the last time step
        output = output[-1]

        # Pass through final linear layer
        output = self.fc(output)

        return output
