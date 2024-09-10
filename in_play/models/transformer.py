import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    """
    Applies positional encoding to input data for use in a transformer model.
    Positional encoding helps the model understand the order of sequences, as transformers lack built-in recurrence or convolution.

    Parameters:
    - d_model (int): The dimensionality of the model (embedding size).
    - max_len (int): The maximum length of the input sequence.
    """

    def __init__(self, d_model, max_len=1000):
        """
        Initialises the positional encoding layer.
        
        Parameters:
        d_model (int): Dimensionality of the model.
        max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        
        Parameters:
        x (Tensor): Input tensor of shape (sequence_length, batch_size, d_model).
        
        Returns:
        Tensor: Input tensor with positional encoding added.
        """
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    """
    A transformer model for sequence-to-sequence tasks, such as time series prediction or natural language processing.

    Parameters:
    - input_dim (int): Number of input features.
    - d_model (int): Dimensionality of the model.
    - nhead (int): Number of attention heads.
    - num_encoder_layers (int): Number of encoder layers.
    - num_decoder_layers (int): Number of decoder layers.
    - output_dim (int): Number of output features.
    - dropout (float): Dropout rate.
    - max_len (int): Maximum sequence length.
    """

    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, output_dim, dropout=0.1, max_len=1000):
        """
        Initialises the Transformer model with the specified parameters.
        
        Parameters:
        input_dim (int): The dimensionality of the input features.
        d_model (int): Dimensionality of the transformer model.
        nhead (int): Number of attention heads in the multi-head attention mechanism.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        output_dim (int): Dimensionality of the output features.
        dropout (float): Dropout rate for regularisation.
        max_len (int): Maximum sequence length.
        """
        super(TransformerModel, self).__init__()

        # Embedding layer to transform input_dim to d_model size required by transformer
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding to help the model understand the sequence order
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer with encoder-decoder structure
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout)

        # Final fully connected output layer
        self.fc = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz):
        """
        Generates a mask to prevent attention to future time steps in the sequence (used for autoregressive tasks).

        Parameters:
        sz (int): The size of the sequence.

        Returns:
        Tensor: A square mask of shape (sz, sz) with zeros in the lower triangular part and -inf in the upper triangular part.
        """
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        Defines the forward pass of the Transformer model.

        Parameters:
        src (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
        Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Embed the input and apply positional encoding
        src = self.embedding(src)
        src = self.positional_encoding(src.permute(1, 0, 2))  # Permute for transformer [sequence_length, batch_size, d_model]

        # Generate mask for the transformer (for autoregressive tasks)
        mask = self._generate_square_subsequent_mask(src.size(0)).to(src.device)

        # Pass through the transformer
        output = self.transformer(src, src, src_mask=mask, tgt_mask=mask)

        # Extract the output of the last time step
        output = output[-1]

        # Pass through the final linear layer
        output = self.fc(output)

        return output
