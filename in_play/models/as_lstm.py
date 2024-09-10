import torch
import torch.nn as nn

class AS_LSTM(nn.Module):
    """
    LSTM model with an attention mechanism for sequence prediction.
    
    Parameters:
    - input_dim (int): Number of input features.
    - hidden_dim (int): Number of hidden units in the LSTM.
    - num_layers (int): Number of LSTM layers.
    - output_dim (int): Number of output features.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        Initialises the AS_LSTM model with the specified parameters.
        
        Parameters:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the LSTM hidden state.
        num_layers (int): Number of LSTM layers.
        output_dim (int): Dimensionality of the output features.
        """
        super(AS_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the attention layer (linear transformation to compute attention scores)
        self.attention = nn.Linear(hidden_dim, 1)

        # Define the output layer (linear transformation for final prediction)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the AS_LSTM model, including LSTM and attention mechanism.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
        Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Initialise hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # LSTM forward pass (output shape: (batch_size, sequence_length, hidden_dim))
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Attention mechanism: apply softmax to compute attention weights
        attention_weights = torch.softmax(self.attention(out), dim=1)
        context_vector = torch.sum(attention_weights * out, dim=1)  # Weighted sum of LSTM outputs

        # Pass the context vector through the output layer
        out = self.linear(context_vector)
        return out
