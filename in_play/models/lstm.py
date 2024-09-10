import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    """
    A basic LSTM (Long Short-Term Memory) model for sequence prediction.

    Parameters:
    - input_dim (int): The number of input features.
    - hidden_dim (int): The number of hidden units in the LSTM.
    - num_layers (int): The number of LSTM layers.
    - output_dim (int): The number of output features.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        Initialises the LSTM model with the specified parameters.
        
        Parameters:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden state.
        num_layers (int): Number of LSTM layers.
        output_dim (int): Dimensionality of the output features.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer (fully connected layer)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the LSTM model.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
        Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Initialise hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Perform the forward pass through the LSTM
        # Detach hidden states to avoid backpropagating all the way through the sequence
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Pass the hidden state from the last time step to the fully connected layer
        out = self.linear(out[:, -1, :])
        return out
