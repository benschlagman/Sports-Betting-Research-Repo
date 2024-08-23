import torch
import torch.nn as nn


class AS_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(AS_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the attention layer
        self.attention = nn.Linear(hidden_dim, 1)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(out), dim=1)  # Apply softmax to get attention weights
        context_vector = torch.sum(attention_weights * out, dim=1)  # Weighted sum of LSTM outputs

        # Pass the context vector through the output layer
        out = self.linear(context_vector)
        return out
