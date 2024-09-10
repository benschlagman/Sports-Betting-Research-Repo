import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for processing sequential data for multiple runners.

    Parameters:
    - input_dim (int): The number of input features.
    - sequence_length (int): The length of the input sequence.
    - num_classes (int): The number of output classes.
    - num_runners (int): The number of runners (e.g., 3 for a 3-runner race).
    """

    def __init__(self, input_dim, sequence_length, num_classes, num_runners):
        """
        Initialises the CNN with convolutional layers, pooling layers, and fully connected layers.

        Parameters:
        input_dim (int): The dimensionality of the input features.
        sequence_length (int): The length of the input sequence.
        num_classes (int): The number of output classes.
        num_runners (int): The number of runners (e.g., 3 runners).
        """
        super(CNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * (sequence_length // 4), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes * num_runners)
        
    def forward(self, x):
        """
        Defines the forward pass of the CNN model.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
        Tensor: Output tensor of shape (batch_size, num_runners, num_classes).
        """
        # Rearrange input to [batch_size, input_dim, sequence_length] for Conv1D
        x = x.permute(0, 2, 1)
        
        # First Convolutional and Pooling Layer
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Second Convolutional and Pooling Layer
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Reshape the output to [batch_size, num_runners, num_classes]
        x = x.view(x.size(0), 3, 2)
        return x
