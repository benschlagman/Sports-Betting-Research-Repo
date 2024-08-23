import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, sequence_length, num_classes, num_runners):
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
        x = x.permute(0, 2, 1)  # Convert to [batch_size, input_dim, sequence_length] for Conv1D
        
        # Convolutional and Pooling Layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten and Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        x = x.view(x.size(0), 3, 2)  # Reshape to [batch_size, num_runners, num_classes]
        return x