import torch
import torch.nn as nn
import numpy as np

class HeavyweightCNN(nn.Module):
    """
    Simulated Heavyweight CNN for high-accuracy radiology screening.
    """
    def __init__(self, config: dict):
        super(HeavyweightCNN, self).__init__()
        self.accuracy = config['simulation']['heavyweight_cnn']['accuracy']
        self.latency = config['simulation']['heavyweight_cnn']['latency']
        
        # Dummy layers to make it a valid torch module
        self.conv = nn.Conv2d(1, 16, kernel_size=3)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc(self.conv(x))

    def predict_simulated(self):
        """
        Returns simulated accuracy and latency.
        """
        actual_acc = np.clip(np.random.normal(self.accuracy, 0.02), 0, 1)
        return actual_acc, self.latency
