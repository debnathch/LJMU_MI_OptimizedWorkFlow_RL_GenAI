import torch
import torch.nn as nn
import numpy as np

class LightweightCNN(nn.Module):
    """
    Simulated Lightweight CNN for rapid radiology screening.
    """
    def __init__(self, config: dict):
        super(LightweightCNN, self).__init__()
        self.accuracy = config['simulation']['lightweight_cnn']['accuracy']
        self.latency = config['simulation']['lightweight_cnn']['latency']
        
        # Dummy layers to make it a valid torch module
        self.conv = nn.Conv2d(1, 4, kernel_size=3)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        # In a real scenario, this would perform inference
        return self.fc(self.conv(x))

    def predict_simulated(self):
        """
        Returns simulated accuracy and latency.
        """
        # Add some random noise to accuracy
        actual_acc = np.clip(np.random.normal(self.accuracy, 0.05), 0, 1)
        return actual_acc, self.latency
