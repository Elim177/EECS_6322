import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define the encoder network
class Encoder(nn.Module):
    def __init__(self, hidden_size=512):
        super(Encoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(512, hidden_size)

    def forward(self, x):
        features = self.cnn(x)
        features = features.mean(dim=[2, 3])
        features = self.fc(features)
        features = F.normalize(features, dim=1)
        return features

# Define the model with a memory bank
class SIMCLR(nn.Module):
    def __init__(self, hidden_size=512, temperature=0.5, memory_size=8192):
        super(SIMCLR, self).__init__()

        self.encoder = Encoder(hidden_size=hidden_size)
        self.temperature = temperature
        self.memory_size = memory_size
        self.register_buffer("memory_bank", torch.randn(self.memory_size, hidden_size))

    def save_memory(self, features):
        # Save the features to the memory bank
        with torch.no_grad():
            idx = torch.randint(0, self.memory_size, (features.shape[0],))
            self.memory_bank[idx] = features.detach()

    def forward(self, x1, x2):
        # Encode the two views of the input
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)

        # Save the features to the memory bank
        self.save_memory(features1)
        self.save_memory(features2)

        # concatenate them
        features = torch.cat([features1, features2], dim=0)
        # this is the similarity values
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix /= self.temperature
        # reconstruct the shape
        batch_size = x1.shape[0]
        target = torch.arange(batch_size, device=similarity_matrix.device)
        target = torch.cat([target + batch_size, target])
        # value for the entropy loss
        loss = F.cross_entropy(similarity_matrix, target)

        return loss
