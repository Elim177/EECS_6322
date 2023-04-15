import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# define a base network to use for simCLR
class Base_Model_Contrastive(nn.Module):
    def __init__(self, hidden_size=512):
        super(Base_Model_Contrastive, self).__init__()
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
        features = Functional.normalize(features, dim=1)
        return features
# Define the model with a memory bank
class SIMCLR(nn.Module):
    # initialize the necessary variables
    def __init__(self, hidden_size=512, temperature=0.5, memory_size=8192):
        super(SIMCLR, self).__init__()
        self.contrastive_model = Base_Model_Contrastive(hidden_size=hidden_size)
        self.temperature = temperature
        self.memory_size = memory_size
        self.register_buffer("memory_bank", torch.randn(self.memory_size, hidden_size))
    # an alternate for the memory bank
    def save_to_memory(self, features):
        # Save the features to the memory bank
        with torch.no_grad():
            idx = torch.randint(0, self.memory_size, (features.shape[0],))
            self.memory_bank[idx] = features.detach()
    # the forward function
    def forward(self, x1, x2):
        # initialize the two views of the input
        features1 = self.contrastive_model(x1)
        features2 = self.contrastive_model(x2)
        # save the features to the memory bank
        self.save_to_memory(features1)
        self.save_to_memory(features2)
        # concatenate them
        features = torch.cat([features1, features2], dim=0)
        # this is the similarity values
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix /= self.temperature
        # reconstruct the shape
        batch_size = x1.shape[0]
        target_value = torch.arange(batch_size, device=similarity_matrix.device)
        target_value = torch.cat([target_value + batch_size, target_value])
        # value for the cross entropy loss
        loss_value = Functional.cross_entropy(similarity_matrix, target_value)
        return loss_value
