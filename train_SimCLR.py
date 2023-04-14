import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import SimCLR
import SIMCLRLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up dataset and data loader
transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# set up the model and optimizer
model = SimCLR().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# set up the loss function
criterion = SimCLRLoss(batch_size=128, temperature=0.5)

# training loop
num_epochs = 100

for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = torch.cat((images, images.flip(3)), dim=0).to(device)  # augmentations
        features = model(images)
        loss = criterion(features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
