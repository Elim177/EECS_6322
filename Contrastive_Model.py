import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet18

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, h, z):
        # Compute cosine similarity matrix
        sim_matrix = torch.mm(z, z.t())
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        # Generate mask
        batch_size = h.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        neg_mask = ~mask
        # Compute positive and negative logits
        pos_logits = torch.diagonal(sim_matrix, offset=1)[mask].view(batch_size, -1)
        neg_logits = sim_matrix[neg_mask].view(batch_size, -1)
        # Compute contrastive loss
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long).to(device)
        loss = F.cross_entropy(logits/self.temperature, labels)
        return loss

# load in the data set as well as the augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# use the CIFAR10 dataset for the evaluation of the model
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = resnet18(pretrained=False)
encoder.fc = nn.Identity()
model = SimCLR(encoder=encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Initialize contrastive loss function
criterion = ContrastiveLoss(temperature=0.5)
# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        h, z = model(x)
        loss = criterion(h, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch Number [{epoch+1}/{num_epochs}], Step Value [{i}/{len(train_loader)}], Loss Value: {loss.item():.4f}")
