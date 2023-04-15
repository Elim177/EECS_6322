import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import SIMCLRLoss
import Resnet_Model
import MemoryBank
from MemoryBank import fill_memory_bank
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up dataset and data loader and an augmenation which
# is similar to the one performed in the paper description
transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# download the dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
train_dataset_transformed = transform(train_loader)
validation_dataset_transformed = transform(validation_loader)
#set up the model and optimizer
model = Resnet_Model.ResNet().to(device)
#use the adam optimizer for the model
optimizer = optim.Adam(model.parameters(), lr=4e-4)
#set up the loss function
criterion = SIMCLRLoss(batch_size=128, temperature=0.1)
#load the Memorybank
memory_bank_train = MemoryBank(len(train_dataset_transformed), 512, 10, 0.1)
memory_bank_valid = MemoryBank(len(validation_dataset_transformed), 512, 10, 0.1)
#training loop
num_epochs = 100
for epoch in range(num_epochs):
    for images, _ in train_loader:
        # concatenate the images which is the "augementations"
        images = torch.cat((images, images.flip(3)), dim=0).to(device)  
        features = model(images)
        loss = criterion(features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # save in the memory bank
        print('Fill memory bank for K nearest neighbours')
        fill_memory_bank(train_loader, model, train_dataset_transformed)
        # Checkpoint
        print('Checkpt')
        torch.save({'optimizer': optimizer.state_dict(),
                     'model': model.state_dict(), 
                    'epoch': epoch + 1}, 'pretext_check_pt_file')
#save final model
torch.save(model.state_dict(), 'final_pretext_model')
#save top neighbours for SCAN step
print('Fill memory bank for mining the nearest neighbors (train) ...')
fill_memory_bank(train_loader, model, train_dataset_transformed)
print('Mine the nearest neighbors (Top-%d)' %(20)) 
indices, acc = memory_bank_train.mine_nearest_neighbors(20)
print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(20, 100*acc))
np.save(['topk_neighbors_train_path'], indices)   
#this is for validation
print('Fill memory bank for mining the nearest neighbors (val)')
fill_memory_bank(validation_loader, model, memory_bank_valid)
print('Mine the nearest neighbors (Top-%d)' %(5)) 
indices, acc = memory_bank_valid.mine_nearest_neighbors(5)
print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(5, 100*acc))
np.save(['topk_neighbors_val_path'], indices)   