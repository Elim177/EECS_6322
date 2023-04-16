import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ClusteringLoss
import Resnet_Model
import numpy as np
from Clustering import calculate_total_loss, get_predictions
from SelfLabelLoss import selflabel_train
import os
from Hungarian_Evaluater import hungarian_evaluate

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
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
train_dataset_transformed = transform(train_loader)
validation_dataset_transformed = transform(validation_loader)
#set up the model and optimizer
model = Resnet_Model.ResNet().to(device)
#use the adam optimizer for the model
optimizer = optim.Adam(model.parameters(), lr=4e-4)
#setup the loss function
criterion = ClusteringLoss(batch_size=128, temperature=0.1)
epoch = 50
for epoch in range(epoch):
        print('Epoch %d/%d' %(epoch+1, epoch))
        # Train
        print('***********Train********')
        selflabel_train(train_loader, model, criterion, optimizer, epoch, False)
        # Evaluate 
        print('********Make prediction on validation set********')
        predictions = get_predictions(validation_loader, model)
        print('********Evaluate based on SCAN loss********')
        scan_stats = calculate_total_loss(predictions)
        # print(scan_stats)
        lowest_loss_head = scan_stats['lowest_loss_head']
        lowest_loss = scan_stats['lowest_loss']
        torch.save({'model': model.module.state_dict(), 'head': lowest_loss_head}, 'self_label_model')

# reload the last updated checkpoint
model_checkpoint = torch.load('self_label_model', map_location='cpu')
model.module.load_state_dict(model_checkpoint['model'])
# get the prediction values
predictions = get_predictions(validation_loader, model)
clustering_values = hungarian_evaluate(model_checkpoint['head'], predictions, 
                            class_names=validation_dataset.dataset.classes, 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(['scan_dir'], 'confusion_matrix_self_label.png'))
print(clustering_values)     
