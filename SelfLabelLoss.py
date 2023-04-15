
import torch

# train the self label data
def selflabel_train(train_loader, model, criterion, optimizer, epoch):
    losses = []
    model.train()
    for i, batch in enumerate(train_loader):
        # get the orginal image and the augmented image
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        # loop thrugh the output weights
        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]
        # assign a value to the loss
        loss = criterion(output, output_augmented)
        losses.append(loss.item())
        # zero the gradient of the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses = sum(losses)/len(losses)
    output = [{'losses': losses}]
    return output