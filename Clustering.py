
import torch
import numpy as np
import torch.nn.functional as functional

# this is the basic scan train
def scan_train(train_loader, model, criterion, opt, epoch_value, update_cluster_head_only=False):
    total_losses = []
    entropy_losses = []
    # if it is the cluster head
    if update_cluster_head_only:
        model.eval()
    else:
        model.train()
    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        # calculate gradient just for backprop of linear layer 
        if update_cluster_head_only: 
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')
        # calculate gradient for backprop of complete network
        else:
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)     
        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            entropy_loss.append(entropy_loss_)
        #register the mean loss and backprop the total loss to cover all subheads
        total_losses.append(np.mean([v.item() for v in total_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))
        total_loss = torch.sum(torch.stack(total_loss, dim=0))
        opt.zero_grad()
        total_loss.backward()
        opt.step()

def get_predictions(dataloader, model):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = []
    targets = []
    for batch in dataloader:
        images = batch['image'].cuda(non_blocking=True)
        output = model(images)
        predictions.append(torch.argmax(output, dim=1))
        targets.append(batch['target'])
    predictions = torch.cat(predictions, dim=0).cpu()
    targets = torch.cat(targets, dim=0)
    output = [{'predictions': predictions,'targets': targets}]
    return output

# calculate the entropy using probabilities
def entropy(probs, input_as_probabilities=False):
    if input_as_probabilities:
        log_probs = torch.log(probs + 1e-10)
    else:
        log_probs = probs
    return -torch.sum(probs * log_probs, dim=1)

# this will calculate the total loss
def calculate_total_loss(predictions):
    # calculate the total loss
    num_heads = len(predictions)
    total_losses = []
    # loop through the 
    for head in predictions:
        probs = head['predictions']
        neighbors = head['targets']
        anchors = torch.arange(neighbors.size(0)).view(-1, 1).expand_as(neighbors)
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True)
        # the similiatiry value for the given probablities
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = functional.binary_cross_entropy(similarity, ones)
        total_loss = -entropy_loss + consistency_loss
        # add the loss to the the list
        total_losses.append(total_loss.item())
    # calculate the lowest and lowest loss heads
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)
    output = [{'lowest_loss_head': lowest_loss_head,'lowest_loss': lowest_loss}]
    return output


