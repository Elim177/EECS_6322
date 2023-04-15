import torch
import torch.nn as nn

# this is the orginal version of simCLR criterion
class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
    # forward function to calculate the loss
    def forward(self, features):
        # batch size , 
        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()
        # concatenate the features
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:,0]
        # do the dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        # for numerical stability purposes
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask
        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        return loss