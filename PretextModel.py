import numpy as np
import torch
# import Faiss for the purposes of fast nearest neighbor search
import faiss

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        # initialize memory bank object
        # number of samples that can be stored
        self.n = n
        # dimensionality of the features
        self.dim = dim
        # tensor to store features
        self.features = torch.FloatTensor(self.n, self.dim) 
        # tensor to store targets
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0  # pointer to indicate where to store next set of samples
        self.device = 'cpu'  # device on which to store tensors
        self.K = 100  # number of nearest neighbors to consider for knn
        self.temperature = temperature  # temperature parameter for weighted knn
        self.C = num_classes  # number of classes

    def weighted_knn(self, predictions):
        # perform weighted k-nearest neighbor (knn) classification using stored features
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)  # one-hot tensor to hold retrieved targets
        batchSize = predictions.shape[0]  # batch size of the predictions
        correlation = torch.matmul(predictions, self.features.t())  # calculate correlation between predictions and stored features
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)  # get indices and values of K nearest neighbors
        candidates = self.targets.view(1,-1).expand(batchSize, -1)  # tensor of target indices for every sample
        retrieval = torch.gather(candidates, 1, yi)  # retrieve target indices for K nearest neighbors
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()  # resize one-hot tensor
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)  # set values of one-hot tensor to retrieved target indices
        yd_transform = yd.clone().div_(self.temperature).exp_()  # transform values of distance using temperature parameter
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)  # compute probability of each class using weighted knn
        _, class_preds = probs.sort(1, True)  # get sorted indices of predicted classes
        class_pred = class_preds[:, 0]  # retrieve predicted classes

        return class_pred

    def knn(self, predictions):
        # perform k-nearest neighbor (knn) classification using stored features
        correlation = torch.matmul(predictions, self.features.t())  # calculate correlation between predictions and stored features
        sample_pred = torch.argmax(correlation, dim=1)  # get index of nearest neighbor
        class_pred = torch.index_select(self.targets, 0, sample_pred)  # retrieve target of nearest neighbor
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample using Faiss
        features = self.features.cpu().numpy()  # get stored features as numpy array
        n, dim = features.shape[0], features.shape[1]  # get number of samples and dimensionality of features
        index = faiss.IndexFlatIP(dim)  # initialize Faiss index for inner product search
        index = faiss.index_cpu_to_all_gpus(index)  #
