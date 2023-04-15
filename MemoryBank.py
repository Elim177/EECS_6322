import numpy as np
import torch
# import Faiss for the purposes of fast nearest neighbor search
import faiss
class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        #initialize memory bank object
        # number of samples that can be stored
        self.n = n
        #dimensionality of the features
        self.dim = dim
        # tensor to store features
        self.features = torch.FloatTensor(self.n, self.dim) 
        # tensor to store targets
        self.targets = torch.LongTensor(self.n)
        # pointer to indicate where to store next set of samples
        self.ptr = 0  
        # device on which to store tensors
        self.device = 'cpu'  
        #number of nearest neighbors to consider for knn
        self.K = 100 
        self.temperature = temperature 
        self.C = num_classes
    
    def weighted_knn(self, predictions):
        # perform weighted k-nearest neighbor (knn) classification using stored features
        # one-hot tensor to hold retrieved targets
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device) 
        # batch size of the predictions
        batchSize = predictions.shape[0]  
        # calculate the correlation between predictions and stored features
        correlation = torch.matmul(predictions, self.features.t()) 
        # get indices and values of K nearest neighbors
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True) 
        # tensor of target indices for every sample
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        # retrieve target indices for K nearest neighbors
        retrieval = torch.gather(candidates, 1, yi)
        # resize one-hot tensors 
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_() 
        # set values of one-hot tensor to retrieved target indices
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        # transform values of distance using temperature parameter
        yd_transform = yd.clone().div_(self.temperature).exp_()
        # compute probability of each class using weighted knn
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)  # get sorted indices of predicted classes
        # retrieve predicted classes
        class_pred = class_preds[:, 0] 

        return class_pred
    def fill_memory_bank(self,dataloader, model, memory_bank):
        model.eval()
        memory_bank.reset()
        for im, batches in enumerate(dataloader):
            images = batches['image'].cuda(non_blocking=True)
            targets = batches['target'].cuda(non_blocking=True)
            output = model(images)
            memory_bank.update(output,targets)
            if im % 100 == 0:
                print('Fill Memory Bank [%d/%d]' %(im, len(dataloader)))
                
    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
       # mine the topk nearest neighbors for every given image
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        # the given image is also included here
        distances, indices = index.search(features, topk+1) 
        # evaluate the accuracy
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            # remove the given image for general calculation
            neighbor_targets = np.take(targets, indices[:,1:], axis=0)
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        else:
            return indices
        
    def k_nearest_neighbours(self, predictions):
        #perform k-nearest neighbor (knn) classification using stored features
        #calculate the correlation between predictions and stored features
        correlation = torch.matmul(predictions, self.features.t())
        #get index of nearest neighbor
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)  # retrieve target of nearest neighbor
        return class_pred