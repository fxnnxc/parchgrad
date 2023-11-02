import torch 
from torch.utils.data import Dataset 



class PerturbedMNIST(Dataset):
    def __init__(self, base_dataset, epsilon=0.1):
        self.data = base_dataset
        self.epsilon = epsilon
        
    def __getitem__(self, index):
        x, y = self.data.__getitem__(index)
        epsilon = torch.normal(torch.zeros_like(x), torch.zeros_like(x).fill_(self.epsilon))
        return x + epsilon, y
    
    def __len__(self):
        return len(self.data)
