import torch
import torch.utils
import torch.utils.data

import numpy as np

import random

class DemoDataset(torch.utils.data.IterableDataset):
    def __init__(self, V, l, max_seq_length=100):
        self.V = V
        self.l = l

        self.max_seq_length = max_seq_length

        self.c = 0

    def __len__(self):
        return self.l
    
    def __iter__(self):
        if self.c >= self.l:
            raise StopIteration
        
        self.c += 1
        return iter([(torch.from_numpy(np.random.randint(1, self.V, size=(self.max_seq_length))), 
                      torch.from_numpy(np.random.randint(1, self.V, size=(self.max_seq_length)))) for x in [0]*self.l])
    

class DemoDataloader(torch.utils.data.DataLoader):
    def __init__(self, batch_size, V=11, l=10000, max_seq_length=100):
        super(DemoDataloader, self).__init__(DemoDataset(V, l, max_seq_length=max_seq_length), batch_size)