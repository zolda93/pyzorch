import os
import random
from zorch import*

class DataLoader:
    def __init__(self,dataset,batch_size=1,shuffle=True):
        self.dataset = dataset
        self.batch = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset)//self.batch

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))
        return self

    def __next__(self):
        
        if len(self) <= self.idx:
            self.idx = 0
            raise StopIteration

        features,labels = self.dataset[self.index[self.idx*self.batch:(self.idx+1)*self.batch]]
        self.idx += 1
        return features,Tensor(labels)

    def show(self,*args,**kwargs):
        self.dataset.show(*args,**kwargs)
