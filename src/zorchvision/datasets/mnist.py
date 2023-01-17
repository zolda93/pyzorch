from zorch import np
from zorch.utils.data import Dataset
from zorchvision.transforms import Compose,ToTensor,Normalize
import matplotlib.pyplot as plt
import gzip
import os
import random


class MNIST(Dataset):

    def __init__(self,root='MNIST',train=True,download=True,
            transforms=Compose([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]),
            target_transforms=None):

        super().__init__(root,train,download,transforms,target_transforms)

    def __len__(self):
        if self.train:
            return 60000
        return 10000

    def state_dict(self):
        return {'label_map':mnist_labels}

    def prepare(self):

        url = 'http://yann.lecun.com/exdb/mnist/'

        files = {
                'train_data.gz':'train-images-idx3-ubyte.gz',
                'train_labels.gz':'train-labels-idx1-ubyte.gz',
                'test_data.gz':'t10k-images-idx3-ubyte.gz',
                'test_labels.gz':'t10k-labels-idx1-ubyte.gz'
                }
        if self.download :
            for filename,value in files.items():
                self._download(url + value,filename)

        if self.train:
            data_path = self.root + '/train_data.gz'
            label_path = self.root + '/train_labels.gz'
        else:
            data_path = self.root + '/test_data.gz'
            label_path = self.root + '/test_labels.gz'

        self.data = self._load_data(data_path)
        self.labels = MNIST.to_one_hot(self._load_label(label_path),10)


    def _load_data(self,filename):
        with gzip.open(filename,'rb') as file:
            data = np.frombuffer(file.read(),np.uint8,offset=16)

        return data.reshape(-1,1,28,28)

    def _load_label(self,filename):
        with gzip.open(filename,'rb') as file:
            targets = np.frombuffer(file.read(),np.uint8,offset=8)
        return targets

    def show(self,row=10,col=10):
        H,W = 28,28
        img = np.zeros((H*row,W*col))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(H,W)
        
        plt.imshow(img,cmap='gray',interpolation='nearest')
        plt.axis('off')
        plt.show()




mnist_labels = {
    0: '0', 
    1: '1', 
    2: '2', 
    3: '3', 
    4: '4', 
    5: '5', 
    6: '6', 
    7: '7', 
    8: '8', 
    9: '9'
}
