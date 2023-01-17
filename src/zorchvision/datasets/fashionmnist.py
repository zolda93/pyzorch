import numpy as np
from zorch.utils.data import Dataset
from zorchvision.transforms import Compose,ToTensor,Normalize
import matplotlib.pyplot as plt
import random
import gzip
import os



class FashionMNIST(Dataset):

    def __init__(self,root='FashionMNIST',train=True,download=False,
            transforms = Compose([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]),
            target_transforms=None):

        super().__init__(root,train,download,transforms,target_transforms)

    def __len__(self):
        if self.train:
            return 60000
        return 10000

    def state_dict(self):
        return {'label_map':fashion_mnist_labels}


    def prepare(self):

        url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        files = {
                'train_data.gz':'train-images-idx3-ubyte.gz',
                'train_labels.gz':'train-labels-idx1-ubyte.gz',
                'test_data.gz':'t10k-images-idx3-ubyte.gz',
                'test_labels.gz':'t10k-labels-idx1-ubyte.gz'
                }

        for filename,value in files.items():
            self._download(url+value,filename)

        if self.train:
            self.data = self._load_data(self.root + '/train_data.gz')
            self.labels = FashionMNIST.to_one_hot(self._load_label(self.root+'/train_labels.gz'), 10)
        else:
            self.data = self._load_data(self.root+'/test_data.gz')
            #self.labels = FashionMNIST.to_one_hot(self._load_label(self.root+'/test_labels.gz'), 10)
            self.labels = self._load_label(self.root+'/test_labels.gz')


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


fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
