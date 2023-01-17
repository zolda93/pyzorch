import numpy as np
from zorchvision.transforms import Compose,ToTensor,Normalize
from zorch.utils.data import Dataset
import matplotlib.pyplot as plt
import tarfile
import random


class STL10(Dataset):

    def __init__(self,root='STL10',train=True,download=False,
            transforms=Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]),
            target_transforms=None):

        super().__init__(root,train,download,transforms,target_transforms)


    def __len__(self):
        if self.train:
            return 5000
        return 8000

    def state_dict(self):
        return {'label_map': stl10_labels}


    def prepare(self):

        url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz' 

        self._download(url,'stl10.tar.gz')
        if self.train:
            self.data = self._load_data(self.root+'/stl10_binary/train_X.bin')
            self.target = STL10.to_one_hot(self._load_label(self.root+'/stl10_binary/train_y.bin'), 10)
        else:
            self.data = self._load_data(self.root+'/stl10_binary/test_X.bin')
            self.target = STL10.to_one_hot(self._load_label(self.root+'/stl10_binary/test_y.bin'), 10)

        self.data = self.data.reshape(-1,3,96,96)



    def _load_data(self, filename):
        with open(filename, 'rb') as file:
            return np.asarray(numpy.fromfile(file, numpy.uint8))


    def _load_label(self, filename):
        with open(filename, 'rb') as file:
            targets = np.asarray(numpy.fromfile(file, numpy.uint8))
        return targets-1


    def show(self, row=5, col=5):
        H, W = 96, 96
        img = np.zeros((H*row, W*col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255

        plt.imshow(img, interpolation='nearest') 
        plt.axis('off')
        plt.show()


stl10_labels = {
    0: 'airplane',
    1: 'bird',
    2: 'car',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'horse',
    7: 'monkey',
    8: 'ship',
    9: 'truck',
}
