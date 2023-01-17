import numpy as  np
from zorch.utils.data import Dataset
from zorchvision.transforms import Compose,ToTensor,Normalize
import matplotlib.pyplot as plt
import tarfile
import pickle
import random



class CIFAR10(Dataset):


    def __init__(self,root='CIFAR10',train=True,download=False,
            transforms = Compose([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]),
            target_transforms=None):
        super().__init__(root,train,download,transforms,target_transforms)


    def __len__(self):
        if self.train:
            return 50000
        return 10000

    def state_dict(self):
        return {'label_map':cifar10_labels}

    def prepare(self):

        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        self._download(url,'cifar-10.tar.gz')

        if self.train:
            self.data = np.empty((50000,3*32*32))
            self.target = np.empty((50000,10))
            for i in range(5):
                self.data[i*10000:(i+1)*10000] = self._load_data(self.root+'/cifar-10.tar.gz', i+1, 'train')
                self.target[i*10000:(i+1)*10000] = CIFAR10.to_one_hot(self._load_label(self.root+'/cifar-10.tar.gz', i+1, 'train'), 10)
        else:
            self.data = self._load_data(self.root+'/cifar-10.tar.gz',1, 'test')
            self.target = CIFAR10.to_one_hot(self._load_label(self.root+'/cifar-10.tar.gz',1, 'test'), 10)

        self.data = self.data.reshape(-1,3,32,32)

    def _load_data(self,filename,idx,data_type='train'):

        assert data_type in ['train','test']

        with tarfile.open(filename,'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    return np.asarray(data_dict[b'data'])
                    


    def _load_label(self, filename, idx, data_type='train'):

        assert data_type in ['train', 'test']

        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    return np.array(data_dict[b'labels'])


    def show(self, row=10, col=10):
        H, W = 32, 32
        img = np.zeros((H*row, W*col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255

        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
        plt.show()




cifar10_labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
