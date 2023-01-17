import numpy as np
from zorch.utils.data import Dataset
from zorchvision.transforms import Compose,ToTensor,Normalize
import matplotlib.pyplot as plt
import tarfile
import pickle
import random


class CIFAR100(Dataset):

    def __init__(self,root='CIFAT100',train=True,download=False,
            transforms=Compose([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]),
            target_transforms=None,label_type='fine'):

        assert label_type in ['fine','coarse']
        self.label_type = label_type
        super().__init__(root,train,download,transforms,target_transforms)


    def __len__(self):
        if self.train:
            return 50000
        return 10000


    def state_dict(self):
        if self.label_type == 'fine':
            return{ 'label_map':cifar100_fine_labels}
        elif self.label_type == 'coarse':
            return {'label_map':cifar100_coarse_labels}


    def prepare(self):

        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

        self._download(url,'cifar-100.tar.gz')
        if self.train:
            self.data = self._load_data(self.root+'/cifar-100.tar.gz', 'train')
            self.target = CIFAR100.to_one_hot(self._load_label(self.root+'/cifar-100.tar.gz', 'train'), 100 if self.label_type=='fine' else 20)
        else:
            self.data = self._load_data(self.root+'/cifar-100.tar.gz', 'test')
            self.target = CIFAR100.to_one_hot(self._load_label(self.root+'/cifar-100.tar.gz', 'test'), 100 if self.label_type=='fine' else 20)

        self.data = self.data.reshape(-1,3,32,32)


    def _load_data(self, filename, data_type='train'):

        assert data_type in ['train', 'test']

        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    return np.asarray(data_dict[b'data'])


    def _load_label(self, filename, data_type='train'):

        assert data_type in ['train', 'test']

        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    if self.label_type == 'fine':
                        return np.array(data_dict[b'fine_labels'])
                    elif self.label_type == 'coarse':
                        return np.array(data_dict[b'coarse_labels'])



    def show(self, row=10, col=10):
        H, W = 32, 32
        img = np.zeros((H*row, W*col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255

        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
        plt.show()




cifar100_fine_labels = dict(enumerate([
    'apple',
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm'
]))

cifar100_coarse_labels = dict(enumerate([
    'aquatic mammals',
    'fish',
    'flowers',
    'food containers',
    'fruit and vegetables',
    'household electrical device',
    'household furniture',
    'insects',
    'large carnivores',
    'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores',
    'medium-sized mammals',
    'non-insect invertebrates',
    'people',
    'reptiles',
    'small mammals',
    'trees',
    'vehicles 1',
    'vehicles 2'
]))
