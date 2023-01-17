import os
import random
import sys
from zorch import*
from zorchvision.transforms import Compose


class Dataset:
    def __init__(self,root,train=True,download=True,transforms=None,target_transforms=None):
        
        print('[*] preparing data...')
        print('this might take few minutes.')
        self.train = train
        self.download=download
        self.transforms = transforms if transforms is not None else Compose()
        self.target_transforms = target_transforms if target_transforms is not None else Compose()
        self.root = root + '/download/{}'.format(self.__class__.__name__.lower())
        self.data = None
        self.labels = None
        self.prepare()
        print('[*] done.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):

        label = self.target_transforms(self.labels[key]) if self.labels is not None else None
        data = self.transforms(self.data[key])
        return data,label

    def show(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def prepare(self):
        pass

    def _download(self,url,filename=None):
        print('[*] start downloading...')

        if not os.path.exists(self.root + '/'):
            os.makedirs(self.root + '/')
        data_dir = self.root

        if filename is None:
            from urllib.parse import urlparse
            parts = urlparse(url)
            filename = op.path.basename(parts.path)

        cache = os.path.join(data_dir,filename)
        if not os.path.exists(cache):
            from urllib.request import urlretrieve
            urlretrieve(url,cache)

        print('[*] downloading 100.00%')

    @staticmethod
    def to_one_hot(labels,num_class):
        one_hot = np.zeros((len(labels),num_class),dtype=np.uint8)
        for c in range(num_class):
            one_hot[:,c][labels==c] = 1
        return one_hot




