from .module import Module
from .. import functional as F

class MaxPoolNd(Module):
    def __init__(self,kernel_size,stride=None,padding=0,return_indices=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.return_indices = return_indices

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}'.format(**self.__dict__)


class MaxPool2d(MaxPoolNd):
    def forward(self,x):
        return F.maxpool2d(x,self.kernel_size,self.stride,self.padding,self.return_indices)



class AvgPoolNd(Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}'.format(**self.__dict__)

class AvgPool2d(AvgPoolNd):
    def forward(self,x):
        return F.avgpool2d(x,self.kernel_size,self.stride,self.padding)
