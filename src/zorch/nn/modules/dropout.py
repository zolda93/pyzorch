from .module import Module
from .. import functional as F

class DropoutNd(Module):
    def __init__(self,p=0.5,inplace=False):
        super().__init__()

        if p < 0. or p > 1.:
            raise ValueError("dropout probability has to be between 0. and 1., but got {}".foramt(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.p, self.inplace)

class Dropout2d(DropoutNd):
    def forward(self,x):
        return F.dropout(x,p=self.p,training=self.training,inplace=self.inplace)


