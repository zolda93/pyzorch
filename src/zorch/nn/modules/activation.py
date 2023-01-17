from .module import Module
from ..import functional as F

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return F.sigmoid(x)

class LogSigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return F.logsigmoid

class ReLU(Module):
    def __init__(self,inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self,x):
        return F.relu(x,inplace=self.inplace)

    def extra_repr(self):
        inplace_str = ", inplace=True" if self.inplace else ""
        return inplace_str

class ReLU6(Module):
    def __init__(self,inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self,x):
        return F.relu6(x,inplace=self.inplace)

    def extra_repr(self):
        inplace_str = ", inpalce=True" if self.inplace else ""
        return inplace_str

class GLU(Module):
    def __init__(self,dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self,x):
        return F.glu(x,dim=self.dim)

    def extra_repr(self):
        return f", dim={self.dim}"

class SELU(Module):
    def __init__(self,inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self,x):
        return F.selu(x,inplace=self.inplace)

    def extra_repr(self):
        inplace_str = ", inplace=True" if self.inplace else ""
        return inplace_str

class ELU(Module):
    def __init__(self,alpha=1.0,inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self,x):
        return F.elu(x,alpha=self.alpha,inplace=self.inplace)

    def extra_repr(self):
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"alpha ={self.alpha},{inplace_str}"

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return F.tanh(x)

class LeakyReLU(Module):
    def __init__(self,negative_slope=0.01,inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace 

    def forward(self,x):
        return F.leakyrelu(x,negative_slope=self.negative_slope,inplace=self.inplace)

    def extra_repr(self):
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"negative_slope={self.negative_slope},{inplace_str}"

class Softmax(Module):
    def __init__(self,axis=None):
        super().__init__()
        self.axis = axis

    def forward(self,x):
        return F.softmax(x,axis=self.axis)

    def extra_repr(self):
        return f"axis={self.axis}"

class LogSoftmax(Module):
    def __init__(self,axis=None):
        super().__init__()
        self.axis = axis

    def forward(self,x):
        return F.logsoftmax(x,axis=self.axis)

    def extra_repr(self):
        return f"axis={self.axis}"

