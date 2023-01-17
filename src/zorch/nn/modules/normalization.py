from zorch import*
from .module import Module
from .. import functional as F
from .. import init
from ..parameter import Parameter
import numbers


class LayerNorm(Module):
    def __init__(self,normalized_shape,eps=1e-5,elementwise_affine=True):

        super().__init__()

        if isinstance(normalized_shape,numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if self.elementwise_affine:
            self.weight = Parameter(ones((self.normalized_shape)))
            self.bias = Parameter(zeros((self.normalized_shape)))
        else:
            self.register_parameter('weight',None)
            self.register_parameter('bias',None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones(self.weight)
            init.zeros(self.bias)


    def forward(self,x):
        return F.layer_norm(x,self.normalized_shape,self.weight,self.bias,self.eps)

    def extra_repr(self):
        return f"{self.normalized_shape},eps={sef.eps},elementwise_affine={self.elementwise_affine}"
