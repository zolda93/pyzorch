from zorch import*
from . import mlop as ml

from collections.abc import Iterable
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x[:n]
        return tuple(repeat(x, n))

    return parse


single = _ntuple(1)
pair = _ntuple(2)
triple = _ntuple(3)
quadruple = _ntuple(4)


def relu(x,inplace=False):
    return Tensor.make_unaryop(ml._relu,x,inplace)

def relu6(x,inplace=False):
    return Tensor.make_unaryop(ml._relu6,x,inplace)

def tanh(x):
    return Tensor.make_unaryop(ml._tanh,x)

def glu(x,dim=-1):
    return Tensor.make_unaryop(ml._glu,x,dim)

def elu(x,alpha=1.0,inplace=False):
    return Tensor.make_unaryop(ml._elu,x,alpha,inplace)

def selu(x,inplace=False):
    return Tensor.make_unaryop(ml._selu,x.inpalce)

def leakyrelu(x,negative_slope=0.01,inplace=False):
    return Tensor.make_unaryop(ml._leakyrelu,x,negative_slope,inplace)

def sigmoid(x):
    return Tensor.make_unaryop(ml._sigmoid,x)

def logsigmoid(x):
    return Tensor.make_unaryop(ml._logsigmoid,x)

def softmax(x,axis=None):
    return Tensor.make_unaryop(ml._softmax,x,axis)

def logsoftmax(x,axis=None):
    return Tensor.make_unaryop(ml._logsoftmax,x,axis=axis)

def l1_loss(x,target,reduction='mean'):
    return Tensor.make_binaryop(ml._l1loss,x,target,reduction=reduction)

def mse_loss(x,target,reduction='mean'):
    return Tensor.make_binaryop(ml._mseloss,x,target,reduction=reduction)

def binary_cross_entropy(x,target,weight=None,reduction='mean'):
    return Tensor.make_binaryop(ml._bceloss,x,target,weight=weight,reduction=reduction)

def binary_cross_entropy_loss_with_logits(x,target,weight=None,pos_weight=None,reduction='mean'):
    return Tensor.make_binaryop(ml._bcewithlogitsloss,x,target,weight=weight,pos_weight=pos_weight,reduction=reduction)

def nll_loss(x,target,weight=None,ignore_idx=-100,reduction='mean'):
    return Tensor.make_binaryop(ml._nllloss,x,target,weight=weight,ignore_idx=ignore_idx,reduction=reduction)

def cross_entropy_loss(x,target,axis=1,weight=None,ignore_idx=-100,reduction='mean'):
    return nll_loss(logsoftmax(x,axis=axis),target,weight=weight,ignore_idx=ignore_idx,reduction=reduction)

def linear(x,w,bias=None):
    return Tensor.make_binaryop(ml._linear,x,w,bias)

def flatten(x):
    return Tensor.make_unaryop(ml._flatten,x)

def dropout(x,p=0.5,training=True,inplace=False):
    return Tensor.make_unaryop(ml._dropout,x,p,training,inplace)

def maxpool2d(x,kernel_size,stride,padding,return_indices):
    kernel_size = pair(kernel_size)
    stride = pair(stride)
    padding = pair(padding)
    return Tensor.make_unaryop(ml._maxpool2d,x,kernel_size,stride,padding,return_indices)

def avgpool2d(x,kernel_size,stride,padding):
    kernel_size = pair(kernel_size)
    stride = pair(stride)
    padding = pair(padding)
    return Tensor.make_unaryop(ml._avgpool2d,x,kernel_size,stride,padding)

def pad(x,padding,mode='constant',value=0.0):
    padding = pair(padding)
    return Tensor.make_unaryop(ml._pad,x,padding,mode,value)


def conv2d(x,w,bias,stride,padding,dilation,groups):
    stride = pair(stride)
    padding = pair(padding)
    dilation = pair(dilation)
    return Tensor.make_binaryop(ml._conv2d,x,w,bias,stride,padding,dilation,groups)

def conv_transpose2d(x,w,bias,stride,padding,output_padding,groups,dilation):
    stride = pair(stride)
    padding = pair(padding)
    dilation = pair(dilation)
    output_padding = pair(output_padding)
    return Tensor.make_binaryop(ml._convtranspose2d,x,w,bias,stride,padding,output_padding,groups,dilation)


def layer_norm(x,normalized_shape,weight,bias,eps):
    return Tensor.make_binaryop(ml._layernorm,x,weight,normalized_shape,bias,eps)

def batch_norm(x,running_mean,running_var,weight,bias,training,momentum,eps):
    return Tensor.make_binaryop(ml._batchnorm,x,weight,running_mean,running_var,bias,training,momentum,eps)
