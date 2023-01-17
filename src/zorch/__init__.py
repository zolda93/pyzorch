from importlib.util import find_spec
import numpy as np

is_cupy = bool(find_spec('cupy'))

if is_cupy:
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    with cp.cuda.Device(0):
        mempool.set_limit(size=(2*1024**3 + (1024**3)//2))

xp = cp if is_cupy else np

xp.set_printoptions(precision=4)

from .autograd import *
from .tensor import*


def tensor(data,requires_grad=False,**kwargs):

    if isinstance(data,Tensor):
        return data
        
    return Tensor(data,requires_grad=requires_grad,**kwargs)


def zeros(*shape,requires_grad=False,**kwargs):
    return Tensor(xp.zeros(*shape,dtype=np.float32),requires_grad=requires_grad,**kwargs)

def ones(*shape,requires_grad=False,**kwargs):
    return Tensor(xp.ones(*shape,dtype=np.float32),requires_grad=requires_grad,**kwargs)

def empty(*shape,requires_grad=False,**kwargs):
    return Tensor(xp.empty(*shape,dtype=np.float32),requires_grad=requires_grad,**kwargs)

def rand(*shape,requires_grad=False,**kwargs):
    return Tensor(xp.random.rand(*shape,dtype=np.float32),requires_grad=False,**kwargs)

def randn(shape,requires_grad=False,**kwargs):
    return Tensor(xp.random.default_rng().standard_normal(size=shape, dtype=np.float32),requires_grad=requires_grad,**kwargs)

def randint(low,high=None,size=None,requires_grad=False,**kwargs):
    return Tensor(xp.random.randint(low,high=high,size=size),requires_grad=requires_grad,**kwargs)

def arange(stop,start=0,requires_grad=False,**kwargs):
    return Tensor(xp.arange(start=start,stop=stop,dtype=np.float32),requires_grad=requiress_grad,**kwargs)

def zeros_like(t,requires_grad=False,**kwargs):
    return zeros(*t.shape,requires_grad=requires_grad,**kwargs)

def ones_like(t,requires_grad=False,**kwargs):
    return ones(*t.shape,requires_grad=requires_grad,**kwargs)

def empty_like(t,requires_grad=False,**kwargs):
    return empty(*t.shape,requires_grad=requires_grad,**kwargs)

def randn_like(t,requires_grad=False,**kwargs):
    return randn(t.shape,requires_grad=requires_grad,**kwargs)
from . import nn
from . import optim


